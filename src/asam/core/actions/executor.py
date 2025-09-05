"""
Action Execution System for ASAM

Executes actions based on detection results (screen lock, notifications, etc.)
"""

import asyncio
import logging
import subprocess
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Optional

from ..detection.types import ActionType, AggregatedResult

logger = logging.getLogger(__name__)


class ActionExecutor(ABC):
    """Abstract base class for action executors"""
    
    @abstractmethod
    async def execute(self, action: ActionType, result: AggregatedResult) -> bool:
        """Execute the specified action"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this executor is available on current platform"""
        pass


class ScreenLockExecutor(ActionExecutor):
    """Cross-platform screen locking"""
    
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.lock_timeout = config.get("lock_timeout_seconds", 300)  # 5 minutes
        self.last_lock_time: Optional[datetime] = None
        self.min_lock_interval = config.get("min_lock_interval_seconds", 30)
    
    async def execute(self, action: ActionType, result: AggregatedResult) -> bool:
        """Lock screen if action requires it"""
        if action != ActionType.BLOCK:
            return True
        
        # Rate limiting - don't lock too frequently
        now = datetime.now()
        if (self.last_lock_time and 
            now - self.last_lock_time < timedelta(seconds=self.min_lock_interval)):
            logger.info("Screen lock rate limited")
            return True
        
        try:
            success = await self._lock_screen()
            if success:
                self.last_lock_time = now
                logger.info(f"Screen locked due to {result.primary_category.value} content "
                           f"(confidence: {result.overall_confidence:.2f})")
            return success
        except Exception as e:
            logger.error(f"Failed to lock screen: {e}")
            return False
    
    async def _lock_screen(self) -> bool:
        """Platform-specific screen locking implementation"""
        import platform
        system = platform.system()
        
        try:
            if system == "Darwin":  # macOS
                # Use macOS screensaver command
                result = subprocess.run(
                    ["/System/Library/CoreServices/Menu Extras/User.menu/Contents/Resources/CGSession", "-suspend"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                return result.returncode == 0
            
            elif system == "Windows":
                # Use Windows rundll32 command
                result = subprocess.run(
                    ["rundll32.exe", "user32.dll,LockWorkStation"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                return result.returncode == 0
            
            elif system == "Linux":
                # Try common Linux screen lockers
                for command in [
                    ["xdg-screensaver", "lock"],
                    ["gnome-screensaver-command", "--lock"],
                    ["xscreensaver-command", "-lock"],
                    ["i3lock", "-n"]
                ]:
                    try:
                        result = subprocess.run(command, capture_output=True, timeout=5)
                        if result.returncode == 0:
                            return True
                    except (subprocess.TimeoutExpired, FileNotFoundError):
                        continue
                return False
            
            else:
                logger.warning(f"Screen locking not implemented for {system}")
                return False
                
        except Exception as e:
            logger.error(f"Screen lock command failed: {e}")
            return False
    
    def is_available(self) -> bool:
        """Screen locking is available on most platforms"""
        import platform
        return platform.system() in ["Darwin", "Windows", "Linux"]


class NotificationExecutor(ActionExecutor):
    """Cross-platform notifications"""
    
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.notification_enabled = config.get("notifications_enabled", True)
        self.last_notification_time: Optional[datetime] = None
        self.min_notification_interval = config.get("min_notification_interval_seconds", 10)
    
    async def execute(self, action: ActionType, result: AggregatedResult) -> bool:
        """Send notification based on action type"""
        if not self.notification_enabled or action == ActionType.LOG_ONLY:
            return True
        
        # Rate limiting
        now = datetime.now()
        if (self.last_notification_time and 
            now - self.last_notification_time < timedelta(seconds=self.min_notification_interval)):
            return True
        
        try:
            message = self._create_notification_message(action, result)
            success = await self._send_notification("ASAM Alert", message, action)
            if success:
                self.last_notification_time = now
            return success
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            return False
    
    def _create_notification_message(self, action: ActionType, result: AggregatedResult) -> str:
        """Create notification message based on detection result"""
        category = result.primary_category.value.replace('_', ' ').title()
        confidence = int(result.overall_confidence * 100)
        
        if action == ActionType.BLOCK:
            return f"🚫 Screen locked - {category} detected ({confidence}% confidence)"
        elif action == ActionType.WARN:
            return f"⚠️ Entertainment content detected - {category} ({confidence}% confidence)"
        else:
            return f"ℹ️ Activity logged - {category} ({confidence}% confidence)"
    
    async def _send_notification(self, title: str, message: str, action: ActionType) -> bool:
        """Platform-specific notification implementation"""
        import platform
        system = platform.system()
        
        try:
            if system == "Darwin":  # macOS
                # Use macOS native notifications
                return await self._send_macos_notification(title, message, action)
            
            elif system == "Windows":
                # Use Windows toast notifications
                return await self._send_windows_notification(title, message, action)
            
            elif system == "Linux":
                # Use notify-send
                return await self._send_linux_notification(title, message, action)
            
            else:
                logger.warning(f"Notifications not implemented for {system}")
                return False
                
        except Exception as e:
            logger.error(f"Notification failed: {e}")
            return False
    
    async def _send_macos_notification(self, title: str, message: str, action: ActionType) -> bool:
        """Send macOS notification"""
        try:
            # Try terminal-notifier first (if installed)
            result = subprocess.run([
                "terminal-notifier",
                "-title", title,
                "-message", message,
                "-timeout", "10"
            ], capture_output=True, timeout=5)
            
            if result.returncode == 0:
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Fallback to osascript
        try:
            script = f'''display notification "{message}" with title "{title}"'''
            result = subprocess.run([
                "osascript", "-e", script
            ], capture_output=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False
    
    async def _send_windows_notification(self, title: str, message: str, action: ActionType) -> bool:
        """Send Windows notification"""
        try:
            # Use plyer library if available
            import plyer
            plyer.notification.notify(
                title=title,
                message=message,
                timeout=10
            )
            return True
        except ImportError:
            # Fallback to PowerShell
            try:
                ps_script = f'''
                [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
                $template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
                $template.SelectSingleNode("//text[@id='1']").InnerText = "{title}"
                $template.SelectSingleNode("//text[@id='2']").InnerText = "{message}"
                $toast = [Windows.UI.Notifications.ToastNotification]::new($template)
                [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("ASAM").Show($toast)
                '''
                result = subprocess.run([
                    "powershell", "-Command", ps_script
                ], capture_output=True, timeout=10)
                return result.returncode == 0
            except Exception:
                return False
    
    async def _send_linux_notification(self, title: str, message: str, action: ActionType) -> bool:
        """Send Linux notification"""
        try:
            # Determine urgency level
            urgency = "critical" if action == ActionType.BLOCK else "normal"
            
            result = subprocess.run([
                "notify-send",
                "--urgency", urgency,
                "--expire-time", "10000",
                title,
                message
            ], capture_output=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False
    
    def is_available(self) -> bool:
        """Notifications are available on most platforms"""
        import platform
        return platform.system() in ["Darwin", "Windows", "Linux"]


class LoggingExecutor(ActionExecutor):
    """Logs all actions and results"""
    
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.detailed_logging = config.get("detailed_logging", True)
    
    async def execute(self, action: ActionType, result: AggregatedResult) -> bool:
        """Log the detection result and action"""
        try:
            if self.detailed_logging:
                # Detailed log with all analyzer results
                analyzer_details = []
                for detection in result.individual_results:
                    analyzer_details.append(
                        f"{detection.analyzer_type.value}:{detection.confidence:.2f}"
                    )
                
                logger.info(
                    f"Action: {action.value} | "
                    f"Category: {result.primary_category.value} | "
                    f"Confidence: {result.overall_confidence:.2f} | "
                    f"Analyzers: [{', '.join(analyzer_details)}] | "
                    f"Duration: {result.analysis_duration_ms}ms"
                )
            else:
                # Simple log
                logger.info(
                    f"Action: {action.value} | "
                    f"Category: {result.primary_category.value} | "
                    f"Confidence: {result.overall_confidence:.2f}"
                )
            
            return True
        except Exception as e:
            logger.error(f"Failed to log action: {e}")
            return False
    
    def is_available(self) -> bool:
        """Logging is always available"""
        return True


class ActionExecutionManager:
    """Manages multiple action executors and coordination"""
    
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.executors: list[ActionExecutor] = []
        self.action_history: list[tuple[datetime, ActionType, AggregatedResult]] = []
        self.max_history = config.get("action_history_size", 100)
        self._setup_executors()
    
    def _setup_executors(self):
        """Initialize available action executors"""
        # Always include logging
        self.executors.append(LoggingExecutor(self.config))
        
        # Add screen lock executor if available
        screen_lock = ScreenLockExecutor(self.config)
        if screen_lock.is_available():
            self.executors.append(screen_lock)
        else:
            logger.warning("Screen lock executor not available")
        
        # Add notification executor if available
        notification = NotificationExecutor(self.config)
        if notification.is_available():
            self.executors.append(notification)
        else:
            logger.warning("Notification executor not available")
        
        logger.info(f"Initialized {len(self.executors)} action executors")
    
    async def execute_action(self, action: ActionType, result: AggregatedResult) -> bool:
        """Execute action using all available executors"""
        now = datetime.now()
        success_count = 0
        
        # Execute action with all executors
        for executor in self.executors:
            try:
                success = await executor.execute(action, result)
                if success:
                    success_count += 1
                else:
                    logger.warning(f"Executor {type(executor).__name__} failed")
            except Exception as e:
                logger.error(f"Executor {type(executor).__name__} error: {e}")
        
        # Record in history
        self.action_history.append((now, action, result))
        if len(self.action_history) > self.max_history:
            self.action_history.pop(0)
        
        # Consider successful if at least one executor succeeded
        return success_count > 0
    
    def get_recent_actions(self, hours: int = 1) -> list[tuple[datetime, ActionType, AggregatedResult]]:
        """Get actions from the last N hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [(timestamp, action, result) for timestamp, action, result in self.action_history 
                if timestamp > cutoff]
    
    def get_action_stats(self) -> dict[str, Any]:
        """Get statistics about recent actions"""
        recent_actions = self.get_recent_actions(24)  # Last 24 hours
        
        stats = {
            "total_actions": len(recent_actions),
            "blocks": len([a for _, action, _ in recent_actions if action == ActionType.BLOCK]),
            "warnings": len([a for _, action, _ in recent_actions if action == ActionType.WARN]),
            "allows": len([a for _, action, _ in recent_actions if action == ActionType.ALLOW]),
            "executors_available": len(self.executors),
            "executor_types": [type(e).__name__ for e in self.executors]
        }
        
        return stats
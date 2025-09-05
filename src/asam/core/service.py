"""
ASAM Core Service

Main service orchestrator for the Advanced Screen Activity Monitor.
Integrates detection pipeline with action execution and monitoring loops.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Optional

from .detection import DetectionEngine
from .detection.types import ActionType, AggregatedResult, ScreenCapture, TextContent

logger = logging.getLogger(__name__)


class ASAMService:
    """Main ASAM service orchestrator with integrated detection pipeline"""

    def __init__(self, config_manager=None):
        """Initialize the ASAM service"""
        self.config_manager = config_manager
        self.running = False
        self.logger = logger

        # Core components
        self.detection_engine: Optional[DetectionEngine] = None

        # Service configuration
        self.analysis_interval = 5.0  # Seconds between analyses
        self.max_consecutive_blocks = 3  # Max blocks before action

        # State tracking
        self.consecutive_entertainment_detections = 0
        self.last_analysis_time: Optional[datetime] = None
        self.last_result: Optional[AggregatedResult] = None

        # Monitoring task
        self.monitoring_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the ASAM service with full detection pipeline"""
        if self.running:
            self.logger.warning("Service already running")
            return

        self.logger.info("ASAM Service starting...")

        try:
            # Load configuration
            if self.config_manager:
                config = await self.config_manager.load_config()
            else:
                config = self._get_default_config()

            # Initialize detection engine
            detection_config = config.get("detection", {})
            self.detection_engine = DetectionEngine(detection_config)

            if not await self.detection_engine.initialize():
                raise Exception("Failed to initialize detection engine")

            # Update service configuration from config
            self.analysis_interval = config.get("analysis_interval", 5.0)
            self.max_consecutive_blocks = config.get("max_consecutive_blocks", 3)

            # Start monitoring loop
            self.running = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())

            self.logger.info("ASAM Service started successfully")

        except Exception as e:
            self.logger.error(f"Failed to start ASAM service: {e}")
            await self.stop()
            raise

    async def stop(self) -> None:
        """Stop the ASAM service"""
        if not self.running:
            self.logger.warning("Service not running")
            return

        self.logger.info("ASAM Service stopping...")
        self.running = False

        # Stop monitoring loop
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None

        # Cleanup detection engine
        if self.detection_engine:
            await self.detection_engine.cleanup()
            self.detection_engine = None

        # Reset state
        self.consecutive_entertainment_detections = 0
        self.last_analysis_time = None
        self.last_result = None

        self.logger.info("ASAM Service stopped")

    def is_running(self) -> bool:
        """Check if service is running"""
        return self.running

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop that performs periodic analysis"""
        self.logger.info("Starting monitoring loop")

        while self.running:
            try:
                await self._perform_analysis_cycle()
                await asyncio.sleep(self.analysis_interval)

            except asyncio.CancelledError:
                self.logger.info("Monitoring loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(1.0)  # Brief pause before retry

    async def _perform_analysis_cycle(self) -> None:
        """Perform a single analysis cycle"""
        if not self.detection_engine:
            return

        analysis_start = datetime.now()
        self.logger.debug("Starting analysis cycle")

        try:
            # Collect data for analysis
            screen_capture = await self._capture_screen()
            text_content = await self._extract_text_content()

            # Perform detection analysis
            result = await self.detection_engine.analyze_screen_content(
                screen_capture=screen_capture, text_content=text_content
            )

            # Process the result
            await self._process_analysis_result(result)

            # Update state
            self.last_analysis_time = analysis_start
            self.last_result = result

            self.logger.debug(
                f"Analysis cycle complete: {result.primary_category.value} "
                f"(confidence: {result.overall_confidence:.3f}, "
                f"action: {result.recommended_action.value})"
            )

        except Exception as e:
            self.logger.error(f"Analysis cycle failed: {e}")

    async def _capture_screen(self) -> Optional[ScreenCapture]:
        """Capture current screen content"""
        # Placeholder for screen capture implementation
        # This would use platform-specific screen capture APIs
        try:
            # TODO: Implement actual screen capture
            # For now, return None to skip vision analysis
            return None
        except Exception as e:
            self.logger.error(f"Screen capture failed: {e}")
            return None

    async def _extract_text_content(self) -> Optional[TextContent]:
        """Extract text content from various sources"""
        # Placeholder for text content extraction
        # This would integrate with browser extension, OCR, clipboard monitoring, etc.
        try:
            # TODO: Implement actual text extraction
            # For now, return None to skip text analysis
            return None
        except Exception as e:
            self.logger.error(f"Text extraction failed: {e}")
            return None

    async def _process_analysis_result(self, result: AggregatedResult) -> None:
        """Process analysis result and take appropriate action"""
        self.logger.info(
            f"Detection result: {result.primary_category.value} "
            f"(confidence: {result.overall_confidence:.3f}, "
            f"action: {result.recommended_action.value})"
        )

        # Track consecutive entertainment detections
        if result.is_entertainment and result.overall_confidence > 0.5:
            self.consecutive_entertainment_detections += 1
        else:
            self.consecutive_entertainment_detections = 0

        # Take action based on recommendation
        await self._execute_action(result)

    async def _execute_action(self, result: AggregatedResult) -> None:
        """Execute the recommended action"""
        action = result.recommended_action

        if action == ActionType.LOG_ONLY:
            self._log_detection(result)

        elif action == ActionType.WARN:
            await self._show_warning(result)

        elif action == ActionType.BLOCK:
            await self._execute_block_action(result)

        # Always log for monitoring
        self._log_detection(result)

    def _log_detection(self, result: AggregatedResult) -> None:
        """Log detection result"""
        self.logger.info(
            f"Content detected: {result.primary_category.value} "
            f"(confidence: {result.overall_confidence:.3f}, "
            f"analyzers: {len(result.individual_results)}, "
            f"duration: {result.analysis_duration_ms}ms)"
        )

    async def _show_warning(self, result: AggregatedResult) -> None:
        """Show warning to user"""
        # Placeholder for warning implementation
        # This would show system notifications, popups, etc.
        self.logger.warning(
            f"Entertainment content detected: {result.primary_category.value} "
            f"(confidence: {result.overall_confidence:.3f})"
        )

        # TODO: Implement actual warning system
        # - System notifications
        # - Warning dialog
        # - Status bar indicator

    async def _execute_block_action(self, result: AggregatedResult) -> None:
        """Execute blocking action"""
        if self.consecutive_entertainment_detections >= self.max_consecutive_blocks:
            self.logger.critical(
                f"Maximum entertainment detections reached ({self.consecutive_entertainment_detections}). "
                f"Executing block action for: {result.primary_category.value}"
            )

            # TODO: Implement actual blocking actions
            # - Lock screen
            # - Close applications
            # - Block network access
            # - Send notifications

            # For now, just log the action
            await self._show_warning(result)
        else:
            # Just warn for now
            await self._show_warning(result)

    def _get_default_config(self) -> dict[str, Any]:
        """Get default service configuration"""
        return {
            "analysis_interval": 5.0,
            "max_consecutive_blocks": 3,
            "detection": {
                "analysis_timeout": 30.0,
                "parallel_analysis": True,
                "analyzers": {
                    "text": {
                        "enabled": True,
                        "ollama_url": "http://localhost:11434",
                        "model_name": "llama3.2:3b",
                        "confidence_threshold": 0.75,
                    },
                    "vision": {
                        "enabled": True,
                        "confidence_threshold": 0.6,
                    },
                    "process": {
                        "enabled": True,
                        "confidence_threshold": 0.7,
                    },
                    "network": {
                        "enabled": True,
                        "confidence_threshold": 0.5,
                    },
                },
                "aggregator": {
                    "block_threshold": 0.85,
                    "warn_threshold": 0.65,
                    "log_threshold": 0.45,
                    "analyzer_weights": {
                        "text": 0.4,
                        "vision": 0.25,
                        "process": 0.25,
                        "network": 0.1,
                    },
                },
            },
        }

    def get_status(self) -> dict[str, Any]:
        """Get current service status"""
        status = {
            "running": self.running,
            "analysis_interval": self.analysis_interval,
            "consecutive_detections": self.consecutive_entertainment_detections,
            "last_analysis": (
                self.last_analysis_time.isoformat() if self.last_analysis_time else None
            ),
            "detection_engine": None,
        }

        if self.detection_engine:
            status["detection_engine"] = self.detection_engine.get_status()

        if self.last_result:
            status["last_result"] = {
                "category": self.last_result.primary_category.value,
                "confidence": self.last_result.overall_confidence,
                "action": self.last_result.recommended_action.value,
                "duration_ms": self.last_result.analysis_duration_ms,
            }

        return status

    async def force_analysis(self) -> Optional[AggregatedResult]:
        """Force an immediate analysis cycle (for testing)"""
        if not self.detection_engine:
            return None

        self.logger.info("Forcing immediate analysis")
        await self._perform_analysis_cycle()
        return self.last_result

    async def test_analyzers(self) -> dict[str, Any]:
        """Test all analyzers (for diagnostics)"""
        if not self.detection_engine:
            return {"error": "Detection engine not initialized"}

        return await self.detection_engine.test_analyzers()

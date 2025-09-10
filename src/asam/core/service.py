"""
ASAM Core Service

Main service orchestrator for the Advanced Screen Activity Monitor.
Integrates detection pipeline with action execution and monitoring loops.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .actions import ActionExecutionManager
from .capture import ScreenCaptureManager
from .config import AsamConfig, ConfigValidator
from .detection import DetectionEngine
from .detection.types import AggregatedResult, ScreenCapture, TextContent

logger = logging.getLogger(__name__)


class AsamService:
    """Main ASAM service orchestrator with integrated detection pipeline"""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the ASAM service"""
        self.config_path = config_path
        self.is_running = False
        self.logger = logger

        # Configuration
        self.config: Optional[AsamConfig] = None
        self.config_validator = ConfigValidator()

        # Core components
        self.detection_engine: Optional[DetectionEngine] = None
        self.capture_manager: Optional[ScreenCaptureManager] = None
        self.action_manager: Optional[ActionExecutionManager] = None

        # State tracking
        self.consecutive_entertainment_detections = 0
        self.last_analysis_time: Optional[datetime] = None
        self.last_result: Optional[AggregatedResult] = None

        # Monitoring task
        self.monitoring_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Initialize the ASAM service components"""
        self.logger.info("Initializing ASAM Service...")

        try:
            # Load and validate configuration
            await self._load_configuration()

            # Initialize core components
            await self._initialize_components()

            self.logger.info("ASAM Service initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize ASAM service: {e}")
            raise

    async def start(self) -> None:
        """Start the ASAM service monitoring loop"""
        if self.is_running:
            self.logger.warning("Service already running")
            return

        if not self.config:
            await self.initialize()

        self.logger.info("ASAM Service starting...")

        try:
            # Start monitoring loop
            self.is_running = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())

            self.logger.info("ASAM Service started successfully")

        except Exception as e:
            self.logger.error(f"Failed to start ASAM service: {e}")
            await self.stop()
            raise

    async def stop(self) -> None:
        """Stop the ASAM service"""
        if not self.is_running:
            self.logger.warning("Service not running")
            return

        self.logger.info("ASAM Service stopping...")
        self.is_running = False

        # Stop monitoring loop
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None

        self.logger.info("ASAM Service stopped")

    async def cleanup(self) -> None:
        """Cleanup all service components"""
        self.logger.info("ASAM Service cleanup...")

        # Stop service if running
        if self.is_running:
            await self.stop()

        # Cleanup detection engine
        if self.detection_engine:
            await self.detection_engine.cleanup()
            self.detection_engine = None

        # Reset state
        self.consecutive_entertainment_detections = 0
        self.last_analysis_time = None
        self.last_result = None
        self.config = None

        self.logger.info("ASAM Service cleanup complete")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop that performs periodic analysis"""
        self.logger.info("Starting monitoring loop")

        while self.is_running:
            try:
                await self._perform_analysis_cycle()
                await asyncio.sleep(self.config.detection.analysis_interval_seconds)

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

            # Process the result and execute actions
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

    async def _load_configuration(self) -> None:
        """Load and validate configuration"""
        try:
            if self.config_path and self.config_path.exists():
                self.config, warnings = self.config_validator.load_and_validate(
                    self.config_path
                )

                # Log configuration warnings
                for warning in warnings:
                    self.logger.warning(f"Configuration warning: {warning}")
            else:
                self.logger.info("Using default configuration")
                self.config = self.config_validator.create_default_config()

        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise

    async def _initialize_components(self) -> None:
        """Initialize core service components"""
        try:
            # Initialize detection engine
            detection_config = self.config.dict()
            self.detection_engine = DetectionEngine(detection_config)
            await self.detection_engine.initialize()

            # Initialize screen capture manager
            capture_config = self.config.screen_capture.dict()
            self.capture_manager = ScreenCaptureManager(capture_config)

            # Initialize action execution manager
            action_config = self.config.actions.dict()
            self.action_manager = ActionExecutionManager(action_config)

            self.logger.info("Core components initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise

    async def _capture_screen(self) -> Optional[ScreenCapture]:
        """Capture current screen content"""
        if not self.capture_manager:
            return None

        try:
            return await self.capture_manager.capture_screen()
        except Exception as e:
            self.logger.error(f"Screen capture failed: {e}")
            return None

    async def _extract_text_content(self) -> Optional[TextContent]:
        """Extract text content from various sources"""
        try:
            # TODO: Implement text extraction from:
            # - Browser extension
            # - OCR from screen capture
            # - Clipboard monitoring
            # - Active window title
            return None
        except Exception as e:
            self.logger.error(f"Text extraction failed: {e}")
            return None

    async def _process_analysis_result(self, result: AggregatedResult) -> None:
        """Process analysis result and execute appropriate actions"""
        self.logger.info(
            f"Detection result: {result.primary_category.value} "
            f"(confidence: {result.overall_confidence:.3f}, "
            f"action: {result.recommended_action.value})"
        )

        # Track consecutive entertainment detections
        if (
            result.is_entertainment
            and result.overall_confidence > self.config.detection.confidence_threshold
        ):
            self.consecutive_entertainment_detections += 1
        else:
            self.consecutive_entertainment_detections = 0

        # Execute actions using action manager
        if self.action_manager:
            try:
                success = await self.action_manager.execute_action(
                    result.recommended_action, result
                )
                if not success:
                    self.logger.warning(
                        f"Action execution failed for: {result.recommended_action.value}"
                    )
            except Exception as e:
                self.logger.error(f"Error executing action: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get current service status"""
        status = {
            "running": self.is_running,
            "consecutive_detections": self.consecutive_entertainment_detections,
            "last_analysis": (
                self.last_analysis_time.isoformat() if self.last_analysis_time else None
            ),
            "config_path": str(self.config_path) if self.config_path else None,
            "components": {
                "detection_engine": self.detection_engine is not None,
                "capture_manager": self.capture_manager is not None,
                "action_manager": self.action_manager is not None,
            },
        }

        if self.config:
            status["analysis_interval"] = (
                self.config.detection.analysis_interval_seconds
            )
            status["confidence_threshold"] = self.config.detection.confidence_threshold

        if self.detection_engine:
            try:
                status["detection_engine_status"] = self.detection_engine.get_status()
            except Exception as e:
                status["detection_engine_error"] = str(e)

        if self.action_manager:
            try:
                status["action_stats"] = self.action_manager.get_action_stats()
            except Exception as e:
                status["action_stats_error"] = str(e)

        if self.capture_manager:
            try:
                # Non-blocking screen info (may fail)
                status["screen_available"] = True
            except Exception:
                status["screen_available"] = False

        if self.last_result:
            status["last_result"] = {
                "category": self.last_result.primary_category.value,
                "confidence": self.last_result.overall_confidence,
                "action": self.last_result.recommended_action.value,
                "duration_ms": self.last_result.analysis_duration_ms,
                "analyzers_count": len(self.last_result.individual_results),
            }

        return status

    async def force_analysis(self) -> Optional[AggregatedResult]:
        """Force an immediate analysis cycle (for testing)"""
        if not self.detection_engine:
            return None

        self.logger.info("Forcing immediate analysis")
        await self._perform_analysis_cycle()
        return self.last_result

    async def get_screen_info(self) -> dict[str, Any]:
        """Get screen capture information"""
        if not self.capture_manager:
            return {"error": "Screen capture manager not initialized"}

        try:
            return await self.capture_manager.get_screen_info()
        except Exception as e:
            return {"error": f"Failed to get screen info: {e}"}

    async def test_components(self) -> dict[str, Any]:
        """Test all components (for diagnostics)"""
        results = {
            "detection_engine": "not_initialized",
            "capture_manager": "not_initialized",
            "action_manager": "not_initialized",
            "overall_status": "unknown",
        }

        # Test detection engine
        if self.detection_engine:
            try:
                engine_status = self.detection_engine.get_status()
                results["detection_engine"] = "ok" if engine_status else "error"
            except Exception as e:
                results["detection_engine"] = f"error: {e}"

        # Test capture manager
        if self.capture_manager:
            try:
                screen_info = await self.capture_manager.get_screen_info()
                results["capture_manager"] = (
                    "ok"
                    if "error" not in screen_info
                    else f"error: {screen_info.get('error')}"
                )
            except Exception as e:
                results["capture_manager"] = f"error: {e}"

        # Test action manager
        if self.action_manager:
            try:
                action_stats = self.action_manager.get_action_stats()
                results["action_manager"] = "ok" if action_stats else "error"
            except Exception as e:
                results["action_manager"] = f"error: {e}"

        # Overall status
        component_statuses = [v for k, v in results.items() if k != "overall_status"]
        if all("ok" in status for status in component_statuses):
            results["overall_status"] = "healthy"
        elif any("error" in status for status in component_statuses):
            results["overall_status"] = "degraded"
        else:
            results["overall_status"] = "not_ready"

        return results

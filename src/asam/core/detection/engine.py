"""
Detection Engine

Main orchestrator for the multi-layer detection pipeline.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Optional

from .aggregator import ConfidenceAggregator
from .analyzers import (
    AnalyzerBase,
    NetworkAnalyzer,
    ProcessAnalyzer,
    TextAnalyzer,
    VisionAnalyzer,
)
from .rules import AdvancedDetectionRulesEngine
from .types import AggregatedResult, DetectionResult, ScreenCapture, TextContent


class DetectionEngine:
    """Main detection engine that orchestrates all analyzers"""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Engine configuration
        self.analysis_timeout = self.config.get("analysis_timeout", 30.0)
        self.parallel_analysis = self.config.get("parallel_analysis", True)
        self.enable_fallback = self.config.get("enable_fallback", True)

        # Initialize components
        self.analyzers: dict[str, AnalyzerBase] = {}
        self.aggregator = ConfidenceAggregator(self.config.get("aggregator", {}))
        self.rules_engine = AdvancedDetectionRulesEngine(self.config.get("rules", {}))

        # Engine state
        self.initialized = False
        self.running = False
        self._analysis_stats = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "avg_duration_ms": 0.0,
        }

    async def initialize(self) -> bool:
        """Initialize all detection components"""
        if self.initialized:
            return True

        self.logger.info("Initializing detection engine...")

        try:
            # Initialize analyzers based on configuration
            await self._initialize_analyzers()

            self.initialized = True
            self.logger.info("Detection engine initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize detection engine: {e}")
            return False

    async def _initialize_analyzers(self) -> None:
        """Initialize all configured analyzers"""
        analyzer_configs = self.config.get("analyzers", {})

        # Text analyzer (LLM-based)
        if analyzer_configs.get("text", {}).get("enabled", True):
            text_analyzer = TextAnalyzer(analyzer_configs.get("text", {}))
            if await text_analyzer.initialize():
                self.analyzers["text"] = text_analyzer
                self.logger.info("Text analyzer initialized")
            else:
                self.logger.warning("Text analyzer initialization failed")

        # Vision analyzer (computer vision)
        if analyzer_configs.get("vision", {}).get("enabled", True):
            vision_analyzer = VisionAnalyzer(analyzer_configs.get("vision", {}))
            if await vision_analyzer.initialize():
                self.analyzers["vision"] = vision_analyzer
                self.logger.info("Vision analyzer initialized")
            else:
                self.logger.warning("Vision analyzer initialization failed")

        # Process analyzer
        if analyzer_configs.get("process", {}).get("enabled", True):
            process_analyzer = ProcessAnalyzer(analyzer_configs.get("process", {}))
            if await process_analyzer.initialize():
                self.analyzers["process"] = process_analyzer
                self.logger.info("Process analyzer initialized")
            else:
                self.logger.warning("Process analyzer initialization failed")

        # Network analyzer
        if analyzer_configs.get("network", {}).get("enabled", True):
            network_analyzer = NetworkAnalyzer(analyzer_configs.get("network", {}))
            if await network_analyzer.initialize():
                self.analyzers["network"] = network_analyzer
                self.logger.info("Network analyzer initialized")
            else:
                self.logger.warning("Network analyzer initialization failed")

        if not self.analyzers:
            raise Exception("No analyzers successfully initialized")

    async def analyze_screen_content(
        self,
        screen_capture: Optional[ScreenCapture] = None,
        text_content: Optional[TextContent] = None,
        additional_data: Optional[dict[str, Any]] = None,
    ) -> AggregatedResult:
        """
        Perform comprehensive analysis of screen content

        Args:
            screen_capture: Screen capture data for vision analysis
            text_content: Text content for LLM analysis
            additional_data: Additional context data

        Returns:
            AggregatedResult with final decision
        """
        if not self.initialized:
            raise RuntimeError("Detection engine not initialized")

        analysis_start_time = datetime.now()
        self._analysis_stats["total_analyses"] += 1

        self.logger.debug("Starting content analysis")

        try:
            # Run all applicable analyzers
            detection_results = await self._run_analyzers(
                screen_capture, text_content, additional_data
            )

            # Apply advanced rules engine for intelligent decision making
            rules_decision = self.rules_engine.evaluate_detection_results(
                detection_results, additional_data
            )

            # Aggregate results using rules engine output
            aggregated_result = self.aggregator.aggregate_with_rules(
                detection_results, analysis_start_time, rules_decision
            )

            # Update statistics
            self._update_analysis_stats(analysis_start_time, success=True)

            self.logger.debug(
                f"Analysis complete: {aggregated_result.primary_category.value} "
                f"(confidence: {aggregated_result.overall_confidence:.3f}, "
                f"duration: {aggregated_result.analysis_duration_ms}ms)"
            )

            return aggregated_result

        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            self._update_analysis_stats(analysis_start_time, success=False)

            if self.enable_fallback:
                return self._create_fallback_result(analysis_start_time)
            else:
                raise

    async def _run_analyzers(
        self,
        screen_capture: Optional[ScreenCapture],
        text_content: Optional[TextContent],
        additional_data: Optional[dict[str, Any]],
    ) -> list[DetectionResult]:
        """Run all applicable analyzers on the provided data"""
        analysis_tasks = []

        # Text analysis task
        if text_content and "text" in self.analyzers:
            task = self._run_analyzer_with_timeout("text", text_content)
            analysis_tasks.append(task)

        # Vision analysis task
        if screen_capture and "vision" in self.analyzers:
            task = self._run_analyzer_with_timeout("vision", screen_capture)
            analysis_tasks.append(task)

        # Process analysis task (no input data required)
        if "process" in self.analyzers:
            task = self._run_analyzer_with_timeout("process", None)
            analysis_tasks.append(task)

        # Network analysis task (no input data required)
        if "network" in self.analyzers:
            task = self._run_analyzer_with_timeout("network", None)
            analysis_tasks.append(task)

        if not analysis_tasks:
            self.logger.warning("No applicable analyzers for provided data")
            return []

        # Run analyzers in parallel or sequentially
        if self.parallel_analysis:
            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
        else:
            results = []
            for task in analysis_tasks:
                try:
                    result = await task
                    results.append(result)
                except Exception as e:
                    results.append(e)

        # Filter out failed results and exceptions
        detection_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                analyzer_name = (
                    list(self.analyzers.keys())[i]
                    if i < len(self.analyzers)
                    else "unknown"
                )
                self.logger.warning(f"Analyzer '{analyzer_name}' failed: {result}")
            elif result is not None:
                detection_results.append(result)

        return detection_results

    async def _run_analyzer_with_timeout(
        self, analyzer_name: str, data: Any
    ) -> Optional[DetectionResult]:
        """Run an analyzer with timeout protection"""
        try:
            analyzer = self.analyzers[analyzer_name]

            # Run analyzer with timeout
            result = await asyncio.wait_for(
                analyzer.analyze(data), timeout=self.analysis_timeout
            )

            return result

        except asyncio.TimeoutError:
            self.logger.warning(f"Analyzer '{analyzer_name}' timed out")
            return None
        except Exception as e:
            self.logger.error(f"Analyzer '{analyzer_name}' error: {e}")
            return None

    def _update_analysis_stats(self, start_time: datetime, success: bool) -> None:
        """Update analysis statistics"""
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000

        if success:
            self._analysis_stats["successful_analyses"] += 1
        else:
            self._analysis_stats["failed_analyses"] += 1

        # Update average duration (running average)
        total = self._analysis_stats["total_analyses"]
        current_avg = self._analysis_stats["avg_duration_ms"]
        self._analysis_stats["avg_duration_ms"] = (
            current_avg * (total - 1) + duration_ms
        ) / total

    def _create_fallback_result(self, start_time: datetime) -> AggregatedResult:
        """Create a fallback result when analysis fails"""
        from .types import ActionType, ContentCategory

        duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        return AggregatedResult(
            overall_confidence=0.0,
            primary_category=ContentCategory.UNKNOWN,
            recommended_action=ActionType.ALLOW,  # Fail safe - allow when uncertain
            individual_results=[],
            timestamp=datetime.now(),
            analysis_duration_ms=duration_ms,
        )

    async def cleanup(self) -> None:
        """Cleanup all detection components"""
        self.logger.info("Cleaning up detection engine...")

        # Cleanup all analyzers
        cleanup_tasks = [analyzer.cleanup() for analyzer in self.analyzers.values()]

        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        self.analyzers.clear()
        self.initialized = False

        self.logger.info("Detection engine cleanup complete")

    def get_status(self) -> dict[str, Any]:
        """Get current engine status"""
        return {
            "initialized": self.initialized,
            "running": self.running,
            "active_analyzers": list(self.analyzers.keys()),
            "analysis_stats": self._analysis_stats.copy(),
            "aggregator_stats": self.aggregator.get_statistics(),
        }

    def get_analyzer_status(self, analyzer_name: str) -> dict[str, Any]:
        """Get status of a specific analyzer"""
        if analyzer_name not in self.analyzers:
            return {"error": "Analyzer not found"}

        analyzer = self.analyzers[analyzer_name]
        return {
            "name": analyzer_name,
            "type": analyzer.analyzer_type.value,
            "enabled": analyzer.enabled,
            "ready": asyncio.create_task(analyzer.is_ready()),  # Async status check
            "config": analyzer.config,
        }

    async def test_analyzers(self) -> dict[str, Any]:
        """Test all analyzers with sample data"""
        test_results = {}

        for name, analyzer in self.analyzers.items():
            try:
                # Create appropriate test data based on analyzer type
                test_data = None
                if analyzer.analyzer_type.value == "text":
                    test_data = TextContent(
                        content="This is a test message for analysis.",
                        source="test",
                        timestamp=datetime.now(),
                    )
                elif analyzer.analyzer_type.value == "vision":
                    # Would need to create test image data
                    test_data = None

                # Run test analysis
                if test_data:
                    result = await self._run_analyzer_with_timeout(name, test_data)
                    test_results[name] = {
                        "status": "success" if result else "no_result",
                        "result": result.__dict__ if result else None,
                    }
                else:
                    test_results[name] = {"status": "no_test_data"}

            except Exception as e:
                test_results[name] = {"status": "error", "error": str(e)}

        return test_results

"""
Base Analyzer Interface

Abstract base class for all content analyzers.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from ..types import AnalysisType, DetectionResult


class AnalyzerBase(ABC):
    """Abstract base class for content analyzers"""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.enabled = self.config.get("enabled", True)

    @property
    @abstractmethod
    def analyzer_type(self) -> AnalysisType:
        """Return the type of analysis this analyzer performs"""
        pass

    @abstractmethod
    async def analyze(self, data: Any) -> Optional[DetectionResult]:
        """
        Analyze the provided data and return detection result

        Args:
            data: Input data to analyze (type varies by analyzer)

        Returns:
            DetectionResult if analysis was successful, None otherwise
        """
        pass

    async def is_ready(self) -> bool:
        """
        Check if analyzer is ready to perform analysis

        Returns:
            True if analyzer is ready, False otherwise
        """
        return self.enabled

    async def initialize(self) -> bool:
        """
        Initialize the analyzer (load models, connect to services, etc.)

        Returns:
            True if initialization successful, False otherwise
        """
        self.logger.info(f"Initializing {self.__class__.__name__}")
        return True

    async def cleanup(self) -> None:
        """Cleanup resources when analyzer is no longer needed"""
        self.logger.info(f"Cleaning up {self.__class__.__name__}")

    def get_confidence_threshold(self) -> float:
        """Get the confidence threshold for this analyzer"""
        return self.config.get("confidence_threshold", 0.5)

    def should_analyze(self, data: Any) -> bool:
        """
        Check if this analyzer should process the given data

        Args:
            data: Input data to check

        Returns:
            True if analyzer should process this data, False otherwise
        """
        return self.enabled

"""
Detection System Types

Data models and enums for the detection pipeline.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class AnalysisType(Enum):
    """Types of analysis performed"""

    TEXT = "text"
    VISION = "vision"
    PROCESS = "process"
    NETWORK = "network"
    AUDIO = "audio"


class ContentCategory(Enum):
    """Categories of detected content"""

    PRODUCTIVE = "productive"
    ENTERTAINMENT = "entertainment"
    SOCIAL_MEDIA = "social_media"
    GAMING = "gaming"
    VIDEO_STREAMING = "video_streaming"
    NEWS = "news"
    SHOPPING = "shopping"
    UNKNOWN = "unknown"


class ActionType(Enum):
    """Actions that can be taken"""

    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"
    LOG_ONLY = "log_only"


@dataclass
class DetectionResult:
    """Result from a single analyzer"""

    analyzer_type: AnalysisType
    confidence: float  # 0.0 to 1.0
    category: ContentCategory
    evidence: dict[str, Any]
    timestamp: datetime
    metadata: Optional[dict[str, Any]] = None

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Confidence must be between 0.0 and 1.0, got {self.confidence}"
            )


@dataclass
class AggregatedResult:
    """Final aggregated detection result"""

    overall_confidence: float
    primary_category: ContentCategory
    recommended_action: ActionType
    individual_results: list[DetectionResult]
    timestamp: datetime
    analysis_duration_ms: int

    @property
    def is_entertainment(self) -> bool:
        """Check if content is classified as entertainment"""
        entertainment_categories = {
            ContentCategory.ENTERTAINMENT,
            ContentCategory.GAMING,
            ContentCategory.VIDEO_STREAMING,
            ContentCategory.SOCIAL_MEDIA,
        }
        return self.primary_category in entertainment_categories


@dataclass
class ScreenCapture:
    """Screen capture data for analysis"""

    image: Any  # PIL Image
    image_array: Any  # numpy array
    timestamp: float
    width: int
    height: int
    capture_time: float
    source: str
    active_window_title: Optional[str] = None
    active_process_name: Optional[str] = None


@dataclass
class TextContent:
    """Text content extracted for analysis"""

    content: str
    source: str  # "browser", "ocr", "clipboard", etc.
    timestamp: datetime
    metadata: Optional[dict[str, Any]] = None


@dataclass
class ProcessInfo:
    """Information about running processes"""

    pid: int
    name: str
    executable_path: str
    cpu_percent: float
    memory_percent: float
    is_foreground: bool
    timestamp: datetime


@dataclass
class NetworkActivity:
    """Network activity information"""

    process_name: str
    destination_host: str
    destination_port: int
    bytes_sent: int
    bytes_received: int
    protocol: str
    timestamp: datetime

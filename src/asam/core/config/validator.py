"""
Configuration Validation for ASAM

Validates configuration files and provides schema validation.
"""

import logging
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, ValidationError, validator

logger = logging.getLogger(__name__)


class DetectionConfig(BaseModel):
    """Detection system configuration"""

    confidence_threshold: float = Field(
        0.75, ge=0.0, le=1.0, description="Overall confidence threshold for actions"
    )
    analysis_interval_seconds: float = Field(
        5.0, gt=0.0, description="Time between analysis cycles"
    )
    max_concurrent_analyses: int = Field(
        3, ge=1, le=10, description="Maximum concurrent analysis tasks"
    )

    class Config:
        extra = "forbid"


class TextDetectionConfig(BaseModel):
    """Text detection configuration"""

    enabled: bool = Field(True, description="Enable text-based detection")
    llm_model: str = Field(
        "llama3.2:3b", description="LLM model name for text analysis"
    )
    ollama_host: str = Field("http://localhost:11434", description="Ollama server URL")
    max_text_length: int = Field(
        8000, ge=100, le=50000, description="Maximum text length to analyze"
    )
    timeout_seconds: float = Field(10.0, gt=0.0, description="LLM request timeout")
    weight: float = Field(
        0.4, ge=0.0, le=1.0, description="Weight in aggregated scoring"
    )

    class Config:
        extra = "forbid"


class VisionDetectionConfig(BaseModel):
    """Vision detection configuration"""

    enabled: bool = Field(True, description="Enable vision-based detection")
    motion_threshold: float = Field(
        6.0, ge=0.0, description="Motion detection sensitivity"
    )
    ad_detection_threshold: float = Field(
        0.7, ge=0.0, le=1.0, description="Advertisement detection threshold"
    )
    gaming_ui_threshold: float = Field(
        0.8, ge=0.0, le=1.0, description="Gaming UI detection threshold"
    )
    weight: float = Field(
        0.3, ge=0.0, le=1.0, description="Weight in aggregated scoring"
    )

    class Config:
        extra = "forbid"


class ProcessDetectionConfig(BaseModel):
    """Process detection configuration"""

    enabled: bool = Field(True, description="Enable process-based detection")
    check_interval_seconds: float = Field(
        2.0, gt=0.0, description="Process check interval"
    )
    cpu_threshold: float = Field(
        50.0, ge=0.0, le=100.0, description="CPU usage threshold for gaming detection"
    )
    weight: float = Field(
        0.2, ge=0.0, le=1.0, description="Weight in aggregated scoring"
    )
    gaming_processes: list[str] = Field(
        default_factory=list, description="Known gaming process patterns"
    )
    entertainment_processes: list[str] = Field(
        default_factory=list, description="Known entertainment process patterns"
    )

    class Config:
        extra = "forbid"


class NetworkDetectionConfig(BaseModel):
    """Network detection configuration"""

    enabled: bool = Field(True, description="Enable network-based detection")
    check_interval_seconds: float = Field(
        3.0, gt=0.0, description="Network check interval"
    )
    weight: float = Field(
        0.1, ge=0.0, le=1.0, description="Weight in aggregated scoring"
    )
    gaming_domains: list[str] = Field(
        default_factory=list, description="Known gaming domains"
    )
    streaming_domains: list[str] = Field(
        default_factory=list, description="Known streaming domains"
    )
    social_media_domains: list[str] = Field(
        default_factory=list, description="Known social media domains"
    )

    class Config:
        extra = "forbid"


class ScreenCaptureConfig(BaseModel):
    """Screen capture configuration"""

    capture_quality: int = Field(
        85, ge=1, le=100, description="Screenshot quality (1-100)"
    )
    exclude_menu_bar: bool = Field(
        True, description="Exclude menu bar from captures (macOS)"
    )
    max_capture_size: tuple[int, int] = Field(
        (1920, 1080), description="Maximum capture dimensions"
    )

    @validator("max_capture_size")
    def validate_capture_size(cls, v):
        if len(v) != 2 or any(dim <= 0 for dim in v):
            raise ValueError(
                "max_capture_size must be (width, height) with positive values"
            )
        return v

    class Config:
        extra = "forbid"


class ActionConfig(BaseModel):
    """Action execution configuration"""

    notifications_enabled: bool = Field(True, description="Enable notifications")
    lock_timeout_seconds: int = Field(300, ge=0, description="Screen lock timeout")
    min_lock_interval_seconds: int = Field(
        30, ge=1, description="Minimum time between locks"
    )
    min_notification_interval_seconds: int = Field(
        10, ge=1, description="Minimum time between notifications"
    )
    detailed_logging: bool = Field(True, description="Enable detailed action logging")
    action_history_size: int = Field(
        100, ge=1, le=1000, description="Maximum action history entries"
    )

    class Config:
        extra = "forbid"


class LoggingConfig(BaseModel):
    """Logging configuration"""

    level: str = Field("INFO", description="Log level")
    log_file: Optional[str] = Field(None, description="Log file path")
    dev_mode: bool = Field(False, description="Development mode logging")
    max_log_size_mb: int = Field(
        10, ge=1, le=100, description="Maximum log file size in MB"
    )
    backup_count: int = Field(5, ge=1, le=20, description="Number of log file backups")

    @validator("level")
    def validate_log_level(cls, v):
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()

    class Config:
        extra = "forbid"


class SecurityConfig(BaseModel):
    """Security configuration"""

    anti_tamper_enabled: bool = Field(True, description="Enable anti-tamper protection")
    file_integrity_check: bool = Field(
        True, description="Enable file integrity monitoring"
    )
    process_monitoring: bool = Field(True, description="Enable process monitoring")
    require_admin_privileges: bool = Field(
        False, description="Require admin privileges to run"
    )

    class Config:
        extra = "forbid"


class AsamConfig(BaseModel):
    """Main ASAM configuration"""

    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    text_detection: TextDetectionConfig = Field(default_factory=TextDetectionConfig)
    vision_detection: VisionDetectionConfig = Field(
        default_factory=VisionDetectionConfig
    )
    process_detection: ProcessDetectionConfig = Field(
        default_factory=ProcessDetectionConfig
    )
    network_detection: NetworkDetectionConfig = Field(
        default_factory=NetworkDetectionConfig
    )
    screen_capture: ScreenCaptureConfig = Field(default_factory=ScreenCaptureConfig)
    actions: ActionConfig = Field(default_factory=ActionConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    @validator(
        "text_detection", "vision_detection", "process_detection", "network_detection"
    )
    def validate_weights_sum_reasonable(cls, v, values):
        """Validate that analyzer weights sum to a reasonable value"""
        weights = []
        for field_name in [
            "text_detection",
            "vision_detection",
            "process_detection",
            "network_detection",
        ]:
            if field_name in values and hasattr(values[field_name], "weight"):
                weights.append(values[field_name].weight)
            elif hasattr(v, "weight"):
                weights.append(v.weight)

        if weights and abs(sum(weights) - 1.0) > 0.1:
            logger.warning(
                f"Analyzer weights sum to {sum(weights):.2f}, consider adjusting for optimal results"
            )

        return v

    class Config:
        extra = "forbid"


class ConfigValidator:
    """Configuration validation and management"""

    def __init__(self):
        self.schema = AsamConfig

    def validate_config(
        self, config_data: dict[str, Any]
    ) -> tuple[AsamConfig, list[str]]:
        """
        Validate configuration data against schema

        Returns:
            tuple: (validated_config, list_of_warnings)
        """
        warnings = []

        try:
            # Validate using Pydantic
            validated_config = self.schema(**config_data)

            # Additional custom validations
            warnings.extend(self._validate_analyzer_weights(validated_config))
            warnings.extend(self._validate_llm_connectivity(validated_config))
            warnings.extend(self._validate_paths(validated_config))

            return validated_config, warnings

        except ValidationError as e:
            # Convert validation errors to readable format
            error_details = []
            for error in e.errors():
                field = " -> ".join(str(loc) for loc in error["loc"])
                message = error["msg"]
                error_details.append(f"{field}: {message}")

            raise ValueError(
                "Configuration validation failed:\n" + "\n".join(error_details)
            )

    def _validate_analyzer_weights(self, config: AsamConfig) -> list[str]:
        """Validate analyzer weight configuration"""
        warnings = []

        # Check if all analyzers are disabled
        analyzers = [
            ("text_detection", config.text_detection),
            ("vision_detection", config.vision_detection),
            ("process_detection", config.process_detection),
            ("network_detection", config.network_detection),
        ]

        enabled_analyzers = [name for name, analyzer in analyzers if analyzer.enabled]
        if not enabled_analyzers:
            warnings.append("All analyzers are disabled - detection will not function")

        # Check weight distribution
        total_weight = sum(
            analyzer.weight for _, analyzer in analyzers if analyzer.enabled
        )
        if total_weight == 0:
            warnings.append(
                "Total analyzer weights sum to 0 - detection may not work properly"
            )
        elif abs(total_weight - 1.0) > 0.2:
            warnings.append(
                f"Analyzer weights sum to {total_weight:.2f}, consider normalizing to 1.0"
            )

        return warnings

    def _validate_llm_connectivity(self, config: AsamConfig) -> list[str]:
        """Validate LLM configuration (non-blocking)"""
        warnings = []

        if not config.text_detection.enabled:
            return warnings

        # Basic URL validation
        ollama_host = config.text_detection.ollama_host
        if not ollama_host.startswith(("http://", "https://")):
            warnings.append(f"Ollama host URL should include protocol: {ollama_host}")

        # Model name validation
        model = config.text_detection.llm_model
        if not model or len(model.strip()) == 0:
            warnings.append("LLM model name is empty")

        return warnings

    def _validate_paths(self, config: AsamConfig) -> list[str]:
        """Validate file paths in configuration"""
        warnings = []

        # Check log file path
        if config.logging.log_file:
            log_path = Path(config.logging.log_file)
            if not log_path.parent.exists():
                warnings.append(f"Log file directory does not exist: {log_path.parent}")

        return warnings

    def create_default_config(self) -> AsamConfig:
        """Create a default configuration"""
        return AsamConfig()

    def load_and_validate(self, config_path: Path) -> tuple[AsamConfig, list[str]]:
        """Load and validate configuration from file"""
        try:
            import yaml

            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")

            with open(config_path, encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            if not config_data:
                config_data = {}

            return self.validate_config(config_data)

        except Exception as e:
            raise ValueError(f"Failed to load configuration from {config_path}: {e}")

    def save_config(self, config: AsamConfig, config_path: Path) -> None:
        """Save configuration to file"""
        try:
            import yaml

            # Create directory if needed
            config_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to dict and save
            config_dict = config.dict()

            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    config_dict, f, default_flow_style=False, indent=2, sort_keys=True
                )

            logger.info(f"Configuration saved to {config_path}")

        except Exception as e:
            raise ValueError(f"Failed to save configuration to {config_path}: {e}")

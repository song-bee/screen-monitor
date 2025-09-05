# API Specification
## Advanced Screen Activity Monitor (ASAM)

### 1. NATIVE MESSAGING API

#### 1.1 Browser Extension ↔ Native App Communication

**Message Format:**
```json
{
  "type": "message_type",
  "timestamp": 1704844800,
  "data": {
    // Message-specific data
  },
  "requestId": "unique_request_id"
}
```

#### 1.2 Content Analysis Messages

**Text Content Analysis Request:**
```json
{
  "type": "analyze_text_content",
  "timestamp": 1704844800,
  "data": {
    "url": "https://example.com/article",
    "title": "Page Title",
    "content": "Main text content...",
    "metadata": {
      "reading_time": 300,
      "text_density": 0.75,
      "language": "en",
      "word_count": 1500
    }
  },
  "requestId": "req_001"
}
```

**Text Content Analysis Response:**
```json
{
  "type": "text_analysis_result",
  "timestamp": 1704844800,
  "data": {
    "classification": {
      "type": "entertainment|work|social|other",
      "confidence": 0.85,
      "reasoning": "Content appears to be fiction based on narrative structure"
    },
    "action_taken": "none|warning|lock_screen",
    "threshold_reached": true
  },
  "requestId": "req_001"
}
```

**Browser Activity Monitoring:**
```json
{
  "type": "browser_activity",
  "timestamp": 1704844800,
  "data": {
    "tab_id": 12345,
    "url": "https://youtube.com/watch?v=xyz",
    "title": "Video Title",
    "activity_type": "video_playing|tab_focused|navigation",
    "duration": 120,
    "media_info": {
      "has_video": true,
      "has_audio": true,
      "is_fullscreen": false,
      "video_duration": 600
    }
  },
  "requestId": "req_002"
}
```

### 2. INTERNAL SERVICE API

#### 2.1 Detection Engine API

```python
class DetectionEngine:
    async def register_detector(self, detector: BaseDetector) -> bool
    async def unregister_detector(self, detector_id: str) -> bool
    async def get_detection_status(self) -> DetectionStatus
    async def update_config(self, config_updates: Dict[str, Any]) -> bool
```

**Detection Status Response:**
```python
@dataclass
class DetectionStatus:
    active_detectors: List[str]
    current_confidence: float
    last_detection_time: datetime
    total_detections_today: int
    system_performance: Dict[str, float]
```

#### 2.2 Configuration API

```python
class ConfigurationAPI:
    def get_config(self, section: str = None) -> Dict[str, Any]
    def update_config(self, updates: Dict[str, Any]) -> bool
    def reset_config(self, section: str = None) -> bool
    def validate_config(self, config: Dict[str, Any]) -> ValidationResult
```

**Configuration Schema:**
```json
{
  "detection": {
    "confidence_threshold": 0.75,
    "analysis_interval": 5,
    "text_detection": {
      "enabled": true,
      "llm_model": "llama3.2:3b",
      "max_tokens": 1000,
      "cache_results": true
    },
    "visual_detection": {
      "enabled": true,
      "motion_threshold": 6.0,
      "color_threshold": 3.0,
      "capture_interval": 2
    },
    "audio_detection": {
      "enabled": true,
      "sample_rate": 22050,
      "analysis_window": 5
    }
  },
  "actions": {
    "primary_action": "lock_screen",
    "warning_delay": 10,
    "notification_enabled": true
  },
  "security": {
    "service_protection": true,
    "extension_monitoring": true,
    "tamper_alerts": true
  }
}
```

### 3. LOGGING AND EVENTS API

#### 3.1 Event Types

**Detection Events:**
```json
{
  "event_type": "detection",
  "timestamp": "2024-01-09T15:30:00Z",
  "detection_id": "det_001",
  "detector_type": "text|vision|audio|process|network",
  "content_type": "entertainment|work|social|unknown",
  "confidence": 0.85,
  "source_info": "chrome://youtube.com",
  "action_taken": "warning",
  "metadata": {
    "llm_model": "llama3.2:3b",
    "analysis_duration": 2.1,
    "content_length": 1500
  }
}
```

**System Events:**
```json
{
  "event_type": "system",
  "timestamp": "2024-01-09T15:30:00Z",
  "event_subtype": "startup|shutdown|error|configuration_change",
  "severity": "DEBUG|INFO|WARNING|ERROR|CRITICAL",
  "message": "Service started successfully",
  "details": {
    "version": "1.0.0",
    "platform": "macOS 14.2",
    "configuration_file": "/path/to/config.yaml"
  }
}
```

**Security Events:**
```json
{
  "event_type": "security",
  "timestamp": "2024-01-09T15:30:00Z",
  "security_event_type": "tamper_attempt|extension_removal|service_termination",
  "severity": "WARNING|ERROR|CRITICAL",
  "description": "Browser extension was disabled",
  "source": "extension_monitor",
  "action_taken": "alert_sent",
  "threat_level": "medium|high|critical"
}
```

### 4. REMOTE API (FUTURE FEATURE)

#### 4.1 Authentication

**Login Request:**
```json
POST /api/v1/auth/login
{
  "device_id": "device_uuid",
  "device_name": "MacBook Pro",
  "version": "1.0.0"
}
```

**Authentication Response:**
```json
{
  "token": "jwt_token_here",
  "expires_at": "2024-01-09T23:30:00Z",
  "refresh_token": "refresh_token_here",
  "device_registered": true
}
```

#### 4.2 Data Sync

**Upload Detection Data:**
```json
POST /api/v1/devices/{device_id}/detections
Authorization: Bearer {token}

{
  "detections": [
    {
      "timestamp": "2024-01-09T15:30:00Z",
      "detector_type": "text",
      "content_type": "entertainment",
      "confidence": 0.85,
      "action_taken": "warning"
    }
  ],
  "system_info": {
    "uptime": 86400,
    "cpu_usage": 2.5,
    "memory_usage": 450
  }
}
```

**Get Configuration Updates:**
```json
GET /api/v1/devices/{device_id}/config
Authorization: Bearer {token}

Response:
{
  "config_version": 5,
  "config_updates": {
    "detection.confidence_threshold": 0.80,
    "actions.warning_delay": 15
  },
  "force_update": false
}
```

### 5. ERROR HANDLING

#### 5.1 Error Response Format

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable error message",
    "details": {
      "field": "specific_field_with_error",
      "validation_errors": ["list of validation errors"]
    },
    "request_id": "req_001",
    "timestamp": "2024-01-09T15:30:00Z"
  }
}
```

#### 5.2 Common Error Codes

```python
class APIErrorCodes:
    # General errors
    INVALID_REQUEST = "INVALID_REQUEST"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    RATE_LIMITED = "RATE_LIMITED"

    # Detection errors
    DETECTOR_NOT_FOUND = "DETECTOR_NOT_FOUND"
    DETECTION_FAILED = "DETECTION_FAILED"
    LLM_UNAVAILABLE = "LLM_UNAVAILABLE"

    # Configuration errors
    INVALID_CONFIG = "INVALID_CONFIG"
    CONFIG_VALIDATION_FAILED = "CONFIG_VALIDATION_FAILED"
    CONFIG_SAVE_FAILED = "CONFIG_SAVE_FAILED"

    # System errors
    SYSTEM_ERROR = "SYSTEM_ERROR"
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
```

### 6. WEBHOOK API (FUTURE FEATURE)

#### 6.1 Webhook Configuration

**Register Webhook:**
```json
POST /api/v1/webhooks
{
  "url": "https://example.com/webhook",
  "events": ["detection", "security_alert", "system_error"],
  "secret": "webhook_secret_for_verification",
  "active": true
}
```

**Webhook Payload:**
```json
{
  "webhook_id": "webhook_001",
  "event_type": "detection",
  "timestamp": "2024-01-09T15:30:00Z",
  "device_id": "device_uuid",
  "data": {
    "detection": {
      "content_type": "entertainment",
      "confidence": 0.85,
      "action_taken": "lock_screen"
    }
  },
  "signature": "hmac_sha256_signature"
}
```

### 7. METRICS AND MONITORING API

#### 7.1 Performance Metrics

**Get System Metrics:**
```json
GET /api/v1/metrics/system

Response:
{
  "timestamp": "2024-01-09T15:30:00Z",
  "metrics": {
    "cpu_usage_percent": 2.5,
    "memory_usage_mb": 450,
    "detection_latency_ms": 1200,
    "detections_per_minute": 0.5,
    "accuracy_score": 0.92,
    "uptime_seconds": 86400
  }
}
```

#### 7.2 Detection Analytics

**Get Detection Statistics:**
```json
GET /api/v1/analytics/detections?period=24h

Response:
{
  "period": "24h",
  "total_detections": 45,
  "detection_breakdown": {
    "entertainment": 30,
    "work": 10,
    "social": 5
  },
  "confidence_distribution": {
    "high": 35,
    "medium": 8,
    "low": 2
  },
  "actions_taken": {
    "none": 20,
    "warning": 15,
    "lock_screen": 10
  }
}
```

This API specification provides a comprehensive foundation for all communication within the ASAM system, supporting both current functionality and future enhancements.

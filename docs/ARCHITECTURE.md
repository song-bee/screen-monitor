# System Architecture Document
## Advanced Screen Activity Monitor (ASAM)

### 1. SYSTEM OVERVIEW

```
┌─────────────────────────────────────────────────────────────────┐
│                    ASAM System Architecture                     │
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │   Browser       │    │   Desktop       │    │   Network    │ │
│  │   Extensions    │    │   Applications  │    │   Monitor    │ │
│  └─────────────────┘    └─────────────────┘    └──────────────┘ │
│           │                       │                      │      │
│           ▼                       ▼                      ▼      │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Content Detection Layer                        ││
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐││
│  │  │Text/LLM  │ │Computer  │ │Audio     │ │Process/Window    │││
│  │  │Analysis  │ │Vision    │ │Analysis  │ │Monitoring        │││
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘││
│  └─────────────────────────────────────────────────────────────┘│
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │               Core Processing Engine                        ││
│  │           (Decision Making & Coordination)                  ││
│  └─────────────────────────────────────────────────────────────┘│
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                Action & Logging Layer                       ││
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐││
│  │  │Screen    │ │Logging   │ │Alerts    │ │Remote Sync       │││
│  │  │Control   │ │System    │ │System    │ │(Future)          │││
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### 2. COMPONENT ARCHITECTURE

#### 2.1 Core Service Architecture
```python
# Main service structure
asam_service/
├── core/
│   ├── service_manager.py      # Main service orchestrator
│   ├── detection_engine.py     # Detection coordination
│   ├── action_engine.py        # Response actions
│   └── config_manager.py       # Configuration handling
├── detectors/
│   ├── text_detector.py        # LLM-based text analysis
│   ├── vision_detector.py      # Computer vision analysis
│   ├── audio_detector.py       # Audio pattern analysis
│   ├── process_detector.py     # System process monitoring
│   └── network_detector.py     # Network traffic analysis
├── integrations/
│   ├── browser_bridge.py       # Browser extension communication
│   ├── system_bridge.py        # OS-specific integrations
│   └── remote_bridge.py        # Server communication
└── utils/
    ├── logging.py              # Structured logging
    ├── security.py             # Security and protection
    └── performance.py          # Performance monitoring
```

#### 2.2 Detection Layer Architecture

##### 2.2.1 Text Detection Pipeline
```
Browser Content → Extension Analysis → Content Extraction →
LLM Classification → Confidence Scoring → Decision Engine
```

**Components:**
- **Content Extractor**: Removes navigation, ads, extracts main text
- **Local LLM**: Ollama + Llama3.2-3B for content classification
- **Pattern Matcher**: Fast pre-filtering for known entertainment patterns
- **Confidence Aggregator**: Combines multiple signals for final score

##### 2.2.2 Visual Detection Pipeline
```
Screen Capture → Region Analysis → Motion Detection →
Object Classification → Pattern Recognition → Confidence Scoring
```

**Components:**
- **Screen Sampler**: Intelligent region-based capture
- **Motion Analyzer**: Frame differencing with noise filtering
- **Object Detector**: YOLO-based game/video element detection
- **Ad Filter**: Advertisement pattern exclusion

##### 2.2.3 Audio Detection Pipeline
```
Audio Capture → Frequency Analysis → Pattern Recognition →
Content Classification → Confidence Scoring
```

**Components:**
- **Audio Sampler**: System audio monitoring
- **FFT Analyzer**: Frequency domain analysis
- **Pattern Classifier**: Game/video audio signatures
- **Noise Filter**: Background audio exclusion

#### 2.3 Data Flow Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Input     │    │ Processing  │    │   Output    │
│   Sources   │───▶│   Layer     │───▶│   Actions   │
└─────────────┘    └─────────────┘    └─────────────┘
│                           │                      │
▼                           ▼                      ▼
• Browser DOM               • Multi-signal         • Screen Control
• Screen Pixels            • Analysis              • Logging
• Audio Stream             • LLM Processing        • Notifications
• Process List             • Confidence           • Remote Upload
• Network Traffic          • Calculation           • Security Alerts
```

### 3. CROSS-PLATFORM ARCHITECTURE

#### 3.1 Platform Abstraction Layer
```python
class PlatformAdapter:
    """Abstract base for platform-specific operations"""

    @abstractmethod
    def lock_screen(self) -> bool

    @abstractmethod
    def get_active_window(self) -> WindowInfo

    @abstractmethod
    def install_service(self) -> bool

    @abstractmethod
    def get_system_audio(self) -> AudioStream

# Platform implementations
class MacOSAdapter(PlatformAdapter):
    def lock_screen(self):
        subprocess.run(["pmset", "displaysleepnow"])

class WindowsAdapter(PlatformAdapter):
    def lock_screen(self):
        subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"])
```

#### 3.2 Service Architecture by Platform

##### macOS Service
```xml
<!-- ~/Library/LaunchAgents/com.asam.monitor.plist -->
<dict>
    <key>Label</key>
    <string>com.asam.monitor</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/asam-service</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
```

##### Windows Service
```python
# Windows Service using pywin32
class ASAMWindowsService(win32serviceutil.ServiceFramework):
    _svc_name_ = "ASAMService"
    _svc_display_name_ = "Advanced Screen Activity Monitor"
    _svc_description_ = "Monitors screen activity for productivity"
```

### 4. DATA ARCHITECTURE

#### 4.1 Configuration Schema
```yaml
# config.yaml
detection:
  confidence_threshold: 0.75
  analysis_interval: 5  # seconds

  text_detection:
    llm_model: "llama3.2:3b"
    max_tokens: 1000
    patterns:
      entertainment: ["chapter", "novel", "episode"]

  visual_detection:
    motion_threshold: 6.0
    color_threshold: 3.0
    ignore_regions: [
      {x: 0, y: 0, w: 100, h: 50}  # Status bar
    ]

  audio_detection:
    enabled: true
    sample_rate: 22050

actions:
  primary_action: "lock_screen"
  warning_delay: 10  # seconds
  notification_enabled: true

security:
  service_protection: true
  extension_monitoring: true
  tamper_alerts: true

logging:
  level: "INFO"
  retention_days: 30
  remote_upload: false
```

#### 4.2 Database Schema
```sql
-- SQLite schema for local logging
CREATE TABLE detection_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    detection_type TEXT NOT NULL,  -- 'text', 'visual', 'audio', 'process'
    content_type TEXT,             -- 'entertainment', 'work', 'unknown'
    confidence REAL,               -- 0.0-1.0
    source_info TEXT,              -- URL, window title, etc.
    action_taken TEXT,             -- 'none', 'warning', 'lock_screen'
    metadata JSON                  -- Additional detection details
);

CREATE TABLE system_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    event_type TEXT NOT NULL,      -- 'startup', 'shutdown', 'error', 'security'
    details TEXT,
    severity TEXT DEFAULT 'INFO'   -- 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
);

CREATE TABLE configuration_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    config_key TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT,
    changed_by TEXT                -- 'user', 'system', 'remote'
);
```

### 5. SECURITY ARCHITECTURE

#### 5.1 Security Components
```python
class SecurityManager:
    def __init__(self):
        self.watchdog = ProcessWatchdog()
        self.tamper_detector = TamperDetector()
        self.privilege_manager = PrivilegeManager()

    def protect_service(self):
        """Implement anti-tampering measures"""
        # Register for process termination events
        # Monitor service file integrity
        # Check parent process legitimacy

    def monitor_extensions(self):
        """Watch for browser extension removal"""
        # Check extension manifest files
        # Monitor extension communication
        # Detect extension disable/remove
```

#### 5.2 Protection Mechanisms
- **Service Protection**: Run as system service, require admin privileges to stop
- **File Integrity**: Monitor service files for unauthorized changes
- **Extension Monitoring**: Detect browser extension tampering within 10 seconds
- **Process Injection Protection**: Detect attempts to inject code into service
- **Configuration Protection**: Encrypt sensitive configuration data

### 6. INTEGRATION ARCHITECTURE

#### 6.1 Browser Extension Integration
```javascript
// Extension architecture
chrome_extension/
├── manifest.json              # Extension configuration
├── background/
│   ├── service_worker.js     # Background processing
│   └── native_bridge.js      # Native app communication
├── content/
│   ├── content_analyzer.js   # DOM analysis
│   ├── text_extractor.js     # Content extraction
│   └── activity_monitor.js   # User activity tracking
└── popup/
    ├── popup.html            # Extension popup UI
    └── popup.js              # Popup interactions
```

#### 6.2 Native Messaging Protocol
```json
{
  "type": "content_analysis",
  "data": {
    "url": "https://example.com",
    "title": "Page Title",
    "content": "extracted text content...",
    "metadata": {
      "reading_time": 120,
      "text_density": 0.85,
      "language": "en"
    }
  },
  "timestamp": 1704844800
}
```

### 7. SCALABILITY ARCHITECTURE

#### 7.1 Performance Optimization
- **Lazy Loading**: Load detection modules only when needed
- **Caching**: Cache LLM results for similar content
- **Batching**: Process multiple detections together
- **Sampling**: Intelligent sampling reduces processing load

#### 7.2 Resource Management
```python
class ResourceManager:
    def __init__(self):
        self.cpu_monitor = CPUMonitor(threshold=5.0)
        self.memory_monitor = MemoryMonitor(threshold=500)  # MB
        self.detection_scheduler = AdaptiveScheduler()

    def adjust_detection_frequency(self, system_load):
        """Dynamically adjust detection based on system resources"""
        if system_load > 0.8:
            return self.detection_scheduler.reduce_frequency()
        else:
            return self.detection_scheduler.normal_frequency()
```

### 8. DEPLOYMENT ARCHITECTURE

#### 8.1 Installation Components
```
Installation Package:
├── Service Binary (asam-service)
├── Browser Extensions (Chrome, Firefox)
├── Configuration Files
├── LLM Model (llama3.2-3b.gguf)
├── Installation Scripts
└── Uninstallation Scripts
```

#### 8.2 Update Architecture
- **Automatic Updates**: Check for service updates weekly
- **Model Updates**: Update LLM models monthly
- **Extension Updates**: Through browser extension stores
- **Configuration Sync**: Sync settings from remote server (future)

### 9. MONITORING AND OBSERVABILITY

#### 9.1 System Monitoring
```python
class SystemMonitor:
    def collect_metrics(self):
        return {
            'cpu_usage': self.get_cpu_usage(),
            'memory_usage': self.get_memory_usage(),
            'detection_rate': self.get_detection_rate(),
            'accuracy_score': self.get_accuracy_score(),
            'uptime': self.get_uptime(),
            'error_count': self.get_error_count()
        }
```

#### 9.2 Health Checks
- **Service Health**: Regular health checks every 60 seconds
- **Component Status**: Monitor each detection component
- **Resource Usage**: Track CPU, memory, disk usage
- **Network Connectivity**: Monitor remote server connection
- **Extension Status**: Browser extension health monitoring

This architecture provides a robust, scalable, and maintainable foundation for the ASAM system while supporting the specific requirements for cross-platform deployment and advanced content detection.

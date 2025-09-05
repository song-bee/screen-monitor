# Project Requirements Document
## Advanced Screen Activity Monitor

### 1. PROJECT OVERVIEW

**Project Name**: Advanced Screen Activity Monitor (ASAM)
**Version**: 2.0
**Date**: 2025-01-09

### 2. FUNCTIONAL REQUIREMENTS

#### 2.1 Core Detection Capabilities
- **REQ-F001**: Detect vivid games and videos through motion analysis
- **REQ-F002**: Detect entertainment text content (novels, social media) via LLM analysis
- **REQ-F003**: Identify streaming video content on local and remote devices
- **REQ-F004**: Distinguish between entertainment content and legitimate work/study materials
- **REQ-F005**: Support windowed applications and small-region content detection

#### 2.2 Content Analysis
- **REQ-F006**: Implement multi-layer detection system combining:
  - System process monitoring
  - Browser extension content analysis
  - Computer vision motion detection
  - Audio pattern recognition
  - Network traffic analysis
- **REQ-F007**: Use local LLM for privacy-preserving content classification
- **REQ-F008**: Implement confidence-based thresholds (configurable percentage)
- **REQ-F009**: Filter out advertisements and non-content elements

#### 2.3 Action Management
- **REQ-F010**: Automatically lock/turn off screen when prohibited content detected
- **REQ-F011**: Provide configurable warning system before action
- **REQ-F012**: Support different action types (lock, sleep, shutdown, notification)

#### 2.4 Background Operation
- **REQ-F013**: Run as system service (daemon/service)
- **REQ-F014**: Minimize resource usage and system impact
- **REQ-F015**: Start automatically on system boot
- **REQ-F016**: Operate without user interface dependencies

#### 2.5 Logging and Monitoring
- **REQ-F017**: Log all detections with timestamps and confidence scores
- **REQ-F018**: Track system usage patterns and statistics
- **REQ-F019**: Upload activity logs to remote server (future feature)
- **REQ-F020**: Provide local log viewing capabilities

#### 2.6 Security and Anti-Circumvention
- **REQ-F021**: Protect service from unauthorized termination (admin/root only)
- **REQ-F022**: Detect browser extension tampering/removal
- **REQ-F023**: Alert when security components are compromised
- **REQ-F024**: Implement service watchdog for automatic restart

### 3. NON-FUNCTIONAL REQUIREMENTS

#### 3.1 Performance
- **REQ-NF001**: CPU usage < 5% during normal operation
- **REQ-NF002**: RAM usage < 500MB including LLM
- **REQ-NF003**: Detection latency < 3 seconds
- **REQ-NF004**: System startup impact < 2 seconds

#### 3.2 Reliability
- **REQ-NF005**: Service uptime > 99.9%
- **REQ-NF006**: Graceful handling of system sleep/wake cycles
- **REQ-NF007**: Automatic recovery from crashes
- **REQ-NF008**: Data integrity for logs and configuration

#### 3.3 Scalability
- **REQ-NF009**: Support multiple monitor setups
- **REQ-NF010**: Handle high-frequency content changes
- **REQ-NF011**: Accommodate future feature additions

#### 3.4 Security
- **REQ-NF012**: Local data encryption for sensitive logs
- **REQ-NF013**: Secure communication with remote server
- **REQ-NF014**: Protection against privilege escalation attacks

### 4. PLATFORM REQUIREMENTS

#### 4.1 Primary Support (Full Features)
- **macOS 12.0+** (Monterey and later)
  - Native notifications via Notification Center
  - Status bar integration
  - System sleep controls
  - Process monitoring APIs

#### 4.2 Secondary Support (Core Features)
- **Windows 10/11** (Development and debugging)
  - Windows Service integration
  - Win32 APIs for system control
  - Process and window monitoring

#### 4.3 Future Support (Minimal Changes Required)
- **Linux** (Ubuntu 20.04+, Fedora 35+)
  - Systemd service integration
  - X11/Wayland compatibility

### 5. TECHNICAL CONSTRAINTS

#### 5.1 LLM Integration
- **Local processing preferred** for privacy
- **Model size**: 2-4GB maximum
- **Response time**: < 3 seconds per analysis
- **Confidence scoring**: 0.0-1.0 scale with configurable thresholds

#### 5.2 Browser Integration
- **Chrome/Chromium**: Primary support via extension
- **Firefox**: Secondary support via WebExtension
- **Safari**: Future consideration via App Extension

#### 5.3 Resource Limits
- **Disk space**: < 1GB total installation
- **Network usage**: < 10MB/day for remote features
- **Battery impact**: < 2% on laptop devices

### 6. USER STORIES

#### 6.1 Primary User (Student/Professional)
- **US-001**: As a user, I want the system to automatically detect when I'm reading entertainment content so I can stay focused on work
- **US-002**: As a user, I want my screen to lock when I watch videos during work hours
- **US-003**: As a user, I want the system to run in background without interfering with my normal computer use

#### 6.2 Administrator (IT Department/Parent)
- **US-004**: As an administrator, I want to view usage reports and statistics
- **US-005**: As an administrator, I want to configure detection sensitivity and policies
- **US-006**: As an administrator, I want to prevent users from disabling the monitoring system

### 7. ACCEPTANCE CRITERIA

#### 7.1 Detection Accuracy
- **Text content detection**: > 90% accuracy on entertainment vs work content
- **Video detection**: > 95% accuracy for fullscreen and windowed videos
- **Game detection**: > 90% accuracy across different game types
- **False positive rate**: < 5% for legitimate work activities

#### 7.2 System Integration
- **Service reliability**: Runs continuously for 30+ days without restart
- **Cross-application compatibility**: Works with 95% of common applications
- **Performance impact**: No noticeable system slowdown during normal use

#### 7.3 Security
- **Tamper resistance**: Detects 100% of direct service termination attempts
- **Extension monitoring**: Detects browser extension removal within 10 seconds
- **Recovery**: Automatic restart within 5 seconds of unexpected termination

### 8. RISKS AND ASSUMPTIONS

#### 8.1 Technical Risks
- **LLM performance**: Local models may have accuracy limitations
- **Resource usage**: Computer vision processing may impact performance
- **Browser compatibility**: Extension APIs may change

#### 8.2 Assumptions
- **Target hardware**: Modern computers with 8GB+ RAM
- **Network availability**: Internet connection for initial setup and updates
- **User cooperation**: Users won't attempt sophisticated bypass methods

### 9. FUTURE ENHANCEMENTS

#### 9.1 Phase 2 Features
- **Mobile device monitoring**: iOS/Android app integration
- **Network-wide monitoring**: Router-level detection
- **Machine learning**: Personalized detection models
- **Team management**: Multi-user dashboard

#### 9.2 Phase 3 Features
- **Productivity analytics**: Detailed usage insights
- **Integration APIs**: Third-party application support
- **Cloud synchronization**: Multi-device coordination
- **Advanced reporting**: Customizable reports and alerts

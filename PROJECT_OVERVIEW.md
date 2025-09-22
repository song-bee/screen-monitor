# Advanced Screen Activity Monitor (ASAM) - Project Overview

## 📋 Project Summary

**ASAM** is a sophisticated, AI-powered screen monitoring system designed to detect entertainment content (games, videos, novels, social media) and automatically lock the screen to maintain productivity. The system combines multiple detection technologies including local LLM analysis, computer vision, audio analysis, and browser integration to provide highly accurate content classification with minimal false positives.

## 🎯 Key Features

### Core Detection Capabilities
- **🤖 AI-Powered Text Analysis**: Local LLM (Llama 3.2-3B) for privacy-preserving content classification
- **👁️ Computer Vision**: Motion detection and object recognition for games/videos
- **🎵 Audio Analysis**: Entertainment audio pattern recognition
- **🌐 Browser Integration**: Real-time web content analysis via Chrome/Firefox extensions
- **🌍 Network Monitoring**: Detection of streaming content across network devices

### Smart Detection System
- **Multi-Layer Analysis**: Combines text, visual, audio, and network signals
- **Confidence-Based Thresholds**: Configurable detection sensitivity (e.g., 75% confidence threshold)
- **Ad Filtering**: Intelligent advertisement exclusion to reduce false positives
- **Context Awareness**: Distinguishes between entertainment and legitimate work content

### Security & Anti-Circumvention
- **Service Protection**: Admin/root-only termination with watchdog processes
- **Extension Monitoring**: Detects browser extension tampering within 10 seconds
- **File Integrity**: Monitors service files for unauthorized modifications
- **Tamper Detection**: Multiple security layers with automatic alerts

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    ASAM System Architecture                     │
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐│
│  │   Browser       │    │   Desktop       │    │   Network    ││
│  │   Extensions    │    │   Applications  │    │   Monitor    ││
│  └─────────────────┘    └─────────────────┘    └──────────────┘│
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
│  │  │Control   │ │System    │ │(Future)  │ │                  │││
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.11+**: Main service implementation with asyncio for performance
- **Local LLM**: Ollama + Llama 3.2-3B (2GB model, 3-4GB RAM usage)
- **Computer Vision**: OpenCV, MediaPipe, YOLOv8 for visual analysis
- **Browser Extensions**: Chrome/Firefox WebExtensions with native messaging
- **Database**: SQLite with encryption for local data storage

### AI/ML Components
- **Ollama**: Local LLM server for privacy-preserving text analysis
- **YOLO**: Object detection for game/video element identification
- **Audio ML**: TensorFlow Lite models for audio classification
- **Custom Models**: Trained entertainment content detection models

### Platform Integration
- **macOS**: Native AppKit integration, status bar, notifications
- **Windows**: Win32 APIs, Windows Services, system integration
- **Linux**: X11/Wayland support, systemd services (future)

## 📁 Project Structure

```
asam/
├── docs/                               # Comprehensive documentation
│   ├── REQUIREMENTS.md                 # Detailed requirements specification
│   ├── ARCHITECTURE.md                 # System architecture documentation
│   ├── TECHNOLOGY_STACK.md             # Technology specifications
│   ├── SOLUTION_STRUCTURE.md           # Complete code organization
│   ├── IMPLEMENTATION_PLAN.md          # 12-week development plan
│   ├── API_SPECIFICATION.md            # API documentation
│   └── SECURITY_GUIDELINES.md          # Security implementation guide
│
├── src/asam/                           # Main source code
│   ├── core/                           # Core business logic
│   ├── detectors/                      # Detection modules
│   ├── integrations/                   # External integrations
│   ├── platform/                       # Platform-specific code
│   └── security/                       # Security components
│
├── extensions/                         # Browser extensions
│   ├── chrome/                         # Chrome extension
│   └── firefox/                        # Firefox extension
│
├── tests/                              # Comprehensive test suite
├── scripts/                            # Build and deployment automation
└── resources/                          # AI models and configurations
```

## 🚀 Implementation Plan

### Phase 1: Foundation (Weeks 1-4)
- Core service architecture and configuration management
- Platform abstraction layer for cross-platform support
- Local LLM integration with Ollama
- Basic detection framework and database setup

### Phase 2: Detection Systems (Weeks 5-8)
- Browser extension development with content analysis
- Computer vision implementation for games/videos
- Audio analysis system for entertainment detection
- Network traffic monitoring for streaming detection

### Phase 3: Integration & Security (Weeks 9-10)
- Comprehensive security implementation
- Anti-tamper and bypass protection
- Multi-signal decision engine
- Action execution and warning systems

### Phase 4: Testing & Deployment (Weeks 11-12)
- Complete testing suite (unit, integration, e2e)
- Performance optimization and benchmarking
- Cross-platform packaging and deployment
- Documentation finalization

## 📊 Performance Targets

### Detection Accuracy
- **Text Content**: >90% accuracy for entertainment vs. work content
- **Video Detection**: >95% accuracy for fullscreen and windowed videos
- **Game Detection**: >90% accuracy across different game types
- **False Positive Rate**: <5% for legitimate work activities

### System Performance
- **CPU Usage**: <5% average during normal operation
- **Memory Usage**: <500MB total (including LLM)
- **Detection Latency**: <3 seconds per analysis
- **Service Uptime**: >99.9% reliability over 30+ days

### Security Metrics
- **Tamper Detection**: 100% detection of direct service termination
- **Extension Monitoring**: Detection within 10 seconds of removal
- **Recovery Time**: <5 seconds automatic restart after unexpected termination

## 🔒 Security Features

### Multi-Layer Protection
- **Service Watchdogs**: Multiple independent processes monitor service health
- **File Integrity**: SHA-256 checksums verify service file integrity
- **Extension Monitoring**: Real-time browser extension status tracking
- **Process Protection**: Admin/root privileges required for termination

### Privacy Protection
- **Local Processing**: All AI analysis performed locally (no cloud dependencies)
- **Data Encryption**: Sensitive configuration data encrypted at rest
- **PII Sanitization**: Automatic removal of personal information from logs
- **Minimal Data Collection**: Only essential data for functionality

## 🌍 Platform Support

### Primary Platform (Full Features)
- **macOS 12.0+**: Complete feature set with native integrations
  - Status bar integration showing real-time metrics
  - Native notifications via Notification Center
  - AppKit APIs for system control

### Secondary Platform (Core Features)
- **Windows 10/11**: Full functionality for development and deployment
  - Windows Service integration
  - Win32 APIs for system control
  - Native Windows notifications

### Future Platform (Planned)
- **Linux**: Ubuntu 20.04+, Fedora 35+ with minimal code changes
  - systemd service integration
  - X11/Wayland compatibility

## 📈 Future Enhancements

### Phase 2 Features
- **Remote Dashboard**: Web-based monitoring and configuration
- **Mobile Integration**: iOS/Android companion apps
- **Advanced Analytics**: Detailed usage patterns and insights
- **Team Management**: Multi-user enterprise features

### Phase 3 Features
- **Machine Learning**: Personalized detection models
- **API Integration**: Third-party application support
- **Cloud Synchronization**: Multi-device coordination
- **Advanced Reporting**: Customizable reports and alerts

## 🎯 Use Cases

### Individual Users
- **Students**: Maintain focus during study sessions
- **Remote Workers**: Avoid distractions during work hours
- **Freelancers**: Track and control entertainment consumption

### Organizations
- **Educational Institutions**: Classroom and library computer management
- **Corporate IT**: Employee productivity monitoring
- **Parents**: Child screen time management and content filtering

## 📚 Documentation

The project includes comprehensive documentation covering all aspects:

1. **[REQUIREMENTS.md](docs/REQUIREMENTS.md)**: Detailed functional and non-functional requirements
2. **[ARCHITECTURE.md](docs/ARCHITECTURE.md)**: Complete system architecture and component design
3. **[TECHNOLOGY_STACK.md](docs/TECHNOLOGY_STACK.md)**: Detailed technology specifications
4. **[SOLUTION_STRUCTURE.md](docs/SOLUTION_STRUCTURE.md)**: Complete code organization and structure
5. **[IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)**: 12-week development timeline
6. **[API_SPECIFICATION.md](docs/API_SPECIFICATION.md)**: Complete API documentation
7. **[SECURITY_GUIDELINES.md](docs/SECURITY_GUIDELINES.md)**: Security implementation guide

## 🏁 Getting Started

1. **Review Documentation**: Start with the requirements and architecture documents
2. **Set Up Development Environment**: Python 3.11+, Ollama, VS Code
3. **Install Dependencies**: Follow the setup instructions in TECHNOLOGY_STACK.md
4. **Run Initial Tests**: Verify all components are working correctly
5. **Begin Implementation**: Follow the phased approach in IMPLEMENTATION_PLAN.md

This project provides a comprehensive, production-ready solution for intelligent screen activity monitoring with strong privacy protection and sophisticated detection capabilities.
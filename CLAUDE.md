# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Advanced Screen Activity Monitor (ASAM)** is a sophisticated AI-powered screen monitoring system that detects entertainment content (games, videos, novels, social media) and automatically locks the screen to maintain productivity. The system uses local LLM processing for privacy-preserving content analysis.

### Legacy Components (to be refactored)
- **screen-monitor.py**: Original OCR-based detection system
- **motion.py**: Original motion/color analysis system
- **a.py**: Simple motion detection utility

### New Architecture (under development)
- **Multi-layer detection**: Text (LLM), computer vision, audio, process monitoring
- **Browser extensions**: Chrome/Firefox integration for web content analysis
- **Local AI processing**: Ollama + Llama 3.2-3B for content classification
- **Security framework**: Anti-tamper protection and service monitoring

## Dependencies and Setup

**Activate virtual environment first:**
```bash
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

**New Architecture Dependencies:**
- **Ollama**: Local LLM server for content analysis (`curl -fsSL https://ollama.ai/install.sh | sh`)
- **Llama 3.2-3B**: AI model for text classification (`ollama pull llama3.2:3b`)
- **Python 3.11+**: Recommended (current venv has Python 3.9.6)

**Legacy Dependencies:**
- **Tesseract OCR**: Required for original screen-monitor.py
- **macOS-specific**: pyobjc frameworks for native macOS features

## Core Architecture

### New Multi-Layer Detection System
```
src/asam/
├── core/                           # Main service orchestration
│   ├── service.py                  # Main ASAM service
│   ├── detection/                  # Detection engine
│   │   ├── engine.py              # Detection coordinator
│   │   ├── aggregator.py          # Signal aggregation
│   │   ├── types.py               # Detection data types
│   │   └── analyzers/             # Individual analyzers
│   │       ├── base.py            # Base analyzer class
│   │       ├── text.py            # LLM-based content analysis
│   │       ├── vision.py          # Computer vision detection
│   │       ├── process.py         # System process monitoring
│   │       └── network.py         # Network activity analysis
│   ├── capture/                    # Screen capture functionality
│   │   └── screen.py              # Screenshot and data capture
│   ├── actions/                    # Action execution
│   │   └── executor.py            # Screen lock and warnings
│   └── config/                     # Configuration management
│       └── validator.py           # Config validation
├── config/                         # Global configuration
│   └── manager.py                 # Configuration manager
├── models/                         # Data models
│   └── detection.py              # Detection result models
├── utils/                         # Utilities
│   └── logging.py                # Logging configuration
├── detectors/                     # [Future] Standalone detectors
├── integrations/                  # [Future] External integrations
├── platform/                      # [Future] OS-specific implementations
└── security/                      # [Future] Anti-tamper protection
```

### Detection Flow
1. **Browser Extension** → extracts web content → sends to native app
2. **Text Detector** → LLM analysis → confidence score
3. **Vision Detector** → screen analysis → motion/object detection  
4. **Decision Engine** → aggregates signals → determines action
5. **Action Engine** → executes screen lock or warning

### Key Configuration
```yaml
detection:
  confidence_threshold: 0.75    # Action threshold
  text_detection:
    llm_model: "llama3.2:3b"   # Local LLM model
  visual_detection:
    motion_threshold: 6.0       # Motion sensitivity
```

## Running the Applications

**Legacy applications (current):**
```bash
python screen-monitor.py     # OCR-based detection
python motion.py            # Motion/color analysis
python a.py                 # Simple motion detection
```

**New ASAM service (current implementation):**
```bash
python -m asam              # Main service entry point
# or alternatively:
python -m asam.main         # Direct main module execution
```

## Platform-Specific Features

### macOS
- Status bar integration showing live motion/color metrics
- Native notifications via terminal-notifier
- Screen cropping to exclude menu bars
- AppKit integration for system-level features

### Cross-platform
- Screen locking works on macOS, Windows, and Linux
- Screenshot capture using PIL/pyautogui
- OpenCV for motion analysis

## Development Notes

- **Virtual Environment**: Always activate with `source venv/bin/activate` (Python 3.9.6)
- **Project Structure**: Following new architecture in `docs/SOLUTION_STRUCTURE.md`
- **Implementation Plan**: 12-week phased approach in `docs/IMPLEMENTATION_PLAN.md`
- **Testing**: pytest framework for comprehensive test coverage
- **Security**: Anti-tamper protection and service monitoring built-in
- **Privacy**: All AI processing happens locally, no cloud dependencies

## Development Commands

```bash
# Activate environment
source venv/bin/activate

# Install development dependencies
pip install -r requirements.txt

# Run tests (when implemented)
python -m pytest tests/

# Run ASAM service
python -m asam

# Run with specific config (when implemented)
python -m asam --config custom_config.yaml
```

## Implementation Status

- ✅ **Phase 0**: Planning and documentation complete
- ✅ **Phase 1**: Foundation and core infrastructure
  - Core service architecture implemented
  - Detection engine with analyzer framework
  - Configuration management system
  - Basic screen capture functionality
- 🔄 **Phase 2**: Detection systems implementation (in progress)
  - Base analyzer framework complete
  - Text, vision, process, and network analyzers implemented
  - Signal aggregation system ready
- ⏳ **Phase 3**: Integration and security
- ⏳ **Phase 4**: Testing and deployment
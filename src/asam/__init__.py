"""
Advanced Screen Activity Monitor (ASAM)

A sophisticated AI-powered screen monitoring system for productivity enhancement.
"""

__version__ = "1.0.0"
__author__ = "ASAM Development Team"
__description__ = "AI-powered screen activity monitoring for enhanced productivity"

from .config.manager import ConfigManager

# Core imports for external use
from .core.service import AsamService
from .models.detection import Detection

__all__ = [
    "AsamService",
    "ConfigManager",
    "Detection",
]

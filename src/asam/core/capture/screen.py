"""
Screen capture functionality for ASAM

Provides cross-platform screen capture capabilities with platform-specific optimizations.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Any
import numpy as np
from PIL import Image, ImageGrab

from ..detection.types import ScreenCapture

logger = logging.getLogger(__name__)


class ScreenCaptureProvider(ABC):
    """Abstract base class for screen capture providers"""
    
    @abstractmethod
    async def capture(self, region: Optional[tuple[int, int, int, int]] = None) -> ScreenCapture:
        """Capture screen or region"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available on current platform"""
        pass


class PILScreenCaptureProvider(ScreenCaptureProvider):
    """Cross-platform screen capture using PIL"""
    
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.quality = config.get("capture_quality", 85)
        self.exclude_menu_bar = config.get("exclude_menu_bar", True)
    
    async def capture(self, region: Optional[tuple[int, int, int, int]] = None) -> ScreenCapture:
        """Capture screen using PIL ImageGrab"""
        try:
            start_time = time.time()
            
            # Capture screen
            if region:
                screenshot = ImageGrab.grab(bbox=region)
            else:
                screenshot = ImageGrab.grab()
            
            # macOS: Exclude menu bar if configured
            if self.exclude_menu_bar and hasattr(ImageGrab, 'grab'):
                import platform
                if platform.system() == "Darwin":
                    width, height = screenshot.size
                    # Crop out top 24 pixels (menu bar)
                    screenshot = screenshot.crop((0, 24, width, height))
            
            # Convert to numpy array for analysis
            image_array = np.array(screenshot)
            
            capture_time = time.time() - start_time
            
            return ScreenCapture(
                image=screenshot,
                image_array=image_array,
                timestamp=time.time(),
                width=screenshot.width,
                height=screenshot.height,
                capture_time=capture_time,
                source="PIL"
            )
            
        except Exception as e:
            logger.error(f"PIL screen capture failed: {e}")
            raise
    
    def is_available(self) -> bool:
        """PIL ImageGrab is available on most platforms"""
        try:
            ImageGrab.grab((0, 0, 1, 1))  # Test capture
            return True
        except Exception:
            return False


class MacOSScreenCaptureProvider(ScreenCaptureProvider):
    """macOS-specific screen capture using native APIs"""
    
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.quality = config.get("capture_quality", 85)
        self._cocoa_available = self._check_cocoa_availability()
    
    def _check_cocoa_availability(self) -> bool:
        """Check if PyObjC Cocoa is available"""
        try:
            import Cocoa
            return True
        except ImportError:
            return False
    
    async def capture(self, region: Optional[tuple[int, int, int, int]] = None) -> ScreenCapture:
        """Capture screen using macOS native APIs"""
        if not self._cocoa_available:
            raise RuntimeError("PyObjC Cocoa not available for native macOS capture")
        
        try:
            import Cocoa
            import Quartz
            
            start_time = time.time()
            
            # Get screen dimensions
            screen = Cocoa.NSScreen.mainScreen()
            screen_frame = screen.frame()
            
            # Define capture region
            if region:
                x, y, w, h = region
                capture_rect = Quartz.CGRectMake(x, y, w, h)
            else:
                capture_rect = Quartz.CGRectMake(
                    0, 0, screen_frame.size.width, screen_frame.size.height
                )
            
            # Capture screen
            image_ref = Quartz.CGWindowListCreateImage(
                capture_rect,
                Quartz.kCGWindowListOptionOnScreenOnly,
                Quartz.kCGNullWindowID,
                Quartz.kCGWindowImageDefault
            )
            
            # Convert to PIL Image
            width = Quartz.CGImageGetWidth(image_ref)
            height = Quartz.CGImageGetHeight(image_ref)
            
            # Create bitmap context
            bytes_per_pixel = 4
            bytes_per_row = bytes_per_pixel * width
            color_space = Quartz.CGColorSpaceCreateDeviceRGB()
            
            bitmap_data = bytearray(width * height * bytes_per_pixel)
            context = Quartz.CGBitmapContextCreate(
                bitmap_data, width, height, 8, bytes_per_row, color_space,
                Quartz.kCGImageAlphaPremultipliedLast | Quartz.kCGBitmapByteOrder32Big
            )
            
            # Draw image to context
            Quartz.CGContextDrawImage(
                context, Quartz.CGRectMake(0, 0, width, height), image_ref
            )
            
            # Convert to PIL Image
            image = Image.frombuffer(
                "RGBA", (width, height), bitmap_data, "raw", "BGRA", 0, 1
            ).convert("RGB")
            
            # Convert to numpy array
            image_array = np.array(image)
            
            capture_time = time.time() - start_time
            
            return ScreenCapture(
                image=image,
                image_array=image_array,
                timestamp=time.time(),
                width=width,
                height=height,
                capture_time=capture_time,
                source="macOS_native"
            )
            
        except Exception as e:
            logger.error(f"macOS native screen capture failed: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if macOS native capture is available"""
        import platform
        return platform.system() == "Darwin" and self._cocoa_available


class ScreenCaptureManager:
    """Manages screen capture providers and fallback logic"""
    
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.providers: list[ScreenCaptureProvider] = []
        self.active_provider: Optional[ScreenCaptureProvider] = None
        self._setup_providers()
    
    def _setup_providers(self):
        """Initialize available screen capture providers"""
        import platform
        
        # Platform-specific providers first (better performance)
        if platform.system() == "Darwin":
            macos_provider = MacOSScreenCaptureProvider(self.config)
            if macos_provider.is_available():
                self.providers.append(macos_provider)
        
        # Cross-platform PIL provider as fallback
        pil_provider = PILScreenCaptureProvider(self.config)
        if pil_provider.is_available():
            self.providers.append(pil_provider)
        
        if not self.providers:
            raise RuntimeError("No screen capture providers available")
        
        # Use first available provider
        self.active_provider = self.providers[0]
        logger.info(f"Using screen capture provider: {type(self.active_provider).__name__}")
    
    async def capture_screen(self, region: Optional[tuple[int, int, int, int]] = None) -> ScreenCapture:
        """Capture screen with fallback handling"""
        if not self.active_provider:
            raise RuntimeError("No active screen capture provider")
        
        try:
            return await self.active_provider.capture(region)
        except Exception as e:
            logger.warning(f"Primary provider failed: {e}")
            
            # Try fallback providers
            for provider in self.providers[1:]:
                try:
                    logger.info(f"Trying fallback provider: {type(provider).__name__}")
                    result = await provider.capture(region)
                    self.active_provider = provider  # Switch to working provider
                    return result
                except Exception as fallback_error:
                    logger.warning(f"Fallback provider failed: {fallback_error}")
                    continue
            
            # All providers failed
            raise RuntimeError("All screen capture providers failed")
    
    async def get_screen_info(self) -> dict[str, Any]:
        """Get information about available screens"""
        try:
            # Capture a small region to get screen dimensions
            test_capture = await self.capture_screen(region=(0, 0, 1, 1))
            
            # Try to get full screen dimensions
            full_capture = await self.capture_screen()
            
            return {
                "width": full_capture.width,
                "height": full_capture.height,
                "provider": type(self.active_provider).__name__,
                "available_providers": [type(p).__name__ for p in self.providers]
            }
        except Exception as e:
            logger.error(f"Failed to get screen info: {e}")
            return {"error": str(e)}
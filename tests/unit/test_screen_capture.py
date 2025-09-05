"""
Unit tests for screen capture functionality
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import numpy as np
from PIL import Image

from asam.core.capture.screen import (
    ScreenCaptureManager, PILScreenCaptureProvider, MacOSScreenCaptureProvider, ScreenCapture
)


class TestPILScreenCaptureProvider:
    """Test PIL-based screen capture"""
    
    def setup_method(self):
        """Setup for each test"""
        self.config = {
            "capture_quality": 85,
            "exclude_menu_bar": True
        }
        self.provider = PILScreenCaptureProvider(self.config)
    
    @pytest.mark.asyncio
    async def test_capture_full_screen(self):
        """Test full screen capture"""
        # Mock PIL ImageGrab
        mock_image = MagicMock(spec=Image.Image)
        mock_image.width = 1920
        mock_image.height = 1080
        mock_image.size = (1920, 1080)
        
        with patch('asam.core.capture.screen.ImageGrab') as mock_grab:
            mock_grab.grab.return_value = mock_image
            
            with patch('numpy.array') as mock_array:
                mock_array.return_value = np.zeros((1080, 1920, 3), dtype=np.uint8)
                
                result = await self.provider.capture()
                
                assert isinstance(result, ScreenCapture)
                assert result.width == 1920
                assert result.height == 1080
                assert result.source == "PIL"
                assert result.capture_time > 0
                
                mock_grab.grab.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_capture_with_region(self):
        """Test capture with specific region"""
        region = (100, 100, 500, 400)
        mock_image = MagicMock(spec=Image.Image)
        mock_image.width = 400
        mock_image.height = 300
        mock_image.size = (400, 300)
        
        with patch('asam.core.capture.screen.ImageGrab') as mock_grab:
            mock_grab.grab.return_value = mock_image
            
            with patch('numpy.array') as mock_array:
                mock_array.return_value = np.zeros((300, 400, 3), dtype=np.uint8)
                
                result = await self.provider.capture(region)
                
                mock_grab.grab.assert_called_once_with(bbox=region)
                assert result.width == 400
                assert result.height == 300
    
    @pytest.mark.asyncio
    async def test_macos_menu_bar_exclusion(self):
        """Test macOS menu bar exclusion"""
        mock_image = MagicMock(spec=Image.Image)
        mock_image.width = 1920
        mock_image.height = 1080
        mock_image.size = (1920, 1080)
        
        # Mock cropped image
        mock_cropped = MagicMock(spec=Image.Image)
        mock_cropped.width = 1920
        mock_cropped.height = 1056  # 1080 - 24
        mock_image.crop.return_value = mock_cropped
        
        with patch('asam.core.capture.screen.ImageGrab') as mock_grab, \
             patch('platform.system', return_value="Darwin"):
            
            mock_grab.grab.return_value = mock_image
            
            with patch('numpy.array') as mock_array:
                mock_array.return_value = np.zeros((1056, 1920, 3), dtype=np.uint8)
                
                result = await self.provider.capture()
                
                # Should crop out menu bar (top 24 pixels)
                mock_image.crop.assert_called_once_with((0, 24, 1920, 1080))
                assert result.height == 1056
    
    def test_is_available(self):
        """Test availability check"""
        with patch('asam.core.capture.screen.ImageGrab') as mock_grab:
            mock_grab.grab.return_value = MagicMock()
            
            assert self.provider.is_available() is True
            
            # Test when ImageGrab fails
            mock_grab.grab.side_effect = Exception("No display")
            assert self.provider.is_available() is False
    
    @pytest.mark.asyncio
    async def test_capture_error_handling(self):
        """Test error handling during capture"""
        with patch('asam.core.capture.screen.ImageGrab') as mock_grab:
            mock_grab.grab.side_effect = Exception("Screen capture failed")
            
            with pytest.raises(Exception) as exc_info:
                await self.provider.capture()
            
            assert "Screen capture failed" in str(exc_info.value)


class TestMacOSScreenCaptureProvider:
    """Test macOS-specific screen capture"""
    
    def setup_method(self):
        """Setup for each test"""
        self.config = {"capture_quality": 85}
    
    @patch('asam.core.capture.screen.MacOSScreenCaptureProvider._check_cocoa_availability', return_value=True)
    def test_initialization_with_cocoa(self, mock_cocoa_check):
        """Test initialization when Cocoa is available"""
        provider = MacOSScreenCaptureProvider(self.config)
        assert provider._cocoa_available is True
    
    @patch('asam.core.capture.screen.MacOSScreenCaptureProvider._check_cocoa_availability', return_value=False)
    def test_initialization_without_cocoa(self, mock_cocoa_check):
        """Test initialization when Cocoa is not available"""
        provider = MacOSScreenCaptureProvider(self.config)
        assert provider._cocoa_available is False
    
    def test_cocoa_availability_check(self):
        """Test Cocoa availability checking"""
        provider = MacOSScreenCaptureProvider(self.config)
        
        # Test when Cocoa import succeeds
        with patch.dict('sys.modules', {'Cocoa': MagicMock()}):
            assert provider._check_cocoa_availability() is True
        
        # Test when Cocoa import fails
        with patch('builtins.__import__', side_effect=ImportError):
            assert provider._check_cocoa_availability() is False
    
    @patch('platform.system', return_value="Darwin")
    @patch('asam.core.capture.screen.MacOSScreenCaptureProvider._check_cocoa_availability', return_value=True)
    def test_is_available_on_macos_with_cocoa(self, mock_cocoa_check, mock_platform):
        """Test availability on macOS with Cocoa"""
        provider = MacOSScreenCaptureProvider(self.config)
        assert provider.is_available() is True
    
    @patch('platform.system', return_value="Linux")
    @patch('asam.core.capture.screen.MacOSScreenCaptureProvider._check_cocoa_availability', return_value=True)
    def test_is_available_on_non_macos(self, mock_cocoa_check, mock_platform):
        """Test availability on non-macOS systems"""
        provider = MacOSScreenCaptureProvider(self.config)
        assert provider.is_available() is False
    
    @pytest.mark.asyncio
    @patch('asam.core.capture.screen.MacOSScreenCaptureProvider._check_cocoa_availability', return_value=False)
    async def test_capture_without_cocoa_fails(self, mock_cocoa_check):
        """Test capture fails when Cocoa is not available"""
        provider = MacOSScreenCaptureProvider(self.config)
        
        with pytest.raises(RuntimeError) as exc_info:
            await provider.capture()
        
        assert "PyObjC Cocoa not available" in str(exc_info.value)


class TestScreenCaptureManager:
    """Test screen capture manager"""
    
    def setup_method(self):
        """Setup for each test"""
        self.config = {"capture_quality": 85}
    
    @patch('platform.system', return_value="Darwin")
    def test_initialization_on_macos(self, mock_platform):
        """Test initialization on macOS"""
        with patch('asam.core.capture.screen.MacOSScreenCaptureProvider') as mock_macos, \
             patch('asam.core.capture.screen.PILScreenCaptureProvider') as mock_pil:
            
            # Mock macOS provider as available
            mock_macos_instance = MagicMock()
            mock_macos_instance.is_available.return_value = True
            mock_macos.return_value = mock_macos_instance
            
            # Mock PIL provider as available
            mock_pil_instance = MagicMock()
            mock_pil_instance.is_available.return_value = True
            mock_pil.return_value = mock_pil_instance
            
            manager = ScreenCaptureManager(self.config)
            
            # Should prefer macOS provider
            assert manager.active_provider == mock_macos_instance
            assert len(manager.providers) == 2
    
    @patch('platform.system', return_value="Linux")
    def test_initialization_on_linux(self, mock_platform):
        """Test initialization on Linux"""
        with patch('asam.core.capture.screen.PILScreenCaptureProvider') as mock_pil:
            # Mock PIL provider as available
            mock_pil_instance = MagicMock()
            mock_pil_instance.is_available.return_value = True
            mock_pil.return_value = mock_pil_instance
            
            manager = ScreenCaptureManager(self.config)
            
            # Should use PIL provider
            assert manager.active_provider == mock_pil_instance
            assert len(manager.providers) == 1
    
    def test_initialization_no_providers(self):
        """Test initialization when no providers are available"""
        with patch('asam.core.capture.screen.PILScreenCaptureProvider') as mock_pil:
            # Mock PIL provider as not available
            mock_pil_instance = MagicMock()
            mock_pil_instance.is_available.return_value = False
            mock_pil.return_value = mock_pil_instance
            
            with pytest.raises(RuntimeError) as exc_info:
                ScreenCaptureManager(self.config)
            
            assert "No screen capture providers available" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_capture_screen_success(self):
        """Test successful screen capture"""
        with patch('asam.core.capture.screen.PILScreenCaptureProvider') as mock_pil:
            # Setup mock provider
            mock_provider = MagicMock()
            mock_provider.is_available.return_value = True
            
            mock_capture = MagicMock(spec=ScreenCapture)
            mock_provider.capture.return_value = mock_capture
            
            mock_pil.return_value = mock_provider
            
            manager = ScreenCaptureManager(self.config)
            result = await manager.capture_screen()
            
            assert result == mock_capture
            mock_provider.capture.assert_called_once_with(None)
    
    @pytest.mark.asyncio
    async def test_capture_screen_with_fallback(self):
        """Test screen capture with fallback provider"""
        with patch('asam.core.capture.screen.PILScreenCaptureProvider') as mock_pil:
            # Setup two providers
            mock_primary = MagicMock()
            mock_primary.is_available.return_value = True
            mock_primary.capture.side_effect = Exception("Primary failed")
            
            mock_fallback = MagicMock()
            mock_fallback.is_available.return_value = True
            mock_capture = MagicMock(spec=ScreenCapture)
            mock_fallback.capture.return_value = mock_capture
            
            mock_pil.side_effect = [mock_primary, mock_fallback]
            
            # Manually setup providers list
            manager = ScreenCaptureManager(self.config)
            manager.providers = [mock_primary, mock_fallback]
            manager.active_provider = mock_primary
            
            result = await manager.capture_screen()
            
            # Should switch to fallback provider
            assert result == mock_capture
            assert manager.active_provider == mock_fallback
    
    @pytest.mark.asyncio
    async def test_capture_screen_all_providers_fail(self):
        """Test screen capture when all providers fail"""
        with patch('asam.core.capture.screen.PILScreenCaptureProvider') as mock_pil:
            # Setup failing provider
            mock_provider = MagicMock()
            mock_provider.is_available.return_value = True
            mock_provider.capture.side_effect = Exception("All failed")
            
            mock_pil.return_value = mock_provider
            
            manager = ScreenCaptureManager(self.config)
            
            with pytest.raises(RuntimeError) as exc_info:
                await manager.capture_screen()
            
            assert "All screen capture providers failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_get_screen_info(self):
        """Test getting screen information"""
        with patch('asam.core.capture.screen.PILScreenCaptureProvider') as mock_pil:
            # Setup mock provider
            mock_provider = MagicMock()
            mock_provider.is_available.return_value = True
            
            mock_test_capture = MagicMock()
            mock_full_capture = MagicMock()
            mock_full_capture.width = 1920
            mock_full_capture.height = 1080
            
            mock_provider.capture.side_effect = [mock_test_capture, mock_full_capture]
            mock_pil.return_value = mock_provider
            
            manager = ScreenCaptureManager(self.config)
            info = await manager.get_screen_info()
            
            assert info["width"] == 1920
            assert info["height"] == 1080
            assert "PILScreenCaptureProvider" in info["provider"]
            assert len(info["available_providers"]) >= 1
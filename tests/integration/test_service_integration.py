"""
Integration tests for ASAM service
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from asam.core.config import ConfigValidator
from asam.core.service import AsamService


class TestServiceIntegration:
    """Integration tests for the main service"""

    @pytest.fixture
    def temp_config(self):
        """Create temporary configuration file"""
        config_data = {
            "detection": {"confidence_threshold": 0.6},
            "text_detection": {"enabled": False},  # Disable to avoid LLM dependency
            "vision_detection": {"enabled": True, "weight": 0.6},
            "process_detection": {"enabled": True, "weight": 0.4},
            "network_detection": {"enabled": False},
            "logging": {"level": "INFO", "dev_mode": True},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)

        yield temp_path
        temp_path.unlink()  # Cleanup

    @pytest.mark.asyncio
    async def test_service_initialization_and_cleanup(self, temp_config):
        """Test service initialization and cleanup"""
        service = AsamService(config_path=temp_config)

        # Test initialization
        await service.initialize()
        assert service.config is not None
        assert service.detection_engine is not None
        assert service.action_manager is not None

        # Test cleanup
        await service.cleanup()
        assert service.is_running is False

    @pytest.mark.asyncio
    async def test_service_start_stop(self, temp_config):
        """Test service start and stop operations"""
        service = AsamService(config_path=temp_config)
        await service.initialize()

        # Mock screen capture to avoid GUI dependencies
        with patch.object(service.capture_manager, "capture_screen") as mock_capture:
            mock_screen_capture = MagicMock()
            mock_capture.return_value = mock_screen_capture

            # Start service (run briefly then stop)
            start_task = asyncio.create_task(service.start())

            # Wait a moment for service to start
            await asyncio.sleep(0.1)
            assert service.is_running is True

            # Stop service
            await service.stop()

            # Wait for start task to complete
            try:
                await asyncio.wait_for(start_task, timeout=1.0)
            except asyncio.TimeoutError:
                start_task.cancel()
                try:
                    await start_task
                except asyncio.CancelledError:
                    pass

            assert service.is_running is False

        await service.cleanup()

    @pytest.mark.asyncio
    async def test_analysis_cycle(self, temp_config):
        """Test a complete analysis cycle"""
        service = AsamService(config_path=temp_config)
        await service.initialize()

        # Mock dependencies
        with (
            patch.object(service.capture_manager, "capture_screen") as mock_capture,
            patch.object(
                service.detection_engine, "analyze_screen_content"
            ) as mock_analyze,
            patch.object(service.action_manager, "execute_action") as mock_execute,
        ):

            # Setup mocks
            mock_screen_capture = MagicMock()
            mock_capture.return_value = mock_screen_capture

            mock_result = MagicMock()
            mock_result.overall_confidence = 0.3
            mock_result.recommended_action = "allow"
            mock_analyze.return_value = mock_result

            mock_execute.return_value = True

            # Run one analysis cycle
            await service._perform_analysis_cycle()

            # Verify calls were made
            mock_capture.assert_called_once()
            mock_analyze.assert_called_once()
            mock_execute.assert_called_once()

        await service.cleanup()

    @pytest.mark.asyncio
    async def test_service_with_different_configs(self):
        """Test service behavior with different configurations"""
        configs = [
            # Config 1: High sensitivity
            {
                "detection": {"confidence_threshold": 0.3},
                "text_detection": {"enabled": False},
                "vision_detection": {"enabled": True, "weight": 1.0},
                "process_detection": {"enabled": False},
                "network_detection": {"enabled": False},
            },
            # Config 2: Low sensitivity
            {
                "detection": {"confidence_threshold": 0.9},
                "text_detection": {"enabled": False},
                "vision_detection": {"enabled": True, "weight": 1.0},
                "process_detection": {"enabled": False},
                "network_detection": {"enabled": False},
            },
        ]

        for i, config_data in enumerate(configs):
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                yaml.dump(config_data, f)
                temp_config = Path(f.name)

            try:
                service = AsamService(config_path=temp_config)
                await service.initialize()

                # Verify configuration was loaded correctly
                assert (
                    service.config.detection.confidence_threshold
                    == config_data["detection"]["confidence_threshold"]
                )

                await service.cleanup()

            finally:
                temp_config.unlink()

    @pytest.mark.asyncio
    async def test_error_handling_during_operation(self, temp_config):
        """Test error handling during service operation"""
        service = AsamService(config_path=temp_config)
        await service.initialize()

        # Test screen capture failure
        with patch.object(service.capture_manager, "capture_screen") as mock_capture:
            mock_capture.side_effect = Exception("Screen capture failed")

            # Analysis cycle should handle error gracefully
            await service._perform_analysis_cycle()

            # Service should still be operational
            assert service.is_running is False  # Not started in this test

        # Test detection engine failure
        with (
            patch.object(service.capture_manager, "capture_screen") as mock_capture,
            patch.object(
                service.detection_engine, "analyze_screen_content"
            ) as mock_analyze,
        ):

            mock_capture.return_value = MagicMock()
            mock_analyze.side_effect = Exception("Detection failed")

            # Should handle detection failure
            await service._perform_analysis_cycle()

        await service.cleanup()


class TestConfigurationIntegration:
    """Integration tests for configuration system"""

    def test_config_validation_with_service(self):
        """Test configuration validation integrated with service"""
        validator = ConfigValidator()

        # Test valid configuration
        valid_config_data = {
            "detection": {"confidence_threshold": 0.75},
            "text_detection": {"weight": 0.4},
            "vision_detection": {"weight": 0.6},
            "logging": {"level": "INFO"},
        }

        config, warnings = validator.validate_config(valid_config_data)

        # Should validate successfully
        assert config is not None
        assert isinstance(warnings, list)

        # Config should be usable with service
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            validator.save_config(config, Path(f.name))
            temp_config = Path(f.name)

        try:
            # Service should accept the configuration
            service = AsamService(config_path=temp_config)
            assert service.config_path == temp_config

        finally:
            temp_config.unlink()

    def test_config_error_propagation(self):
        """Test configuration error propagation to service"""
        # Create invalid configuration
        invalid_config_data = {
            "detection": {"confidence_threshold": 2.0}  # Invalid: > 1.0
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(invalid_config_data, f)
            temp_config = Path(f.name)

        try:
            # Service initialization should fail with invalid config
            with pytest.raises((ValueError, Exception)):
                service = AsamService(config_path=temp_config)
                # Initialization may fail during __init__ or during initialize()
                asyncio.run(service.initialize())

        finally:
            temp_config.unlink()


@pytest.mark.slow
class TestPerformanceIntegration:
    """Performance integration tests"""

    @pytest.mark.asyncio
    async def test_analysis_performance(self, temp_config=None):
        """Test analysis performance under load"""
        if temp_config is None:
            config_data = {
                "detection": {
                    "confidence_threshold": 0.5,
                    "analysis_interval_seconds": 0.1,
                },
                "text_detection": {"enabled": False},
                "vision_detection": {"enabled": True, "weight": 1.0},
                "process_detection": {"enabled": False},
                "network_detection": {"enabled": False},
            }

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                yaml.dump(config_data, f)
                temp_config = Path(f.name)

        service = AsamService(config_path=temp_config)
        await service.initialize()

        try:
            # Mock screen capture for consistent timing
            with patch.object(
                service.capture_manager, "capture_screen"
            ) as mock_capture:
                mock_capture.return_value = MagicMock()

                # Run multiple analysis cycles and measure time
                import time

                start_time = time.time()

                for _ in range(10):
                    await service._perform_analysis_cycle()

                end_time = time.time()
                total_time = end_time - start_time
                avg_time_per_cycle = total_time / 10

                # Analysis should complete reasonably quickly
                assert avg_time_per_cycle < 1.0  # Less than 1 second per cycle

        finally:
            await service.cleanup()
            if hasattr(temp_config, "unlink"):
                temp_config.unlink()

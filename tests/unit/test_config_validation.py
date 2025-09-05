"""
Unit tests for configuration validation
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import yaml

from asam.core.config import ConfigValidator, AsamConfig


class TestConfigValidator:
    """Test configuration validation functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        self.validator = ConfigValidator()
    
    def test_default_config_creation(self):
        """Test creating default configuration"""
        config = self.validator.create_default_config()
        
        assert isinstance(config, AsamConfig)
        assert config.detection.confidence_threshold == 0.75
        assert config.text_detection.enabled is True
        assert config.vision_detection.enabled is True
        assert config.process_detection.enabled is True
        assert config.network_detection.enabled is True
    
    def test_valid_config_validation(self):
        """Test validation of valid configuration"""
        config_data = {
            "detection": {
                "confidence_threshold": 0.8,
                "analysis_interval_seconds": 3.0
            },
            "text_detection": {
                "enabled": True,
                "weight": 0.5
            },
            "vision_detection": {
                "enabled": True,
                "weight": 0.3
            }
        }
        
        config, warnings = self.validator.validate_config(config_data)
        
        assert isinstance(config, AsamConfig)
        assert config.detection.confidence_threshold == 0.8
        assert config.detection.analysis_interval_seconds == 3.0
        assert len(warnings) >= 0  # May have warnings but should validate
    
    def test_invalid_confidence_threshold(self):
        """Test validation fails for invalid confidence threshold"""
        config_data = {
            "detection": {
                "confidence_threshold": 1.5  # Invalid: > 1.0
            }
        }
        
        with pytest.raises(ValueError) as exc_info:
            self.validator.validate_config(config_data)
        
        assert "confidence_threshold" in str(exc_info.value)
    
    def test_invalid_log_level(self):
        """Test validation fails for invalid log level"""
        config_data = {
            "logging": {
                "level": "INVALID"
            }
        }
        
        with pytest.raises(ValueError) as exc_info:
            self.validator.validate_config(config_data)
        
        assert "Log level must be one of" in str(exc_info.value)
    
    def test_analyzer_weights_warning(self):
        """Test warning for unbalanced analyzer weights"""
        config_data = {
            "text_detection": {"weight": 0.1},
            "vision_detection": {"weight": 0.1},
            "process_detection": {"weight": 0.1},
            "network_detection": {"weight": 0.1}
        }
        
        config, warnings = self.validator.validate_config(config_data)
        
        # Should warn about weights summing to 0.4 (far from 1.0)
        weight_warnings = [w for w in warnings if "weights sum" in w.lower()]
        assert len(weight_warnings) > 0
    
    def test_all_analyzers_disabled_warning(self):
        """Test warning when all analyzers are disabled"""
        config_data = {
            "text_detection": {"enabled": False},
            "vision_detection": {"enabled": False},
            "process_detection": {"enabled": False},
            "network_detection": {"enabled": False}
        }
        
        config, warnings = self.validator.validate_config(config_data)
        
        # Should warn about all analyzers disabled
        disabled_warnings = [w for w in warnings if "disabled" in w.lower()]
        assert len(disabled_warnings) > 0
    
    def test_load_from_file(self):
        """Test loading configuration from YAML file"""
        config_data = {
            "detection": {"confidence_threshold": 0.9},
            "logging": {"level": "DEBUG"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)
        
        try:
            config, warnings = self.validator.load_and_validate(temp_path)
            
            assert config.detection.confidence_threshold == 0.9
            assert config.logging.level == "DEBUG"
        finally:
            temp_path.unlink()  # Clean up
    
    def test_load_from_nonexistent_file(self):
        """Test error when loading from non-existent file"""
        fake_path = Path("/nonexistent/config.yaml")
        
        with pytest.raises(ValueError) as exc_info:
            self.validator.load_and_validate(fake_path)
        
        assert "not found" in str(exc_info.value)
    
    def test_save_config(self):
        """Test saving configuration to file"""
        config = self.validator.create_default_config()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            self.validator.save_config(config, temp_path)
            
            # Verify file was created and contains valid YAML
            assert temp_path.exists()
            
            with open(temp_path, 'r') as f:
                saved_data = yaml.safe_load(f)
            
            assert 'detection' in saved_data
            assert 'text_detection' in saved_data
            assert saved_data['detection']['confidence_threshold'] == 0.75
        finally:
            temp_path.unlink()  # Clean up


class TestAsamConfig:
    """Test AsamConfig model directly"""
    
    def test_config_model_validation(self):
        """Test Pydantic model validation"""
        # Valid config
        config = AsamConfig(
            detection={"confidence_threshold": 0.8},
            text_detection={"weight": 0.4}
        )
        
        assert config.detection.confidence_threshold == 0.8
        assert config.text_detection.weight == 0.4
    
    def test_config_extra_fields_forbidden(self):
        """Test that extra fields are rejected"""
        with pytest.raises(ValueError):
            AsamConfig(detection={"unknown_field": "value"})
    
    def test_config_field_constraints(self):
        """Test field constraints are enforced"""
        # Confidence threshold out of range
        with pytest.raises(ValueError):
            AsamConfig(detection={"confidence_threshold": 2.0})
        
        # Negative interval
        with pytest.raises(ValueError):
            AsamConfig(detection={"analysis_interval_seconds": -1.0})
        
        # Invalid capture size
        with pytest.raises(ValueError):
            AsamConfig(screen_capture={"max_capture_size": (0, 100)})
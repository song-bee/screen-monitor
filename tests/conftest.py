"""
Pytest configuration and shared fixtures for ASAM tests.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

# Test configuration directory
TEST_DATA_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def test_config():
    """Basic test configuration"""
    return {
        "detection": {
            "confidence_threshold": 0.75,
            "analysis_interval": 5,
            "text_detection": {
                "enabled": True,
                "llm_model": "test_model",
            },
            "visual_detection": {
                "enabled": True,
                "motion_threshold": 6.0,
            },
        },
        "actions": {
            "primary_action": "log_only",
            "warning_delay": 10,
        },
        "logging": {
            "level": "DEBUG",
            "file_enabled": False,
        },
    }


@pytest.fixture
def mock_platform_adapter():
    """Mock platform adapter for testing"""
    mock = Mock()
    mock.lock_screen = AsyncMock(return_value=True)
    mock.get_active_window = Mock(return_value=None)
    mock.capture_audio = AsyncMock(return_value=b"mock_audio_data")
    return mock


@pytest.fixture
def mock_llm_integration():
    """Mock LLM integration for testing"""
    mock = AsyncMock()
    mock.classify_content = AsyncMock(
        return_value={
            "type": "entertainment",
            "confidence": 0.85,
            "reasoning": "Test classification",
        }
    )
    return mock


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Test data fixtures
@pytest.fixture
def sample_entertainment_text():
    """Sample entertainment text for testing"""
    return """
    Chapter 1: The Adventure Begins

    In a land far, far away, there lived a young hero who would soon embark
    on the greatest adventure of their lifetime. The story begins in a small
    village where mysteries and magic await around every corner...
    """


@pytest.fixture
def sample_work_text():
    """Sample work-related text for testing"""
    return """
    API Documentation: User Authentication

    This document describes the authentication endpoints available in our REST API.

    POST /api/auth/login
    Parameters:
    - email: string (required)
    - password: string (required)

    Returns authentication token for subsequent API requests.
    """

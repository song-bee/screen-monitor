"""
Logging utilities for ASAM

Provides structured logging setup and configuration.
"""

import logging
import sys
from typing import Optional


def setup_logging(
    level: int = logging.INFO, log_file: Optional[str] = None, dev_mode: bool = False
) -> None:
    """Setup structured logging for ASAM"""

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Development mode adjustments
    if dev_mode:
        # More verbose logging in dev mode
        logging.getLogger("asam").setLevel(logging.DEBUG)

        # Add detailed formatter for dev mode
        dev_formatter = logging.Formatter(
            "%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(dev_formatter)

    logging.info("Logging system initialized")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module"""
    return logging.getLogger(name)

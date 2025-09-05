#!/usr/bin/env python3
"""
ASAM Main Entry Point

Handles command-line interface and service initialization.
"""

import argparse
import asyncio
import logging
import sys

from .config.manager import ConfigManager
from .core.service import ASAMService
from .utils.logging import setup_logging


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Advanced Screen Activity Monitor")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument(
        "--dev-mode",
        action="store_true",
        help="Run in development mode with enhanced logging",
    )
    parser.add_argument(
        "--install-service", action="store_true", help="Install as system service"
    )
    parser.add_argument(
        "--uninstall-service", action="store_true", help="Uninstall system service"
    )
    parser.add_argument("--restart", action="store_true", help="Restart the service")
    parser.add_argument(
        "--version", action="version", version=f"ASAM {__import__('asam').__version__}"
    )

    return parser.parse_args()


async def main():
    """Main application entry point"""
    args = parse_arguments()

    # Setup logging
    log_level = logging.DEBUG if args.dev_mode else logging.INFO
    setup_logging(level=log_level, dev_mode=args.dev_mode)

    logger = logging.getLogger(__name__)
    logger.info("Starting Advanced Screen Activity Monitor (ASAM)")

    try:
        # Initialize configuration
        config_manager = ConfigManager(config_path=args.config)
        await config_manager.load_config()

        # Handle service management commands
        if args.install_service:
            logger.info("Installing ASAM as system service...")
            # TODO: Implement service installation
            return 0

        if args.uninstall_service:
            logger.info("Uninstalling ASAM system service...")
            # TODO: Implement service uninstallation
            return 0

        # Initialize and start main service
        service = ASAMService(config_manager)

        if args.restart:
            logger.info("Restarting ASAM service...")

        await service.start()

    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
        return 0
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


def cli_main():
    """CLI entry point wrapper"""
    try:
        return asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down...")
        return 0


if __name__ == "__main__":
    sys.exit(cli_main())

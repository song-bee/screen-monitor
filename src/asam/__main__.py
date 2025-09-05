#!/usr/bin/env python3
"""
ASAM Package Main Entry

Allows the package to be executed with `python -m asam`.
"""

import sys

from .main import cli_main

if __name__ == "__main__":
    sys.exit(cli_main())

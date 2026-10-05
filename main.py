"""Arcane entry point.

python main.py            # start the bots
python main.py check      # preflight checks
python main.py chat       # chat with a personality in the terminal
python main.py --help
"""

from __future__ import annotations

import sys

from arcane.cli import main

if __name__ == "__main__":
    sys.exit(main())

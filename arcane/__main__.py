"""Allow ``python -m arcane``."""

from __future__ import annotations

import sys

from arcane.cli import main

if __name__ == "__main__":
    sys.exit(main())

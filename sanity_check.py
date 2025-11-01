#!/usr/bin/env python
"""
Launcher script for running the Pokemon Gold environment sanity check.

This script adds the src/ directory to Python's path and runs the sanity check.
You can run this from the project root directory:
    python sanity_check.py

The sanity check verifies:
- ROM file can be loaded
- Savestate can be loaded
- Environment can be reset
- Actions work correctly
- RAM reading functions work
"""

import sys
import runpy
from pathlib import Path

# Add src/ to Python path so imports work
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

# Run the sanity check script as if executed directly
if __name__ == "__main__":
    script_path = src_dir / "sanity_check.py"
    runpy.run_path(str(script_path), run_name="__main__")

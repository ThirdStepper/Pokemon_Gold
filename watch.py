#!/usr/bin/env python
"""
Launcher script for watching a trained Pokemon Gold RL agent.

This script adds the src/ directory to Python's path and launches the viewer.
You can run this from the project root directory:
    python watch.py

Controls:
    Q - Quit
    R - Reset episode
    SPACE - Pause/Resume
    F - Fast forward toggle
"""

import sys
import runpy
from pathlib import Path

# Add src/ to Python path so imports work
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

# Run the watch agent script as if executed directly
if __name__ == "__main__":
    script_path = src_dir / "watch_agent.py"
    runpy.run_path(str(script_path), run_name="__main__")

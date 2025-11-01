#!/usr/bin/env python
"""
Launcher script for training the Pokemon Gold RL agent.

This script adds the src/ directory to Python's path and launches the training.
You can run this from the project root directory:
    python train.py
"""

import sys
import runpy
from pathlib import Path

# Add src/ to Python path so imports work
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

# Run the training script as if executed directly
if __name__ == "__main__":
    script_path = src_dir / "train_pokemon_gold.py"
    runpy.run_path(str(script_path), run_name="__main__")

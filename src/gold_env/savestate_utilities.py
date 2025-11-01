# savestate_utilities.py
"""
Savestate Management Utilities for Pokemon Gold Environment

This mixin provides methods for loading and saving emulator savestates.
Savestates allow quick episode restarts and skipping game intro sequences.
"""

from io import BytesIO
from pathlib import Path
from typing import Optional


class SavestateUtilsMixin:
    """
    Mixin providing savestate management utilities.

    Savestates let us skip the intro and restart episodes quickly by
    saving/loading the emulator's complete state.

    Required attributes from parent class:
    - self.pyboy: PyBoy emulator instance
    - self.boot_state: Optional[BytesIO] - in-memory savestate
    - self.init_state_path: Optional[Path] - disk savestate file
    """

    def _load_init_state_if_any(self) -> bool:
        """
        Load a savestate to skip the game intro and start at gameplay.

        This tries to load from:
        1. self.boot_state (in-memory savestate) - fastest
        2. self.init_state_path (disk file) - slower but persistent

        Returns:
            True if a savestate was loaded, False otherwise
        """
        # Try in-memory savestate first (fastest)
        if self.boot_state is not None:
            self.boot_state.seek(0)
            self.pyboy.load_state(self.boot_state)
            return True

        # Fall back to disk file
        if self.init_state_path and self.init_state_path.is_file():
            with open(self.init_state_path, "rb") as f:
                self.pyboy.load_state(f)
            return True

        return False

    def capture_boot_state(self, to_path: Optional[Path] = None):
        """
        Save the current emulator state for fast episode restarts.

        Call this after manually playing through the intro to reach gameplay.
        The state is stored in memory and optionally saved to disk.

        Args:
            to_path: Optional path to save the state file to disk
        """
        buf = BytesIO()
        buf.seek(0)
        self.pyboy.save_state(buf)
        self.boot_state = buf

        # Optionally write to disk for persistence
        if to_path:
            to_path.parent.mkdir(parents=True, exist_ok=True)
            with open(to_path, "wb") as f:
                buf.seek(0)
                f.write(buf.read())

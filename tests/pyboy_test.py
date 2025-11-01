# pyboy_quick.py
from pathlib import Path
from pyboy import PyBoy

# Change this to your actual ROM location:
rom = Path(r"C:\_Projects\_Python\ML\Pokemon_Gold\_rom\Pokemon_Gold.gbc")  # example absolute path

if not rom.is_file():
    raise FileNotFoundError(f"ROM not found at: {rom}\n"
                            f"Tip: print your CWD with:\n"
                            f"  import os; print(os.getcwd())")

# Headless run (no window). For a GUI later: just remove window='null'.
pyboy = PyBoy(str(rom))

for _ in range(600):  # ~2 seconds of emulated time
    pyboy.tick()

pyboy.stop()
print("PyBoy ticked OK.")

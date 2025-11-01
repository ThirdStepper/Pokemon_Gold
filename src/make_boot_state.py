from io import BytesIO
from threading import Thread, Event
from time import sleep
from pyboy import PyBoy
import config

# Paths are now configured in config.py
# ROM path from config
# Output savestate path from config

def wait_for_enter(done_evt: Event):
    input("When you've reached your desired start point, press Enter here to save the state...\n")
    done_evt.set()

def main():
    config.INIT_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Visible window so you can play to the right moment:
    pyboy = PyBoy(
        str(config.ROM_PATH),
        symbols=str(config.SYM_PATH),
        window="SDL2",
        sound_emulated=False
    )
    done = Event()
    Thread(target=wait_for_enter, args=(done,), daemon=True).start()
    print("PyBoy running. Use the window to progress to gameplay. (Ctrl+C to abort)")

    try:
        while not done.is_set():
            pyboy.tick()
            sleep(0.0)  # yield
        # Save current emulator state
        buf = BytesIO()
        buf.seek(0)
        pyboy.save_state(buf)
        with open(config.INIT_STATE_PATH, "wb") as f:
            buf.seek(0); f.write(buf.read())
        print(f"Saved state → {config.INIT_STATE_PATH}")
    finally:
        pyboy.stop()

if __name__ == "__main__":
    main()

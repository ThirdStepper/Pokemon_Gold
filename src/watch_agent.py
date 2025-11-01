# watch_agent.py
"""
Live Training Viewer for Pokemon Gold RL Agent

This script lets you watch a trained agent play Pokemon Gold in real-time.
It shows the game window running at Game Boy speed (60 FPS) and displays
live statistics in a Dear PyGui GUI window.

Features:
- Loads the latest trained model checkpoint
- Shows game window at 60 FPS (Game Boy speed)
- Displays live stats in tabbed GUI: Overview, Framestack, Advanced
- Auto-reloads newer checkpoints during training (if enabled)
- Keyboard controls: Q=quit, R=reset, SPACE=pause, F=fast forward
- GUI controls: buttons for all keyboard shortcuts + auto-reload toggle

Usage:
    python watch_agent.py
"""

import os
import time
from pathlib import Path
from collections import deque
from io import BytesIO
import config
from viewer_gui import ViewerGUI

# =============================================================================
# CPU THREAD LIMITING - PART 1: Environment Variables
# =============================================================================
# Set thread environment variables BEFORE importing numpy/torch
# This prevents thread oversubscription at the library level

if hasattr(config, 'TORCH_NUM_THREADS') and config.TORCH_NUM_THREADS is not None:
    num_threads = str(config.TORCH_NUM_THREADS)
    os.environ.setdefault("OMP_NUM_THREADS", num_threads)
    os.environ.setdefault("MKL_NUM_THREADS", num_threads)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", num_threads)
    os.environ.setdefault("NUMEXPR_NUM_THREADS", num_threads)

# Now import numpy/torch with thread limits already set
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from pokemon_gold_env import PokemonGoldEnv

# =============================================================================
# CPU THREAD LIMITING - PART 2: PyTorch Threads
# =============================================================================
# Apply PyTorch-specific thread limiting after import

if hasattr(config, 'TORCH_NUM_THREADS') and config.TORCH_NUM_THREADS is not None:
    torch.set_num_threads(config.TORCH_NUM_THREADS)
    # Also set interop threads for better parallelism
    if hasattr(torch, 'set_num_interop_threads'):
        torch.set_num_interop_threads(max(1, config.TORCH_NUM_THREADS // 2))

# =============================================================================

# =============================================================================
# VIEWER CONSTANTS
# =============================================================================

# ANSI escape codes for faster screen clearing (no shell spawn)
ANSI_CLEAR = "\033[2J\033[H"

# Device to run viewer on (default: CPU to keep GPU free for training)
VIEWER_DEVICE = getattr(config, 'VIEWER_DEVICE', 'cpu')

# Default console display refresh interval (seconds)
DISPLAY_REFRESH_DEFAULT = 0.5

# =============================================================================


def get_latest_checkpoint() -> Path:
    """
    Find the most recent model checkpoint in the models directory.

    Checkpoint filenames follow the pattern: pokemon_gold_ppo_XXXXX_steps.zip
    This function parses the step count and returns the highest one.

    Returns:
        Path to the latest checkpoint file

    Raises:
        FileNotFoundError: If no checkpoints are found
    """
    if not config.MODELS_DIR.exists():
        raise FileNotFoundError(f"Models directory not found: {config.MODELS_DIR}")

    # Prefer step-tagged checkpoint files first, fallback to any .zip
    step_zips = list(config.MODELS_DIR.glob("*_steps.zip"))
    checkpoint_files = step_zips if step_zips else list(config.MODELS_DIR.glob("*.zip"))

    if not checkpoint_files:
        raise FileNotFoundError(
            f"No model checkpoints found in {config.MODELS_DIR}\n"
            "Train a model first with train_pokemon_gold.py"
        )

    # Parse step counts from filenames and find the latest
    def extract_steps(path: Path) -> int:
        """Extract step count from checkpoint filename."""
        try:
            # Expected format: pokemon_gold_ppo_5000_steps.zip
            parts = path.stem.split("_")
            for i, part in enumerate(parts):
                if part == "steps" and i > 0:
                    return int(parts[i - 1])
            return 0  # If pattern doesn't match, sort to beginning
        except (ValueError, IndexError):
            return 0

    latest_checkpoint = max(checkpoint_files, key=extract_steps)
    return latest_checkpoint


def get_checkpoint_info(checkpoint_path: Path) -> dict:
    """
    Extract information from a checkpoint filename.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Dictionary with checkpoint info (filename, steps)
    """
    def extract_steps(path: Path) -> int:
        try:
            parts = path.stem.split("_")
            for i, part in enumerate(parts):
                if part == "steps" and i > 0:
                    return int(parts[i - 1])
            return 0
        except (ValueError, IndexError):
            return 0

    return {
        "filename": checkpoint_path.name,
        "steps": extract_steps(checkpoint_path),
        "path": checkpoint_path
    }


def _file_is_stable(path: Path, min_age_sec: float = 1.0) -> bool:
    """
    Best-effort check that a file is not currently being written.

    Prevents loading checkpoint files that are mid-write by checking:
    1. File exists and has non-zero size
    2. File hasn't been modified in the last min_age_sec seconds

    Args:
        path: Path to the file to check
        min_age_sec: Minimum age (seconds since last modification) to consider stable

    Returns:
        True if file appears stable and ready to load, False otherwise
    """
    try:
        st = path.stat()
        return st.st_size > 0 and (time.time() - st.st_mtime) >= min_age_sec
    except FileNotFoundError:
        return False


def make_env():
    """
    Create and configure a Pokemon Gold environment for visualization.

    This creates a windowed environment running at 60 FPS so you can
    watch the agent play in real-time.

    Returns:
        Configured PokemonGoldEnv instance
    """
    env = PokemonGoldEnv(
        rom_path=config.ROM_PATH,
        sym_path=config.SYM_PATH,
        render_mode="rgb_array",
        frame_skip=config.FRAME_SKIP,
        max_steps=config.MAX_STEPS_PER_EPISODE,
        require_init_state=config.REQUIRE_INIT_STATE,
        input_hold_frames=config.INPUT_HOLD_FRAMES,
        post_release_frames=config.POST_RELEASE_FRAMES,
        enable_window=True,  # Show the game window
        target_fps=config.TARGET_FPS  # Run at Game Boy speed
    )

    # Set the savestate path to skip intro
    if config.INIT_STATE_PATH.is_file():
        env.init_state_path = config.INIT_STATE_PATH
    else:
        print(f"[WARNING] Init savestate not found: {config.INIT_STATE_PATH}")

    return env


def print_header():
    """Print the viewer header (minimal console output)."""
    print("\n" + "=" * 70)
    print(" " * 20 + "POKEMON GOLD RL AGENT VIEWER")
    print("=" * 70)
    print("GUI window opened. Close PyBoy window or press Q to quit.")
    print()


def watch_agent():
    """
    Main viewer loop with GUI.

    Loads the latest model and runs it in the environment, displaying
    live statistics in a Dear PyGui window.
    """
    print_header()
    print("Initializing viewer...")
    print(f"ROM Path: {config.ROM_PATH}")
    print(f"SYM Path: {config.SYM_PATH}")

    # Find and load the latest checkpoint
    try:
        checkpoint_path = get_latest_checkpoint()
        checkpoint_info = get_checkpoint_info(checkpoint_path)
        print(f"Loading checkpoint: {checkpoint_info['filename']}")
        print(f"Training steps: {checkpoint_info['steps']:,}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    # Create environment
    print("Creating environment (windowed mode)...")
    env = DummyVecEnv([make_env])

    # Apply frame stacking if configured (must match training configuration)
    if config.FRAME_STACK > 0:
        print(f"Applying frame stacking (n_stack={config.FRAME_STACK})...")
        env = VecFrameStack(env, n_stack=config.FRAME_STACK, channels_order="last")

    # Load model
    print("Loading model...")
    try:
        # Load on viewer device (defaults to CPU to avoid GPU contention with training)
        model = PPO.load(checkpoint_path, env=env, device=VIEWER_DEVICE)
        # Set to inference mode for zero gradient overhead
        if hasattr(model, 'policy'):
            model.policy.set_training_mode(False)
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        env.close()
        return

    # Create viewer states directory if it doesn't exist
    config.VIEWER_STATES_DIR.mkdir(parents=True, exist_ok=True)

    # Rewind ring buffer for in-memory savestates
    rewind_buffer = deque(maxlen=config.VIEWER_REWIND_SLOTS)
    last_rewind_capture = [time.time()]

    # Manual save state (separate from rewind buffer)
    manual_save_state = [None]

    # Create GUI with callbacks
    quit_requested = [False]  # Use list for mutable closure
    force_reset = [False]
    step_once_requested = [False]

    def on_quit():
        quit_requested[0] = True

    def on_reset():
        force_reset[0] = True

    def on_pause_toggle(paused):
        # Pause state is managed by GUI
        pass

    def on_fast_forward_toggle(fast_forward):
        # Update environment fast forward state
        unwrapped = env.envs[0] if hasattr(env, 'envs') else env
        unwrapped.fast_forward = fast_forward

    def on_autoreload_toggle(enabled):
        # Update config (will be checked in main loop)
        config.AUTO_RELOAD_MODELS = enabled

    def on_step_once():
        # Request a single step while paused
        step_once_requested[0] = True

    def on_save_state():
        # Save current state to manual slot (separate from rewind buffer)
        try:
            unwrapped = env.envs[0] if hasattr(env, 'envs') else env
            state_io = BytesIO()
            unwrapped.pyboy.save_state(state_io)
            manual_save_state[0] = state_io
            print("\n[State saved to manual slot]")
        except Exception as e:
            print(f"\n[Error saving state: {e}]")

    def on_load_state():
        # Load state from manual slot
        try:
            if manual_save_state[0] is None:
                print("\n[No saved state to load - save a state first with 'S']")
                return
            unwrapped = env.envs[0] if hasattr(env, 'envs') else env
            manual_save_state[0].seek(0)
            unwrapped.pyboy.load_state(manual_save_state[0])
            print("\n[State loaded from manual slot]")
        except Exception as e:
            print(f"\n[Error loading state: {e}]")

    def on_rewind():
        # Load state from rewind buffer (most recent)
        try:
            if len(rewind_buffer) == 0:
                print("\n[No rewind history available - wait a moment for states to accumulate]")
                return
            # Pop the most recent state from buffer
            state_io = rewind_buffer.pop()
            unwrapped = env.envs[0] if hasattr(env, 'envs') else env
            state_io.seek(0)
            unwrapped.pyboy.load_state(state_io)
            print(f"\n[Rewound to previous state - {len(rewind_buffer)} states remaining in buffer]")
        except Exception as e:
            print(f"\n[Error rewinding: {e}]")

    print("Creating GUI window...")
    gui = ViewerGUI(
        width=1200,
        height=800,
        on_quit=on_quit,
        on_reset=on_reset,
        on_pause_toggle=on_pause_toggle,
        on_fast_forward_toggle=on_fast_forward_toggle,
        on_autoreload_toggle=on_autoreload_toggle,
        on_step_once=on_step_once,
        on_save_state=on_save_state,
        on_load_state=on_load_state,
        on_rewind=on_rewind,
    )
    gui.setup_viewport()

    print("Starting viewer...")
    print("Keyboard shortcuts:")
    print("  Q=quit, R=reset, SPACE=pause, F=fast forward")
    print("  S=save state, L=load state, Backspace=rewind")
    print()

    # Tracking variables
    episode_num = 0
    recent_rewards = deque(maxlen=config.VIEWER_EPISODE_HISTORY)
    last_reload_check = time.time()

    # GUI frame timing (fixed 60 FPS rendering)
    gui_target_fps = 60.0
    gui_frame_time = 1.0 / gui_target_fps

    # ENV FPS tracking (for accurate steps/second display)
    env_step_times = deque(maxlen=30)  # Track last 30 step timestamps

    last_action = None

    try:
        while not quit_requested[0] and not gui.should_close():
            # Reset episode
            obs = env.reset()
            episode_reward = 0.0
            step_count = 0
            episode_num += 1
            gui.reset_episode_data()

            # Episode loop - DECOUPLED RENDERING: GUI runs at 60 FPS, env steps at ~2.5 Hz
            while not quit_requested[0] and not gui.should_close():
                try:
                    # Start of frame timing
                    frame_start_time = time.time()
                    unwrapped_env = env.envs[0]

                    # ============================================================
                    # PHASE 1: HANDLE PAUSE STATE
                    # ============================================================
                    if gui.paused and not step_once_requested[0]:
                        # While paused: just render existing data at 60 FPS, no stepping
                        gui.render()

                        # Frame rate limiting for 60 FPS
                        elapsed = time.time() - frame_start_time
                        if elapsed < gui_frame_time:
                            time.sleep(gui_frame_time - elapsed)

                        continue  # Skip to next frame

                    # ============================================================
                    # PHASE 2: HANDLE SINGLE-STEP
                    # ============================================================
                    if step_once_requested[0]:
                        step_once_requested[0] = False
                        # Fall through to execute one step below

                    # ============================================================
                    # PHASE 3: HANDLE FORCE RESET
                    # ============================================================
                    if force_reset[0]:
                        force_reset[0] = False
                        break  # Exit episode loop

                    # ============================================================
                    # PHASE 4: ENVIRONMENT STEPPING (blocks for ~0.4 seconds)
                    # ============================================================
                    if not gui.paused:
                        # Get action from model
                        with torch.inference_mode():
                            action, _ = model.predict(obs, deterministic=True)
                        last_action = int(action[0]) if hasattr(action, '__len__') else int(action)

                        # Take step in environment (BLOCKING CALL - takes ~0.4s)
                        obs, rewards, dones, infos = env.step(action)
                        episode_reward += float(rewards[0])
                        step_count += 1

                        # Track env step timestamp for FPS calculation
                        env_step_times.append(time.time())

                        # Calculate and update ENV FPS display
                        if len(env_step_times) >= 2:
                            time_span = env_step_times[-1] - env_step_times[0]
                            env_fps = (len(env_step_times) - 1) / time_span if time_span > 0 else 0.0
                            gui.set_env_fps(env_fps)

                        # Auto-capture rewind snapshots periodically
                        if frame_start_time - last_rewind_capture[0] >= config.VIEWER_REWIND_INTERVAL_SEC:
                            try:
                                state_io = BytesIO()
                                unwrapped_env.pyboy.save_state(state_io)
                                rewind_buffer.append(state_io)
                                last_rewind_capture[0] = frame_start_time
                            except Exception as e:
                                pass  # Silent failure

                        # Update all GUI data immediately after env step
                        if not unwrapped_env.fast_forward:
                            # Normal mode: update all data
                            gui.update_all_data(
                                episode_num, step_count, episode_reward, env,
                                checkpoint_info, recent_rewards, obs, last_action
                            )
                        else:
                            # Fast forward mode: only update overview (lightweight)
                            gui.update_overview(
                                episode_num, step_count, episode_reward, env,
                                checkpoint_info, recent_rewards,
                                paused=False, fast_forward=True
                            )

                        # Update status bar
                        if config.AUTO_RELOAD_MODELS:
                            time_since_check = frame_start_time - last_reload_check
                            time_until_next = max(0, config.VIEWER_RELOAD_CHECK_INTERVAL - time_since_check)
                            gui.update_status_bar(
                                f"Auto-reload: next check in {time_until_next:.0f}s | Episode {episode_num}"
                            )
                        else:
                            gui.update_status_bar(f"Episode {episode_num}")

                    # ============================================================
                    # PHASE 5: CHECK IF EPISODE IS DONE (only after env step)
                    # ============================================================
                    if not gui.paused and bool(dones[0]):
                        recent_rewards.append(episode_reward)
                        # Final GUI update for this episode
                        gui.update_all_data(
                            episode_num, step_count, episode_reward, env,
                            checkpoint_info, recent_rewards, obs, last_action
                        )
                        gui.render()
                        time.sleep(1)  # Brief pause at episode end
                        break

                    # ============================================================
                    # PHASE 6: GUI RENDERING (always at 60 FPS)
                    # ============================================================
                    gui.render()

                    # ============================================================
                    # PHASE 7: FRAME RATE LIMITING (maintain 60 FPS)
                    # ============================================================
                    if not unwrapped_env.fast_forward:
                        elapsed = time.time() - frame_start_time
                        if elapsed < gui_frame_time:
                            time.sleep(gui_frame_time - elapsed)

                    # ============================================================
                    # PHASE 8: CHECK FOR NEWER CHECKPOINTS (if auto-reload enabled)
                    # ============================================================
                    if config.AUTO_RELOAD_MODELS:
                        if frame_start_time - last_reload_check > config.VIEWER_RELOAD_CHECK_INTERVAL:
                            last_reload_check = frame_start_time

                            try:
                                latest = get_latest_checkpoint()
                                latest_info = get_checkpoint_info(latest)

                                # If we found a newer checkpoint that's stable, reload
                                if latest_info['steps'] > checkpoint_info['steps'] and _file_is_stable(latest):
                                    print(f"\n[New checkpoint detected: {latest_info['filename']}]")
                                    print("[Reloading model...]")

                                    model = PPO.load(latest, env=env, device=VIEWER_DEVICE)
                                    if hasattr(model, 'policy'):
                                        model.policy.set_training_mode(False)
                                    checkpoint_info = latest_info

                                    print("[Model reloaded successfully!]")
                            except Exception as e:
                                print(f"[Warning: Failed to check/reload checkpoint: {e}]")

                except KeyboardInterrupt:
                    quit_requested[0] = True
                    break

    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nClosing GUI and environment...")
        gui.cleanup()
        env.close()
        print("Viewer closed.")


if __name__ == "__main__":
    watch_agent()

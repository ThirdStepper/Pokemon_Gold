# train_pokemon_gold.py
"""
Training script for Pokemon Gold RL Agent using PPO

This script trains a Proximal Policy Optimization (PPO) agent to play Pokemon Gold.

PPO is a popular RL algorithm that:
- Is relatively stable and easy to tune
- Works well with image observations (CNNs)
- Handles continuous training without catastrophic forgetting

For ML beginners: This is where the actual "learning" happens. The agent will
play the game many times, gradually improving its policy (decision-making).
"""

import os
import re
import time
from pathlib import Path
import config

# PERFORMANCE: Set thread environment variables BEFORE importing NumPy/Torch
# This prevents thread oversubscription and CPU thrashing when using parallel
# environments. Must be done before NumPy/Torch initialize their thread pools.
if config.TORCH_NUM_THREADS is not None:
    os.environ.setdefault("OMP_NUM_THREADS", str(config.TORCH_NUM_THREADS))
    os.environ.setdefault("MKL_NUM_THREADS", str(config.TORCH_NUM_THREADS))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(config.TORCH_NUM_THREADS))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(config.TORCH_NUM_THREADS))

# Now safe to import NumPy and PyTorch
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecNormalize,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from pokemon_gold_env import PokemonGoldEnv

def find_latest_checkpoint(models_dir: Path) -> tuple[Path | None, int]:
    """
    Find the most recent checkpoint file in the models directory.

    Args:
        models_dir: Path to the models directory

    Returns:
        Tuple of (checkpoint_path, steps) or (None, 0) if no checkpoints found
    """
    if not models_dir.exists():
        return None, 0

    # Pattern: pokemon_gold_ppo_<steps>_steps.zip
    checkpoint_pattern = re.compile(r"^pokemon_gold_ppo_(\d+)_steps\.zip$")

    latest_checkpoint = None
    max_steps = 0

    for file in models_dir.glob("*.zip"):
        match = checkpoint_pattern.match(file.name)
        if match:
            steps = int(match.group(1))
            if steps > max_steps:
                max_steps = steps
                latest_checkpoint = file

    return latest_checkpoint, max_steps

# Configure PyTorch threading and GPU optimizations for maximum performance
if config.TORCH_NUM_THREADS is not None:
    torch.set_num_threads(config.TORCH_NUM_THREADS)
    print(f"[CPU Threads] PyTorch using {config.TORCH_NUM_THREADS} threads")
    if hasattr(torch, "set_num_interop_threads"):
        interop_threads = max(1, config.TORCH_NUM_THREADS // 2)
        torch.set_num_interop_threads(interop_threads)
        print(f"[CPU Threads] Inter-op threads set to {interop_threads}")
else:
    print(f"[CPU Threads] Using default PyTorch threading (all cores)")

# Enable CUDA optimizations for faster training on NVIDIA GPUs
if torch.cuda.is_available():
    # cuDNN benchmark mode: Auto-tunes kernel selection for your specific hardware
    # This adds ~5-10% overhead on first few batches but speeds up steady-state training
    torch.backends.cudnn.benchmark = True

    # TF32 (TensorFloat-32): Faster math on Ampere/Ada GPUs (RTX 30xx/40xx)
    # Uses lower precision internally but maintains FP32 range (safe for RL)
    # Using new PyTorch 2.9+ API (replaces deprecated allow_tf32 settings)
    if hasattr(torch.backends.cuda.matmul, 'fp32_precision'):
        # PyTorch 2.8+ new API for TF32
        torch.backends.cuda.matmul.fp32_precision = 'tf32'
        torch.backends.cudnn.conv.fp32_precision = 'tf32'
        tf32_method = "new API"
    else:
        # Fallback for older PyTorch versions (< 2.8)
        if hasattr(torch, "set_float32_matmul_precision"):
            # PyTorch 2.0-2.7: use old API
            torch.set_float32_matmul_precision("high")
            tf32_method = "legacy API (set_float32_matmul_precision)"
        else:
            # PyTorch < 2.0: use allow_tf32 flags
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            tf32_method = "allow_tf32 flags"

    device = "cuda"
    print(f"[GPU] CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"[GPU] Enabled: cuDNN benchmark, TF32 ({tf32_method})")
else:
    device = "cpu"
    print(f"[GPU] CUDA not available, using CPU")

def make_env():
    """
    Create and configure a Pokemon Gold environment.

    This factory function is used by Stable-Baselines3 to create environment
    instances. Using a function (instead of creating env directly) allows
    for easy parallelization later.

    Returns:
        Configured PokemonGoldEnv instance
    """
    env = PokemonGoldEnv(
        rom_path=config.ROM_PATH,
        sym_path=config.SYM_PATH,
        render_mode=None,  # None = headless (faster), "rgb_array" = get frames
        frame_skip=config.FRAME_SKIP,
        max_steps=config.MAX_STEPS_PER_EPISODE,
        require_init_state=config.REQUIRE_INIT_STATE,
        input_hold_frames=config.INPUT_HOLD_FRAMES,
        post_release_frames=config.POST_RELEASE_FRAMES,
    )

    # Set the savestate path to skip intro
    if config.INIT_STATE_PATH.is_file():
        env.init_state_path = config.INIT_STATE_PATH
    else:
        print(f"[WARNING] Init savestate not found: {config.INIT_STATE_PATH}")
        print("Create one with make_boot_state.py first!")

    return env


def train():
    """
    Main training loop for the Pokemon Gold agent.

    Process:
    1. Create parallel vectorized environments for faster data collection
    2. Add monitoring and optional frame stacking
    3. Create a PPO agent with CNN policy
    4. Train for specified number of timesteps
    5. Save the final model
    """
    num_envs = config.NUM_ENVS
    use_subproc = config.USE_SUBPROC

    print(f"Creating {num_envs} parallel environment(s)...")

    # Create vectorized environments with automatic seeding
    # make_vec_env handles seeding: each env gets seed = RANDOM_SEED + env_index
    # This ensures reproducibility while giving each env unique randomness
    if num_envs > 1 and use_subproc:
        print(f"  Using SubprocVecEnv with spawn (parallel execution)")
        vec_env_cls = SubprocVecEnv
        vec_env_kwargs = {"start_method": "spawn"}
    else:
        print(f"  Using DummyVecEnv (sequential execution)")
        vec_env_cls = DummyVecEnv
        vec_env_kwargs = None

    print(f"  Random seed: {config.RANDOM_SEED} (each env gets +0, +1, +2, ...)")

    set_random_seed(config.RANDOM_SEED, using_cuda=torch.cuda.is_available())

    env = make_vec_env(
        make_env,
        n_envs=num_envs,
        seed=config.RANDOM_SEED,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs
    )

    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=config.REWARD_CLIP_ABS)

    if config.FRAME_STACK > 0:
        print(f"  Enabling frame stacking (n_stack={config.FRAME_STACK})")
        env = VecFrameStack(env, n_stack=config.FRAME_STACK, channels_order="last")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path, checkpoint_steps = find_latest_checkpoint(config.MODELS_DIR)

    if checkpoint_path is not None:
        print(f"Found checkpoint: {checkpoint_path.name}")
        print(f"  Resuming training from {checkpoint_steps:,} steps")
        print(f"  Loading model...")

        try:
            model = PPO.load(
                checkpoint_path,
                env=env,
                device=device,
                print_system_info=False
            )
            print(f"  Successfully loaded checkpoint!")
        except Exception as e:
            print(f"  WARNING: Failed to load checkpoint: {e}")
            print(f"  Starting fresh training instead...")
            checkpoint_path = None
            checkpoint_steps = 0

    if checkpoint_path is None:
        print("No checkpoint found. Creating new PPO model with MultiInput policy...")
        print(f"  Policy: MultiInputPolicy (image + game state features)")
        print(f"  Hyperparameters:")
        print(f"    Learning rate: {config.LEARNING_RATE}")
        print(f"    Steps per env: {config.N_STEPS}")
        print(f"    Batch size: {config.BATCH_SIZE}")
        print(f"    Training epochs: {config.N_EPOCHS}")
        print(f"    Device: {device}")

        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=config.LEARNING_RATE,
            n_steps=config.N_STEPS,
            batch_size=config.BATCH_SIZE,
            n_epochs=config.N_EPOCHS,
            verbose=config.TRAINING_VERBOSE_LEVEL,
            tensorboard_log=str(config.LOGS_DIR),
            device=device
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=config.CHECKPOINT_SAVE_FREQUENCY,
        save_path=str(config.MODELS_DIR),
        name_prefix="pokemon_gold_ppo"
    )

    print(f"\nStarting training...")

    # Determine training mode: time-boxed or step-based
    use_time_box = int(getattr(config, "OVERNIGHT_TRAIN_SECONDS", 0)) > 0

    if use_time_box:
        # Time-boxed training mode
        hours = config.OVERNIGHT_TRAIN_SECONDS / 3600
        print(f"  Mode: Time-boxed (~{hours:.1f} hours)")
        print(f"  Training in chunks of {config.TRAIN_CHUNK_STEPS:,} steps")
    else:
        # Step-based training mode
        print(f"  Mode: Step-based")
        print(f"  Total timesteps: {config.TOTAL_TRAINING_STEPS:,}")

    print(f"  Parallel envs: {num_envs}")
    print(f"  Steps per update: {config.N_STEPS * num_envs:,} (n_steps × num_envs)")
    print(f"  Checkpoint frequency: every {config.CHECKPOINT_SAVE_FREQUENCY:,} steps")
    print(f"  Models directory: {config.MODELS_DIR}")
    print(f"  TensorBoard logs: {config.LOGS_DIR}")
    print(f"    (Run 'tensorboard --logdir {config.LOGS_DIR}' to view progress)")
    print()

    if use_time_box:
        # Time-boxed training: run in chunks until time limit reached
        end_time = time.time() + int(config.OVERNIGHT_TRAIN_SECONDS)
        callbacks = CallbackList([checkpoint_callback])
        steps_accumulated = checkpoint_steps  # Start from checkpoint if resuming

        print(f"Training will run until approximately {time.strftime('%H:%M:%S', time.localtime(end_time))}")
        print()

        while time.time() < end_time:
            model.learn(
                total_timesteps=int(config.TRAIN_CHUNK_STEPS),
                callback=callbacks,
                reset_num_timesteps=False,  # Preserve PPO learning rate scheduler
            )

            steps_accumulated += int(config.TRAIN_CHUNK_STEPS)
            elapsed_total = time.time() - (end_time - config.OVERNIGHT_TRAIN_SECONDS)
            remaining = end_time - time.time()

            print(f"[Chunked Training] Completed chunk: {steps_accumulated:,} total steps")
            print(f"  Time elapsed: {elapsed_total/3600:.2f}h | Remaining: {remaining/3600:.2f}h")
            print()

            # Check if we have enough time for another chunk
            if remaining < 60:  # Less than 1 minute remaining
                print("Time limit reached. Stopping training.")
                break
    else:
        model.learn(
            total_timesteps=config.TOTAL_TRAINING_STEPS,
            callback=checkpoint_callback
        )

    final_model_path = config.MODELS_DIR / "pokemon_gold_ppo_final.zip"
    model.save(final_model_path)
    print(f"\nTraining complete! Final model saved to: {final_model_path}")

    return model, env


def evaluate(model, env, num_episodes=5):
    """
    Evaluate the trained model by running a few test episodes.

    Args:
        model: Trained PPO model
        env: Vectorized environment
        num_episodes: Number of episodes to run for evaluation

    Returns:
        List of episode returns (total rewards)
    """
    print(f"\nEvaluating model over {num_episodes} episodes...")
    episode_returns = []

    for episode in range(num_episodes):
        obs = env.reset()
        episode_return = 0.0
        step_count = 0

        while True:
            # deterministic=True means use the mean action (no exploration noise)
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            episode_return += float(rewards[0])
            step_count += 1

            if bool(dones[0]):
                break

        episode_returns.append(episode_return)
        print(f"  Episode {episode + 1}: Return = {episode_return:.3f}, Steps = {step_count}")

    avg_return = np.mean(episode_returns)
    std_return = np.std(episode_returns)
    print(f"\nAverage return: {avg_return:.3f} ± {std_return:.3f}")

    return episode_returns


def print_overnight_training_guide():
    """
    Print helpful instructions for overnight training runs.
    """
    print("\n" + "="*80)
    print("OVERNIGHT TRAINING GUIDE")
    print("="*80)
    print("\n📊 MONITORING TRAINING PROGRESS:")
    print("  Open a new terminal/PowerShell window and run:")
    print(f"    tensorboard --logdir {config.LOGS_DIR}")
    print("  Then open a browser to: http://localhost:6006")
    print("  to watch training metrics update in real-time!")
    print()
    print("💾 CHECKPOINT RESUME:")
    print("  If training stops (crash/interrupt), just re-run this script:")
    print("    python train_pokemon_gold.py")
    print("  It will automatically detect and resume from the latest checkpoint.")
    print()
    print("🔍 CHECKING IF TRAINING IS STILL RUNNING (next morning):")
    print("  Windows Task Manager:")
    print("    - Press Ctrl+Shift+Esc")
    print("    - Look for 'python.exe' processes")
    print("    - Check GPU utilization in Performance tab")
    print()
    print("  Command line:")
    print("    tasklist | findstr python")
    print()
    print("⏱️  TIME-BOXED TRAINING (Optional):")
    print("  To train for a specific duration instead of step count:")
    print("  1. Edit config.py:")
    print("     OVERNIGHT_TRAIN_SECONDS = 10 * 60 * 60  # 10 hours")
    print("  2. Training will stop automatically after that time")
    print("  3. Set to 0 to use step-based training instead")
    print()
    print("="*80 + "\n")


if __name__ == "__main__":
    # Print overnight training guide at startup
    print_overnight_training_guide()

    # Train the agent
    model, env = train()

    # Evaluate the trained agent
    evaluate(model, env, num_episodes=5)

    # Clean up
    env.close()
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\n✅ Final model saved to: {config.MODELS_DIR / 'pokemon_gold_ppo_final.zip'}")
    print(f"📈 Training logs available at: {config.LOGS_DIR}")
    print("\nTo view training results in TensorBoard:")
    print(f"  tensorboard --logdir {config.LOGS_DIR}")
    print("\nDone!")

# sanity_check.py
"""
Sanity check script for Pokemon Gold environment

This script verifies that:
1. The ROM file can be loaded
2. The savestate can be loaded
3. The environment can be reset
4. Actions can be taken and the game responds
5. RAM reading functions work correctly

Run this before training to ensure everything is set up correctly.
"""

from pokemon_gold_env import PokemonGoldEnv
import config
import time


def main():
    """Run a comprehensive sanity check of the environment."""
    print("=" * 60)
    print("Pokemon Gold Environment Sanity Check")
    print("=" * 60)
    print()

    # --- Check file paths ---
    print("1. Checking file paths...")
    if not config.ROM_PATH.is_file():
        print(f"   ERROR: ROM not found at {config.ROM_PATH}")
        return
    print(f"   ✓ ROM found: {config.ROM_PATH}")

    if not config.INIT_STATE_PATH.is_file():
        print(f"   WARNING: Savestate not found at {config.INIT_STATE_PATH}")
        print("   You should create one with make_boot_state.py")
    else:
        print(f"   ✓ Savestate found: {config.INIT_STATE_PATH}")
    print()

    # --- Create environment ---
    print("2. Creating environment (with window at 60 FPS)...")
    try:
        env = PokemonGoldEnv(
            rom_path=config.ROM_PATH,
            sym_path=config.SYM_PATH,
            render_mode=None,
            frame_skip=config.FRAME_SKIP,
            max_steps=config.MAX_STEPS_PER_EPISODE,
            require_init_state=config.REQUIRE_INIT_STATE,
            input_hold_frames=config.INPUT_HOLD_FRAMES,
            post_release_frames=config.POST_RELEASE_FRAMES,
            enable_window=True,  # Show the game window
            target_fps=60  # Run at Game Boy speed for visual verification
        )
        env.init_state_path = config.INIT_STATE_PATH
        print(f"   ✓ Environment created successfully")
        print(f"   - Observation space: {env.observation_space}")
        print(f"   - Action space: {env.action_space}")
        print(f"   - Window mode: Enabled at {env.target_fps} FPS")
    except Exception as e:
        print(f"   ERROR: Failed to create environment: {e}")
        return
    print()

    # --- Reset environment ---
    print("3. Resetting environment...")
    try:
        obs, info = env.reset()
        print(f"   ✓ Reset successful")
        print(f"   - Observation shape: {obs.shape}")
        print(f"   - Observation dtype: {obs.dtype}")
        print(f"   - Observation range: [{obs.min()}, {obs.max()}]")
    except Exception as e:
        print(f"   ERROR: Failed to reset: {e}")
        env.close()
        return
    print()

    # --- Read initial game state ---
    print("4. Reading initial game state from RAM...")
    try:
        xy = env.xy()
        money = env.money()
        map_bank = env.mem8(env.RAM["map_bank"])
        map_number = env.mem8(env.RAM["map_number"])
        bike_flag = env.mem8(env.RAM["bike_flag"])

        print(f"   ✓ RAM reading successful")
        print(f"   - Player position: X={xy[0]}, Y={xy[1]}")
        print(f"   - Money: ${money}")
        print(f"   - Map: Bank={map_bank}, Number={map_number}")
        print(f"   - On bike: {bool(bike_flag)}")
    except Exception as e:
        print(f"   ERROR: Failed to read RAM: {e}")
        env.close()
        return
    print()

    # --- Test actions ---
    print("5. Testing actions...")
    print("   Taking 9 actions from action_names list")
    print("   (Watch the game window to see the actions being executed)")
    print()
    addr, mask = env.gold_event_flag_addr_mask("EVENT_GOT_A_POKEMON_FROM_ELM")

    print(f"Got a pokemon from elm? {hex(addr), hex(mask)}")

    # Action mapping from action names to action numbers
    action_map = {
        "NOOP": 0,
        "UP": 1,
        "DOWN": 2,
        "LEFT": 3,
        "RIGHT": 4,
        "A": 5,
        "B": 6,
        "START": 7
    }

    action_names = ["NOOP", "DOWN", "NOOP", "DOWN", "UP", "NOOP", "START", "B", "NOOP", "LEFT", "RIGHT", "NOOP"]
    for i, action_name in enumerate(action_names):
        try:
            action = action_map[action_name]
            print(f"\n   Step {i+1}: {action_name} (action {action}):", end=" ", flush=True)

            # Extra debug for START action
            if action_name == "START":
                print(f"\n   DEBUG: About to call env.step({action}) for START button")

            obs, reward, _, _, _ = env.step(action)
            new_xy = env.xy()
            new_money = env.money()

            print(f"Position: {new_xy}, Money: ${new_money}, Reward: {reward:.4f}")

            # Extra debug after START action
            if action_name == "START":
                print(f"   DEBUG: START action completed")

            # Pause briefly so you can see the action in the window
            #time.sleep(0.5)

            xy = new_xy
            money = new_money

        except Exception as e:
            print(f"\n   ERROR on step {i+1} ({action_name}): {e}")
            env.close()
            return
    print()

    # --- Test observation consistency ---
    print("6. Testing observation consistency...")
    obs1 = env._get_obs()
    obs2 = env._get_obs()
    if (obs1 == obs2).all():
        print(f"   ✓ Observations are consistent (same state = same observation)")
    else:
        print(f"   WARNING: Observations differ without action (may indicate randomness)")
    print()

    # --- Test episode truncation ---
    print("7. Testing episode truncation...")
    print(f"   Current step: {env._steps} / {env.max_steps}")
    if env._steps < env.max_steps:
        print(f"   ✓ Episode has not truncated yet")
    else:
        print(f"   WARNING: Episode already truncated")
    print()

    # --- Cleanup ---
    print()
    print("=" * 60)
    print("✓ All sanity checks passed!")
    print("=" * 60)
    print()
    print("The game window will remain open for 3 seconds...")
    time.sleep(3)

    env.close()
    print("Window closed.")
    print()
    print("Next steps:")
    print("1. Adjust hyperparameters in config.py if needed")
    print("2. Run train_pokemon_gold.py to start training (headless mode)")
    print("3. While training, run watch_agent.py to see the agent in action")
    print("4. Monitor progress with TensorBoard:")
    print(f"   tensorboard --logdir {config.LOGS_DIR}")
    print()


if __name__ == "__main__":
    main()

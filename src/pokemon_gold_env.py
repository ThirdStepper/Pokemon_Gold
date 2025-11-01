# pokemon_gold_env.py
"""
Pokemon Gold Gymnasium Environment

This module creates a reinforcement learning environment for Pokemon Gold using
the PyBoy Game Boy emulator. The agent observes downsampled grayscale screen
images and can press Game Boy buttons to navigate the game.

For ML beginners: This is a "Gym Environment" - it's a standardized way to
interact with a game/simulation for reinforcement learning.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from io import BytesIO
from typing import Optional, Tuple, Set, List, Dict
import time
from datetime import datetime

# Import configuration settings
import config

# Import environment mixins
from gold_env import (
    RAMHelpersMixin,
    EventFlagsMixin,
    SavestateUtilsMixin,
    ExplorationMixin,
    RewardsMixin
)

# Map agent actions (0-6) to Game Boy button presses
# Format: (action_name, (press_event, release_event))
# NOOP removed - forces agent to always take meaningful actions
ACTIONS = [
    ("UP",    (WindowEvent.PRESS_ARROW_UP,    WindowEvent.RELEASE_ARROW_UP)),
    ("DOWN",  (WindowEvent.PRESS_ARROW_DOWN,  WindowEvent.RELEASE_ARROW_DOWN)),
    ("LEFT",  (WindowEvent.PRESS_ARROW_LEFT,  WindowEvent.RELEASE_ARROW_LEFT)),
    ("RIGHT", (WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT)),
    ("A",     (WindowEvent.PRESS_BUTTON_A,    WindowEvent.RELEASE_BUTTON_A)),
    ("B",     (WindowEvent.PRESS_BUTTON_B,    WindowEvent.RELEASE_BUTTON_B)),
    ("START", (WindowEvent.PRESS_BUTTON_START,WindowEvent.RELEASE_BUTTON_START)),
]

class PokemonGoldEnv(gym.Env, RAMHelpersMixin, EventFlagsMixin,
                     SavestateUtilsMixin, ExplorationMixin, RewardsMixin):
    """
    Custom Gymnasium environment for Pokemon Gold using PyBoy emulator.

    This environment allows an RL agent to play Pokemon Gold by:
    - Observing: Downsampled grayscale screenshots (72x80 pixels)
    - Acting: Pressing Game Boy buttons (D-pad, A, B, Start)
    - Rewarding: Getting points for exploration (movement) and progress (money)

    Attributes:
        observation_space: Dict with 'image' and 'game_state' - multi-input observations
        action_space: Discrete(7) - one of 7 possible button presses (NOOP removed)

    Inherits from:
        gym.Env: Base Gymnasium environment interface
        RAMHelpersMixin: RAM reading utilities
        EventFlagsMixin: Event flag parsing and reading
        SavestateUtilsMixin: Savestate management
        ExplorationMixin: Anti-stuck exploration mechanisms
        RewardsMixin: Comprehensive reward calculation
    """
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        rom_path: Path,
        sym_path: Path,
        render_mode: Optional[str] = None,
        frame_skip: int = config.FRAME_SKIP,
        max_steps: int = config.MAX_STEPS_PER_EPISODE,
        require_init_state: bool = config.REQUIRE_INIT_STATE,
        input_hold_frames: int = config.INPUT_HOLD_FRAMES,
        post_release_frames: int = config.POST_RELEASE_FRAMES,
        menu_open_extra_frames: int = config.MENU_OPEN_EXTRA_FRAMES,
        enable_window: bool = False,
        target_fps: int = config.TARGET_FPS,
    ):
        """
        Initialize the Pokemon Gold environment.

        Args:
            rom_path: Path to the Pokemon Gold ROM file (.gbc)
            sym_path: Path to the Pokemon Gold SYM file (.sym)
            render_mode: How to render (None for headless, "rgb_array" for frames)
            frame_skip: How many emulator frames to skip between observations
            max_steps: Maximum steps before episode ends (prevents infinite loops)
            require_init_state: If True, must provide a savestate file to skip intro
            input_hold_frames: How long to hold button down (so game registers it)
            post_release_frames: How long to wait after releasing button (avoid chatter)
            menu_open_extra_frames: Extra frames to wait after START button for menu to open
            enable_window: If True, show PyBoy window (SDL2)
            target_fps: Target frames per second when window is enabled (0 = unlimited)
        """
        super().__init__()
        self.rom = Path(rom_path)
        if not self.rom.is_file():
            raise FileNotFoundError(f"ROM not found: {self.rom}")
        
        self.sym = Path(sym_path)
        if not self.sym.is_file():
            raise FileNotFoundError(f"SYM not found: {self.sym}")

        self.obs_h = config.OBSERVATION_HEIGHT
        self.obs_w = config.OBSERVATION_WIDTH
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0, high=255,
                shape=(self.obs_h, self.obs_w, 1),
                dtype=np.uint8
            ),
            'game_state': spaces.Box(
                low=0.0, high=1.0,
                shape=(15,),
                dtype=np.float32
            )
        })
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.input_hold_frames = int(input_hold_frames)
        self.post_release_frames = int(post_release_frames)
        self.menu_open_extra_frames = int(menu_open_extra_frames)
        self.require_init_state = require_init_state
        self.enable_window = enable_window
        self.target_fps = target_fps
        self.smooth_render = config.VIEWER_SMOOTH_RENDER
        self.render_every_n = config.RENDER_EVERY_N_FRAMES
        self.frame_time = 1.0 / target_fps if target_fps > 0 else 0
        self.last_frame_time = 0
        self.fast_forward = False
        self.sticky_dpad_enabled = config.STICKY_DPAD_ENABLED
        self.move_hold_frames = config.MOVE_HOLD_FRAMES
        self._held_pair = None
        self._next_frame_time = None
        self._steps = 0
        self.pyboy = None
        self.screen = None
        self.mem = None
        self.boot_state: Optional[BytesIO] = None
        self.init_state_path: Optional[Path] = None
        self.RAM = config.RAM_MAP

        self._last_xy: Optional[Tuple[int, int]] = None
        self._last_money: Optional[int] = None
        self._visited_tiles: Set[Tuple[int, int]] = set()
        self._visited_map_banks: Set[int] = set()
        self._visited_map_numbers: Set[int] = set()
        self._last_party_count: Optional[int] = None
        self._last_total_levels: Optional[int] = None
        self._last_party_data: Optional[List[dict]] = None
        self._last_badge_count: Optional[int] = None
        self._visited_world_tiles: Set[Tuple[int, int]] = set()
        self._completed_plot_flags: Set[str] = set()
        self._seen_species: Set[int] = set()
        self._caught_species: Set[int] = set()
        self._last_action: Optional[int] = None
        self._collision_count: int = 0
        self._total_collisions: int = 0
        self._npc_front_seen_step: int = -1
        self._bump_seen_step: int = -1
        self._last_collision_code: int = 0
        self._enter_map_warp_seen_step: int = -1
        self._enter_map_connection_seen_step: int = -1
        self._plot_flag_last_values: Optional[Dict[str, int]] = None
        self._plot_flag_initial_values: Optional[Dict[str, int]] = None
        self._plot_flags_log_file = None
        self._dialog_depth: int = 0
        self._battle_depth: int = 0
        self._pause_penalties: bool = False
        self._tile_visit_counts: Dict[Tuple[int, int], int] = {}
        self._tile_last_visit: Dict[Tuple[int, int], int] = {}
        self._step_count: int = 0
        from collections import deque
        self._position_history: deque = deque(maxlen=config.STUCK_DETECTION_WINDOW)
        self._action_history: deque = deque(maxlen=config.ACTION_DIVERSITY_WINDOW)
        self._spawn_position: Optional[Tuple[int, int]] = None
        self._last_distance_check_step: int = 0

    def _update_pause_state(self):
        """
        Update pause_penalties flag based on dialog/battle depth.

        This flag is checked by the reward system to skip collision penalties,
        stuck detection, and other anti-stuck mechanisms during dialogues/battles.
        """
        self._pause_penalties = (self._dialog_depth > 0) or (self._battle_depth > 0)

    def _on_open_text(self, _):
        """
        Callback when OpenText is called (dialogue box opens).

        This hook fires when the game displays a textbox for NPC dialogue,
        signs, items, or any other text interaction.
        """
        self._dialog_depth += 1
        self._update_pause_state()

    def _on_close_text(self, _):
        """
        Callback when CloseText is called (dialogue box closes).

        This hook fires when the textbox is dismissed and the player
        regains control.
        """
        self._dialog_depth = max(0, self._dialog_depth - 1)
        self._update_pause_state()

    def _on_start_battle(self, _):
        """
        Callback when StartBattle is called (battle begins).

        This hook fires when transitioning into a wild Pokemon battle or
        trainer battle, before the battle screen is fully initialized.
        """
        self._battle_depth += 1
        self._update_pause_state()

    def _on_exit_battle(self, _):
        """
        Callback when ExitBattle is called (battle ends).

        This hook fires when exiting a battle (win, loss, or flee) and
        returning to the overworld.
        """
        self._battle_depth = max(0, self._battle_depth - 1)
        self._update_pause_state()

    def _init_emulator(self):
        """
        Initialize the PyBoy emulator instance.

        Creates either a headless emulator (for training) or windowed emulator
        (for visualization) based on the enable_window parameter.
        """
        window_type = "SDL2" if self.enable_window else "null"
        self.pyboy = PyBoy(
            str(self.rom),
            symbols=str(self.sym),
            window=window_type,
            sound_emulated=False
        )
        self.screen = self.pyboy.screen
        self.mem = self.pyboy.memory
        self.pyboy.hook_register(None, "OpenText", self._on_open_text, self)
        self.pyboy.hook_register(None, "CloseText", self._on_close_text, self)
        self.pyboy.hook_register(None, "StartBattle", self._on_start_battle, self)
        self.pyboy.hook_register(None, "ExitBattle", self._on_exit_battle, self)
        self.pyboy.hook_register(None, "IsNPCAtCoord.yes", self._on_isnpc_yes, self)
        self.pyboy.hook_register(None, "DoPlayerMovement.bump", self._on_bump, self)
        self.pyboy.hook_register(None, "EnterMapWarp", self._on_enter_map_warp, self)
        self.pyboy.hook_register(None, "EnterMapConnection", self._on_enter_map_connection, self)
        self.last_frame_time = time.time()

    def _limit_frame_rate(self, frames_advanced: int = 1):
        """
        Limit frame rate to target FPS when in window mode (action-level pacing).

        This method sleeps once per action based on how many emulator frames
        were advanced during that action, providing consistent pacing without
        the overhead of sleeping on every tick.

        Args:
            frames_advanced: How many Game Boy frames were simulated during this action

        Performance note:
            Old approach: Sleep N times per action (once per tick) → high overhead
            New approach: Sleep once per action → minimal overhead
        """
        if not self.enable_window or self.target_fps <= 0 or self.fast_forward:
            return

        target_seconds = frames_advanced / 60.0
        now = time.perf_counter()

        if self._next_frame_time is None:
            self._next_frame_time = now + target_seconds
            return

        sleep_seconds = self._next_frame_time - now
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
            now = time.perf_counter()

        self._next_frame_time = max(now, self._next_frame_time) + target_seconds

    def _release_held_if_any(self):
        """
        Release any currently held D-pad button.

        This is called before non-movement actions (A/B/Start) to ensure
        the agent stops walking before interacting with menus or objects.
        """
        if self._held_pair is not None:
            _, held_release = self._held_pair
            self.pyboy.send_input(held_release)
            self._held_pair = None

    def _on_isnpc_yes(self, _):
        self._npc_front_seen_step = self._step_count

    def _on_bump(self, _):
        self._bump_seen_step = self._step_count

    def _on_enter_map_warp(self, _):
        self._enter_map_warp_seen_step = self._step_count

    def _on_enter_map_connection(self, _):
        self._enter_map_connection_seen_step = self._step_count

    def reset(self, seed=None, options=None):
        """
        Reset the environment to start a new episode.

        This is called at the beginning of each episode. It:
        1. Initializes the emulator if needed
        2. Runs warmup frames to let boot screens settle
        3. Loads a savestate to skip the intro (if available)
        4. Resets episode tracking variables

        Args:
            seed: Random seed for reproducibility (not used here)
            options: Additional options (not used here)

        Returns:
            observation: The initial game screen (72x80 grayscale image)
            info: Dictionary of additional info (empty for now)
        """
        super().reset(seed=seed)

        if self.pyboy is None:
            self._init_emulator()

        for _ in range(config.BOOT_WARMUP_FRAMES):
            self.pyboy.tick()

        if not self._load_init_state_if_any():
            if self.require_init_state:
                raise RuntimeError(
                    "No init savestate found. Create one with make_boot_state.py "
                    "and set env.init_state_path to that file."
                )

        self.pyboy.tick(1, True)
        self._steps = 0
        obs = self._get_obs()
        self._last_xy = self.xy()
        self._last_money = self.money()
        self._visited_tiles = {self.xy()}
        self._visited_map_banks = {self.mem8(self.RAM["map_bank"])}
        self._visited_map_numbers = {self.mem8(self.RAM["map_number"])}
        self._last_party_count = len(self.read_party_pokemon())
        party_data = self.read_party_pokemon()
        self._last_total_levels = sum(p["level"] for p in party_data)
        self._last_party_data = party_data
        self._last_badge_count = self.get_badge_count()
        self._visited_world_tiles = {self.get_world_xy()}
        self._completed_plot_flags = set()
        for flag_name in config.PLOT_FLAG_REWARDS.keys():
            if self.check_plot_flag(flag_name):
                self._completed_plot_flags.add(flag_name)
        self._seen_species = self.get_seen_species()
        self._caught_species = self.get_caught_species()

        if config.DEBUG_PLOT_FLAGS:
            config.PLOT_FLAGS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            self._plot_flags_log_file = open(config.PLOT_FLAGS_LOG_PATH, 'a')
            self._plot_flags_log_file.write(f"\n{'='*80}\n")
            self._plot_flags_log_file.write(f"EPISODE RESET - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self._plot_flags_log_file.write(f"{'='*80}\n\n")
            initial_values = {}
            self._plot_flags_log_file.write("INITIAL PLOT FLAG STATE:\n")
            for flag_name in config.PLOT_FLAG_REWARDS.keys():
                if flag_name in self.RAM:
                    addr = self.RAM[flag_name]
                    value = self.mem8(addr)
                    initial_values[flag_name] = value
                    self._plot_flags_log_file.write(
                        f"  {flag_name:25s} @ {addr:#06x}: {value:#04x} (dec: {value:3d}, bin: {value:08b})\n"
                    )
            self._plot_flags_log_file.write("\n")
            self._plot_flags_log_file.flush()
            self._plot_flag_initial_values = initial_values
            self._plot_flag_last_values = initial_values.copy()
        else:
            self._plot_flag_last_values = None
            self._plot_flag_initial_values = None

        self._last_action = None
        self._collision_count = 0
        self._total_collisions = 0
        self._dialog_depth = 0
        self._battle_depth = 0
        self._pause_penalties = False
        self._tile_visit_counts.clear()
        self._tile_last_visit.clear()
        self._step_count = 0
        self._position_history.clear()
        self._action_history.clear()
        self._spawn_position = self.xy()
        self._last_distance_check_step = 0

        info = {}

        return obs, info

    def _get_obs(self) -> dict:
        """
        Get the current observation (multi-input: image + game state).

        Process:
        1. Get RGBA frame from emulator (144x160x4)
        2. Convert to grayscale using integer arithmetic (faster than float mean)
        3. Downsample by 2x (every other pixel) → 72x80
        4. Add channel dimension for CNN compatibility
        5. Extract game state features from RAM

        Returns:
            Dict with keys:
                'image': Grayscale image array of shape (72, 80, 1) with values 0-255
                'game_state': Feature vector of shape (15,) with normalized values 0-1

        Performance note:
        Uses integer arithmetic (R+G+B)//3 instead of float64 mean() for ~10-20%
        speedup. With millions of observations, this adds up significantly.
        """
        # Get the current frame from the emulator (zero-copy view)
        frame = np.array(self.screen.ndarray, copy=False)

        # Convert RGBA to grayscale using integer arithmetic
        # Use uint16 to avoid overflow when adding RGB channels (max 255*3 = 765)
        rgb_u16 = frame[..., :3].astype(np.uint16, copy=False)
        gray = (rgb_u16[..., 0] + rgb_u16[..., 1] + rgb_u16[..., 2]) // 3

        # Downsample by 2 and add channel dimension; cast once at the end
        image = gray[::2, ::2, np.newaxis].astype(np.uint8)  # Shape: (72, 80, 1)

        # Extract game state features
        game_state = self._get_game_state_features()

        return {
            'image': image,
            'game_state': game_state
        }

    def _get_game_state_features(self) -> np.ndarray:
        """
        Extract game state features from RAM for multi-input policy.

        This provides explicit game state information that would be difficult
        for the agent to extract from pixels alone. Features are normalized
        to 0-1 range for neural network compatibility.

        Returns:
            np.ndarray of shape (15,) with dtype float32

        Features (15 total):
            0-2: NPC interaction state (near_npc, in_script, in_textbox)
            3-5: Movement state (is_walking, just_collided, at_door)
            6-9: Facing direction (one-hot: down, up, left, right)
            10-14: Progress metrics (badges, party_size, in_battle, episode_progress, tile_collision)
        """
        near_npc = float(self._npc_front_seen_step == self._step_count)
        in_script = float(self.mem8(self.RAM["script_running"]) > 0)
        in_textbox = float(self._dialog_depth > 0)

        step_dir = self.mem8(self.RAM["player_step_direction"])
        is_walking = float(step_dir != 255)
        collision_code = self.mem8(self.RAM["walking_tile_collision"])
        self._last_collision_code = collision_code

        solid_hits = {7, 21, 145, 149, 151}
        just_collided = float((self._bump_seen_step == self._step_count) or (collision_code in solid_hits))

        warp_pulse = (
            self._enter_map_warp_seen_step == self._step_count
            or self._enter_map_connection_seen_step == self._step_count
        )
        at_door = float(collision_code in {112, 113, 122} or warp_pulse)

        if is_walking:
            d_down = float(step_dir == 0)
            d_up    = float(step_dir == 1)
            d_left  = float(step_dir == 2)
            d_right = float(step_dir == 3)
        else:
            pdir = self.mem8(self.RAM["player_direction"])
            d_down  = float(pdir == 0)
            d_up    = float(pdir == 4)
            d_left  = float(pdir == 8)
            d_right = float(pdir == 12)

        badge_progress = self.get_badge_count() / 8.0
        party_size = len(self.read_party_pokemon()) / 6.0
        in_battle = float(self.mem8(self.RAM["battle_mode"]) > 0)
        episode_progress = min(self._step_count / 1000.0, 1.0)
        tile_collision = float(collision_code > 0)

        return np.array([
            near_npc, in_script, in_textbox,
            is_walking, just_collided, at_door,
            d_down, d_up, d_left, d_right,
            badge_progress, party_size, in_battle, episode_progress, tile_collision
        ], dtype=np.float32)

    def _should_render(self, i: int, last_idx: int) -> bool:
        if not getattr(self, "enable_window", False):
            return False
        if not self.smooth_render:
            return i == last_idx
        n = max(1, self.render_every_n)
        return ((i + 1) % n) == 0

    def _handle_action_input(self, action: int) -> None:
        """
        Execute a button press action in the emulator with sticky D-pad support.

        Sticky D-pad (when enabled):
        - D-pad actions (UP/DOWN/LEFT/RIGHT): Hold button across steps until
          agent chooses a different action or NOOP. This allows proper walking.
        - Non-D-pad actions (A/B/Start): Short press/release, and release any
          held D-pad first (so agent stops walking before interacting).
        - NOOP: Release any held D-pad and idle.

        Traditional mode (when sticky D-pad disabled):
        - All actions are short press/hold/release sequences

        Args:
            action: The action index (0-7, corresponding to ACTIONS list)
        """
        name, pair = ACTIONS[action]
        is_dpad = name in ("UP", "DOWN", "LEFT", "RIGHT")

        if name == "NOOP":
            self._release_held_if_any()
            for i in range(self.frame_skip):
                self.pyboy.tick(1, self._should_render(i, self.frame_skip - 1))
            # Throttle once per action
            self._limit_frame_rate(frames_advanced=self.frame_skip)
            return

        press, release = pair

        if self.sticky_dpad_enabled and is_dpad:
            if self._held_pair is not None and self._held_pair[0] == press:
                for i in range(self.move_hold_frames):
                    self.pyboy.tick(1, self._should_render(i, self.move_hold_frames - 1))
                self._limit_frame_rate(frames_advanced=self.move_hold_frames)
                return

            if self._held_pair is not None and self._held_pair[0] != press:
                self._release_held_if_any()

            self.pyboy.send_input(press)
            self._held_pair = (press, release)
            for i in range(self.move_hold_frames):
                self.pyboy.tick(1, self._should_render(i, self.move_hold_frames - 1))
            self._limit_frame_rate(frames_advanced=self.move_hold_frames)
            return

        if not is_dpad:
            self._release_held_if_any()

        self.pyboy.send_input(press)
        for i in range(self.input_hold_frames):
            self.pyboy.tick(1, self._should_render(i, self.input_hold_frames - 1))

        self.pyboy.send_input(release)
        for i in range(self.post_release_frames):
            self.pyboy.tick(1, self._should_render(i, self.post_release_frames - 1))

        extra_frames = 0
        if name == "START":
            extra_frames = self.menu_open_extra_frames
            for i in range(extra_frames):
                self.pyboy.tick(1, self._should_render(i, extra_frames - 1))

        total_frames = self.input_hold_frames + self.post_release_frames + extra_frames
        self._limit_frame_rate(frames_advanced=total_frames)

    def step(self, action: int):
        """
        Execute one action in the environment and return the result.

        This is the main interaction loop for RL:
        1. Agent chooses an action (0-7)
        2. env executes that action in the emulator
        3. env observes the new game state
        4. env calculates a reward based on what changed
        5. env checks if the episode is done

        Args:
            action: Integer action (0-7) corresponding to ACTIONS list

        Returns:
            observation: New game screen after action (72x80 grayscale image)
            reward: Numeric reward for this step (higher is better)
            terminated: Boolean, True if episode ended naturally (not used yet)
            truncated: Boolean, True if episode hit max_steps limit
            info: Dictionary of extra diagnostic info (empty for now)
        """
        pos_before = self.xy()
        self._handle_action_input(action)
        obs = self._get_obs()

        pos_after = self.xy()
        is_movement_action = 0 <= action <= 3
        collision_detected = is_movement_action and (pos_before == pos_after)

        if collision_detected:
            self._collision_count += 1
            self._total_collisions += 1
        else:
            self._collision_count = 0

        self._last_action = action
        self._action_history.append(action)
        self._position_history.append(pos_after)
        self._step_count += 1

        if config.VERIFY_PAUSE_STATE:
            textbox_active = self.mem8(self.RAM["textbox_flags"]) != 0
            if textbox_active and not self._pause_penalties:
                self._dialog_depth = 1
                self._update_pause_state()

        reward = self._calculate_reward()
        self._steps += 1
        terminated = False
        truncated = self._steps >= self.max_steps

        info = {
            "collision": collision_detected,
            "collision_count": self._collision_count,
            "total_collisions": self._total_collisions,
            "is_dialog": bool(self._pause_penalties),
            "textbox_flags": self.mem8(self.RAM["textbox_flags"]),
            "door_penalty": getattr(self, "_last_door_penalty", 0.0),
            "door_event_count": getattr(self, "_last_door_event_count", 0),
            "door_warp_pulse": getattr(self, "_last_warp_pulse", False),
            "door_collision_code": getattr(self, "_last_door_collision_code", 0),
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        """
        Return the current game screen as an RGB array.

        This is called by external code that wants to visualize the game
        (e.g., for recording videos or displaying in a GUI).

        Returns:
            RGBA numpy array of shape (144, 160, 4)
        """
        return np.array(self.screen.ndarray, copy=False)

    def close(self):
        """
        Call this when done with the environment to free resources.
        """
        if self._plot_flags_log_file and not self._plot_flags_log_file.closed:
            self._plot_flags_log_file.close()
            self._plot_flags_log_file = None

        if self.pyboy:
            self.pyboy.stop()
            self.pyboy = None
            self.screen = None
            self.mem = None
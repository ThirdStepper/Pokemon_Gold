# viewer_gui.py
"""
Dear PyGui-based GUI for Pokemon Gold RL Agent Viewer.

This module provides a tabbed interface for visualizing agent performance with:
- Tab 1: Overview (model info, episode stats, game state)
- Tab 2: Framestack (4 stacked frames visualization)
- Tab 3: Advanced (reward graphs, action distribution, RAM inspector)
"""

import dearpygui.dearpygui as dpg
import numpy as np
from collections import deque
from typing import Optional, Callable, Dict, List
import config
from gui_helpers import (
    numpy_to_dpg_texture,
    extract_frames_from_stack,
    create_texture,
    update_texture,
    scale_image_for_display,
    format_ram_value,
    create_reward_plot,
    update_reward_plot,
    create_action_bar_chart,
    update_action_bar_chart,
    create_collapsible_section,
)


def unwrap_env(env):
    """
    Properly unwrap a vectorized + frame-stacked environment.

    The environment may be wrapped in multiple layers:
    - VecFrameStack (outermost, has 'venv' attribute)
    - VecEnv (has 'envs' list attribute)
    - PokemonGoldEnv (actual environment)

    This function unwraps through all layers to get the base environment.

    Args:
        env: Wrapped environment (VecFrameStack or VecEnv or PokemonGoldEnv)

    Returns:
        Base PokemonGoldEnv with all methods accessible
    """
    # Step 1: Unwrap VecFrameStack (has 'venv' attribute pointing to underlying VecEnv)
    vec_env = env.venv if hasattr(env, 'venv') else env

    # Step 2: Unwrap VecEnv (has 'envs' list attribute)
    base_env = vec_env.envs[0] if hasattr(vec_env, 'envs') else vec_env

    return base_env


class ViewerGUI:
    """
    Main GUI class for Pokemon Gold RL Agent Viewer.

    Manages a Dear PyGui window with tabbed interface showing:
    - Real-time agent statistics
    - Framestack visualization
    - Advanced debugging tools (reward graphs, action distribution, RAM inspector)
    """

    def __init__(
        self,
        width: int = 1200,
        height: int = 800,
        on_reset: Optional[Callable] = None,
        on_pause_toggle: Optional[Callable] = None,
        on_fast_forward_toggle: Optional[Callable] = None,
        on_quit: Optional[Callable] = None,
        on_autoreload_toggle: Optional[Callable] = None,
        on_step_once: Optional[Callable] = None,
        on_save_state: Optional[Callable] = None,
        on_load_state: Optional[Callable] = None,
        on_rewind: Optional[Callable] = None,
    ):
        """
        Initialize the viewer GUI.

        Args:
            width: Window width in pixels
            height: Window height in pixels
            on_reset: Callback for reset button
            on_pause_toggle: Callback for pause button
            on_fast_forward_toggle: Callback for fast forward button
            on_quit: Callback for quit button
            on_autoreload_toggle: Callback for auto-reload toggle
            on_step_once: Callback for single-step button
            on_save_state: Callback for save state button
            on_load_state: Callback for load state button
            on_rewind: Callback for rewind button
        """
        self.width = width
        self.height = height

        # Callbacks
        self.on_reset = on_reset
        self.on_pause_toggle = on_pause_toggle
        self.on_fast_forward_toggle = on_fast_forward_toggle
        self.on_quit = on_quit
        self.on_autoreload_toggle = on_autoreload_toggle
        self.on_step_once = on_step_once
        self.on_save_state = on_save_state
        self.on_load_state = on_load_state
        self.on_rewind = on_rewind

        # State tracking
        self.paused = False
        self.fast_forward = False
        self.auto_reload = config.AUTO_RELOAD_MODELS

        # Data tracking for charts
        self.reward_history_x = deque(maxlen=100)
        self.reward_history_y = deque(maxlen=100)
        self.action_counts = [0] * 7  # 7 actions
        self.step_counter = 0

        # Value caching for optimization (only update widgets when values change)
        self._cached_values = {}

        # RAM inspector throttling (most expensive operation)
        self._last_ram_update = 0.0
        self._ram_update_interval = 0.2  # Update RAM inspector at 5 FPS

        # FPS tracking
        self._last_fps_update = 0.0
        self._fps_update_interval = 0.5  # Update FPS display every 0.5s
        self._env_fps = 0.0  # Externally set by watch_agent.py
        self._gui_frame_count = 0
        self._last_gui_frame_count = 0
        self._last_gui_fps_calc_time = 0.0

        # Initialize Dear PyGui
        dpg.create_context()
        self._create_button_themes()
        self._setup_gui()

    def _create_button_themes(self):
        """Create colored themes for pause and fast forward buttons."""
        # Pause button theme (amber/orange when paused)
        with dpg.theme() as self._pause_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (200, 120, 0))  # Amber
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (230, 140, 0))  # Lighter amber
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (170, 100, 0))  # Darker amber

        # Fast forward button theme (yellow when fast forwarding)
        with dpg.theme() as self._ff_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (200, 200, 0))  # Yellow
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (230, 230, 0))  # Lighter yellow
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (170, 170, 0))  # Darker yellow

    def _setup_gui(self):
        """Set up the Dear PyGui interface with all tabs and controls."""
        # Create main window
        with dpg.window(
            label="Pokemon Gold RL Agent Viewer",
            tag="main_window",
            width=self.width,
            height=self.height,
            no_close=True,
            no_collapse=True,
        ):
            # Control bar at top
            with dpg.group(horizontal=True):
                dpg.add_button(label="Reset (R)", callback=self._on_reset_clicked, width=100)
                dpg.add_button(
                    label="Pause (SPACE)",
                    callback=self._on_pause_clicked,
                    width=120,
                    tag="btn_pause"
                )
                dpg.add_button(
                    label="Step Once",
                    callback=self._on_step_once_clicked,
                    width=100,
                    tag="btn_step_once",
                    show=False  # Hidden by default, shown when paused
                )
                dpg.add_button(
                    label="Fast Forward (F)",
                    callback=self._on_ff_clicked,
                    width=140,
                    tag="btn_ff"
                )
                dpg.add_spacer(width=10)
                dpg.add_button(label="Save (S)", callback=self._on_save_clicked, width=90)
                dpg.add_button(label="Load (L)", callback=self._on_load_clicked, width=90)
                dpg.add_button(label="Rewind (←)", callback=self._on_rewind_clicked, width=110, tag="btn_rewind")
                dpg.add_spacer(width=10)
                dpg.add_button(label="Quit (Q)", callback=self._on_quit_clicked, width=100)
                dpg.add_spacer(width=10)
                dpg.add_checkbox(
                    label="Auto-reload Models",
                    default_value=self.auto_reload,
                    callback=self._on_autoreload_toggled,
                    tag="chk_autoreload"
                )

            dpg.add_separator()

            # Tab bar
            with dpg.tab_bar(tag="tab_bar"):
                # Tab 1: Overview
                with dpg.tab(label="Overview", tag="tab_overview"):
                    self._create_overview_tab()

                # Tab 2: Framestack
                with dpg.tab(label="Framestack", tag="tab_framestack"):
                    self._create_framestack_tab()

                # Tab 3: Advanced
                with dpg.tab(label="Advanced", tag="tab_advanced"):
                    self._create_advanced_tab()

            dpg.add_separator()

            # Status bar at bottom
            with dpg.group(horizontal=True):
                dpg.add_text("Status: ", tag="status_bar")

    def _create_overview_tab(self):
        """Create the Overview tab with current stats."""
        with dpg.child_window(width=-1, height=-1, tag="overview_content"):
            # Model Information Section
            with dpg.collapsing_header(label="Model Information", default_open=True):
                with dpg.table(header_row=False, borders_innerH=True, borders_outerH=True,
                              borders_innerV=True, borders_outerV=True):
                    dpg.add_table_column()
                    dpg.add_table_column()

                    with dpg.table_row():
                        dpg.add_text("Checkpoint:")
                        dpg.add_text("Loading...", tag="txt_checkpoint")

                    with dpg.table_row():
                        dpg.add_text("Training Steps:")
                        dpg.add_text("0", tag="txt_training_steps")

            dpg.add_spacer(height=10)

            # Episode Information Section
            with dpg.collapsing_header(label="Episode Information", default_open=True):
                with dpg.table(header_row=False, borders_innerH=True, borders_outerH=True,
                              borders_innerV=True, borders_outerV=True):
                    dpg.add_table_column()
                    dpg.add_table_column()

                    with dpg.table_row():
                        dpg.add_text("Status:")
                        dpg.add_text("RUNNING", tag="txt_status", color=(0, 255, 0))

                    with dpg.table_row():
                        dpg.add_text("Episode:")
                        dpg.add_text("0", tag="txt_episode")

                    with dpg.table_row():
                        dpg.add_text("Steps:")
                        dpg.add_text("0 / 5000", tag="txt_steps")

                    with dpg.table_row():
                        dpg.add_text("Reward:")
                        dpg.add_text("0.000", tag="txt_reward")

            dpg.add_spacer(height=10)

            # Game State Section
            with dpg.collapsing_header(label="Game State", default_open=True):
                with dpg.table(header_row=False, borders_innerH=True, borders_outerH=True,
                              borders_innerV=True, borders_outerV=True):
                    dpg.add_table_column()
                    dpg.add_table_column()

                    with dpg.table_row():
                        dpg.add_text("Position:")
                        dpg.add_text("X=0, Y=0", tag="txt_position")

                    with dpg.table_row():
                        dpg.add_text("Map:")
                        dpg.add_text("Bank=0, Number=0", tag="txt_map")

                    with dpg.table_row():
                        dpg.add_text("Money:")
                        dpg.add_text("$0", tag="txt_money")

                    with dpg.table_row():
                        dpg.add_text("On Bike:")
                        dpg.add_text("False", tag="txt_bike")

            dpg.add_spacer(height=10)

            # Statistics Section
            with dpg.collapsing_header(label="Statistics", default_open=True):
                with dpg.table(header_row=False, borders_innerH=True, borders_outerH=True,
                              borders_innerV=True, borders_outerV=True):
                    dpg.add_table_column()
                    dpg.add_table_column()

                    with dpg.table_row():
                        dpg.add_text("Step Progress:")
                        dpg.add_text("", tag="txt_step_progress")

                    with dpg.table_row():
                        dpg.add_text("Env FPS:")
                        dpg.add_text("0.0", tag="txt_env_fps")

                    with dpg.table_row():
                        dpg.add_text("GUI FPS:")
                        dpg.add_text("0.0", tag="txt_gui_fps")

                    with dpg.table_row():
                        dpg.add_text("Recent Avg Reward:")
                        dpg.add_text("0.000", tag="txt_avg_reward")

                    with dpg.table_row():
                        dpg.add_text("Best Recent:")
                        dpg.add_text("0.000", tag="txt_best_reward")

                    with dpg.table_row():
                        dpg.add_text("Worst Recent:")
                        dpg.add_text("0.000", tag="txt_worst_reward")

    def _create_framestack_tab(self):
        """Create the Framestack tab with 4 frames displayed in 2x2 grid."""
        with dpg.child_window(width=-1, height=-1, tag="framestack_content"):
            dpg.add_text("Stacked Frames (oldest to newest, left to right, top to bottom):")
            dpg.add_spacer(height=5)

            # Create textures for 4 frames (scaled up for visibility)
            frame_width = 80 * 3  # 80 * 3 = 240
            frame_height = 72 * 3  # 72 * 3 = 216

            # Create 4 texture entries
            for i in range(4):
                create_texture(frame_width, frame_height, f"tex_frame_{i}")

            # Display frames in 2x2 grid
            with dpg.group(horizontal=True):
                with dpg.child_window(width=frame_width + 20, height=frame_height + 40):
                    dpg.add_text("Frame 1 (oldest)")
                    dpg.add_image("tex_frame_0")

                with dpg.child_window(width=frame_width + 20, height=frame_height + 40):
                    dpg.add_text("Frame 2")
                    dpg.add_image("tex_frame_1")

            with dpg.group(horizontal=True):
                with dpg.child_window(width=frame_width + 20, height=frame_height + 40):
                    dpg.add_text("Frame 3")
                    dpg.add_image("tex_frame_2")

                with dpg.child_window(width=frame_width + 20, height=frame_height + 40):
                    dpg.add_text("Frame 4 (newest)")
                    dpg.add_image("tex_frame_3")

            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=5)

            # Current action display
            with dpg.group(horizontal=True):
                dpg.add_text("Current Action: ", color=(200, 200, 200))
                dpg.add_text("UP", tag="txt_current_action", color=(100, 255, 100))

    def _create_advanced_tab(self):
        """Create the Advanced tab with reward graphs, action distribution, and RAM inspector."""
        with dpg.child_window(width=-1, height=-1, tag="advanced_content"):
            # Reward Graph Section
            with dpg.collapsing_header(label="Reward History", default_open=True):
                create_reward_plot("reward_plot")

            dpg.add_spacer(height=10)

            # Action Distribution Section
            with dpg.collapsing_header(label="Action Distribution", default_open=True):
                action_names = ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START"]
                create_action_bar_chart("action_plot", action_names)
                dpg.add_button(label="Reset Counts", callback=self._reset_action_counts)
            
            dpg.add_spacer(height=10)

            # RAM Inspector Section
            with dpg.collapsing_header(label="RAM Inspector", default_open=True):
                self._create_ram_inspector()

    def _create_ram_inspector(self):
        """Create the RAM inspector with collapsible sections for different RAM categories."""
        with dpg.child_window(height=400, border=True):
            # Player Position Section
            with dpg.collapsing_header(label="Player Position", default_open=False):
                with dpg.table(header_row=True, borders_innerH=True, borders_outerH=True,
                              borders_innerV=True, borders_outerV=True):
                    dpg.add_table_column(label="Address")
                    dpg.add_table_column(label="Name")
                    dpg.add_table_column(label="Value (Hex)")
                    dpg.add_table_column(label="Value (Dec)")

                    ram_items = [
                        ("player_x", "Player X"),
                        ("player_y", "Player Y"),
                        ("map_bank", "Map Bank"),
                        ("map_number", "Map Number"),
                        ("world_x", "World X"),
                        ("world_y", "World Y"),
                    ]

                    for key, label in ram_items:
                        with dpg.table_row():
                            ram_entry = config.RAM_MAP.get(key, 0)
                            addr = self._extract_address(ram_entry)
                            dpg.add_text(f"0x{addr:04X}")
                            dpg.add_text(label)
                            dpg.add_text("0x00", tag=f"ram_{key}_hex")
                            dpg.add_text("0", tag=f"ram_{key}_dec")

            # Custom Watch Section
            if config.CUSTOM_WATCH_ADDRESSES:
                with dpg.collapsing_header(label="Custom Watch", default_open=True):
                    with dpg.table(header_row=True, borders_innerH=True, borders_outerH=True,
                                  borders_innerV=True, borders_outerV=True):
                        dpg.add_table_column(label="Name")
                        dpg.add_table_column(label="Address")
                        dpg.add_table_column(label="Value (Hex)")
                        dpg.add_table_column(label="Value (Dec)")

                        for i, watch_entry in enumerate(config.CUSTOM_WATCH_ADDRESSES):
                            with dpg.table_row():
                                name = watch_entry.get("name", f"Watch_{i}")
                                addr = watch_entry.get("addr", 0)
                                dpg.add_text(name)
                                dpg.add_text(f"0x{addr:04X}")
                                dpg.add_text("0x00", tag=f"custom_watch_{i}_hex")
                                dpg.add_text("0", tag=f"custom_watch_{i}_dec")

            # Plot Flags Section
            with dpg.collapsing_header(label="Plot Flags (Story Events)", default_open=False):
                with dpg.table(header_row=True, borders_innerH=True, borders_outerH=True,
                              borders_innerV=True, borders_outerV=True, scrollY=True, height=200):
                    dpg.add_table_column(label="Flag Name")
                    dpg.add_table_column(label="Address")
                    dpg.add_table_column(label="Status")

                    for flag_name in config.PLOT_FLAG_REWARDS.keys():
                        if flag_name in config.RAM_MAP:
                            with dpg.table_row():
                                dpg.add_text(flag_name)
                                ram_entry = config.RAM_MAP[flag_name]
                                addr = self._extract_address(ram_entry)
                                dpg.add_text(f"0x{addr:04X}")
                                dpg.add_text("Not Set", tag=f"flag_{flag_name}", color=(150, 150, 150))

            # Pokemon Party Section
            with dpg.collapsing_header(label="Pokemon Party", default_open=False):
                dpg.add_text("Party Count: 0", tag="ram_party_count")
                with dpg.table(header_row=True, borders_innerH=True, borders_outerH=True,
                              borders_innerV=True, borders_outerV=True):
                    dpg.add_table_column(label="Slot")
                    dpg.add_table_column(label="Species ID")
                    dpg.add_table_column(label="Level")
                    dpg.add_table_column(label="HP")

                    for i in range(6):
                        with dpg.table_row():
                            dpg.add_text(f"#{i + 1}")
                            dpg.add_text("---", tag=f"ram_party_{i}_species")
                            dpg.add_text("---", tag=f"ram_party_{i}_level")
                            dpg.add_text("---", tag=f"ram_party_{i}_hp")

            # Gym Badges Section
            with dpg.collapsing_header(label="Gym Badges", default_open=False):
                with dpg.group():
                    dpg.add_text("Johto Badges (0/8):", tag="ram_johto_badges")
                    with dpg.table(header_row=False):
                        dpg.add_table_column()
                        dpg.add_table_column()
                        dpg.add_table_column()
                        dpg.add_table_column()

                        with dpg.table_row():
                            for i in range(4):
                                dpg.add_text("☐", tag=f"badge_johto_{i}")
                        with dpg.table_row():
                            for i in range(4, 8):
                                dpg.add_text("☐", tag=f"badge_johto_{i}")

                    dpg.add_spacer(height=5)
                    dpg.add_text("Kanto Badges (0/8):", tag="ram_kanto_badges")
                    with dpg.table(header_row=False):
                        dpg.add_table_column()
                        dpg.add_table_column()
                        dpg.add_table_column()
                        dpg.add_table_column()

                        with dpg.table_row():
                            for i in range(4):
                                dpg.add_text("☐", tag=f"badge_kanto_{i}")
                        with dpg.table_row():
                            for i in range(4, 8):
                                dpg.add_text("☐", tag=f"badge_kanto_{i}")

            # Pokedex Section
            with dpg.collapsing_header(label="Pokedex", default_open=False):
                dpg.add_text("Species Seen: 0", tag="ram_pokedex_seen")
                dpg.add_text("Species Caught: 0", tag="ram_pokedex_caught")

            # Event Flags Section (Custom)
            if config.CUSTOM_EVENT_FLAGS:
                with dpg.collapsing_header(label="Event Flags (Custom)", default_open=True):
                    with dpg.table(header_row=True, borders_innerH=True, borders_outerH=True,
                                  borders_innerV=True, borders_outerV=True, scrollY=True, height=200):
                        dpg.add_table_column(label="Flag Name")
                        dpg.add_table_column(label="Status")

                        for i, flag_entry in enumerate(config.CUSTOM_EVENT_FLAGS):
                            with dpg.table_row():
                                flag_name = flag_entry.get("name", f"Flag_{i}")
                                dpg.add_text(flag_name)
                                dpg.add_text("Not Set", tag=f"event_flag_{i}", color=(150, 150, 150))


    # =============================================================================
    # HELPER METHODS
    # =============================================================================

    @staticmethod
    def _extract_address(ram_entry):
        """
        Extract the numeric address from a RAM_MAP entry.

        Handles both formats:
        - New format: {"addr": 0xD20D, "bank": 1}
        - Legacy format: 0xD20D (int)

        Args:
            ram_entry: RAM_MAP entry (dict or int)

        Returns:
            int: The numeric address
        """
        if isinstance(ram_entry, dict):
            return ram_entry.get("addr", 0)
        return ram_entry

    # =============================================================================
    # VALUE CACHING HELPERS
    # =============================================================================

    def _set_value_cached(self, tag: str, value):
        """
        Set a widget value only if it has changed (optimization).

        Args:
            tag: Widget tag
            value: New value to set
        """
        # Convert to string for comparison (dpg values are often strings)
        value_str = str(value)
        if self._cached_values.get(tag) != value_str:
            dpg.set_value(tag, value)
            self._cached_values[tag] = value_str

    def _configure_item_cached(self, tag: str, **kwargs):
        """
        Configure a widget only if the configuration has changed (optimization).

        Args:
            tag: Widget tag
            **kwargs: Configuration parameters to update
        """
        cache_key = f"{tag}_config"
        current_config = self._cached_values.get(cache_key, {})

        # Check if any value has changed
        has_changed = False
        for key, value in kwargs.items():
            if current_config.get(key) != value:
                has_changed = True
                break

        if has_changed:
            dpg.configure_item(tag, **kwargs)
            self._cached_values[cache_key] = kwargs.copy()

    # =============================================================================
    # BUTTON CALLBACKS
    # =============================================================================

    def _on_reset_clicked(self):
        """Handle reset button click."""
        if self.on_reset:
            self.on_reset()

    def _on_pause_clicked(self):
        """Handle pause button click."""
        self.paused = not self.paused
        if self.on_pause_toggle:
            self.on_pause_toggle(self.paused)
        self._update_pause_button()

    def _on_ff_clicked(self):
        """Handle fast forward button click."""
        self.fast_forward = not self.fast_forward
        if self.on_fast_forward_toggle:
            self.on_fast_forward_toggle(self.fast_forward)
        self._update_ff_button()

    def _on_quit_clicked(self):
        """Handle quit button click."""
        if self.on_quit:
            self.on_quit()

    def _on_autoreload_toggled(self, sender, app_data):
        """Handle auto-reload checkbox toggle."""
        self.auto_reload = app_data
        if self.on_autoreload_toggle:
            self.on_autoreload_toggle(app_data)

    def _on_step_once_clicked(self):
        """Handle step once button click."""
        if self.on_step_once:
            self.on_step_once()

    def _on_save_clicked(self):
        """Handle save state button click."""
        if self.on_save_state:
            self.on_save_state()

    def _on_load_clicked(self):
        """Handle load state button click."""
        if self.on_load_state:
            self.on_load_state()

    def _on_rewind_clicked(self):
        """Handle rewind button click."""
        if self.on_rewind:
            self.on_rewind()

    def _reset_action_counts(self):
        """Reset action distribution counts."""
        self.action_counts = [0] * 7
        update_action_bar_chart("action_plot_series", self.action_counts)

    def _update_pause_button(self):
        """Update pause button appearance based on state."""
        if self.paused:
            # Amber/yellow color when paused
            dpg.configure_item("btn_pause", label="Resume (SPACE)")
            dpg.bind_item_theme("btn_pause", self._pause_theme if hasattr(self, '_pause_theme') else 0)
            # Show "Step Once" button when paused
            dpg.configure_item("btn_step_once", show=True)
        else:
            dpg.configure_item("btn_pause", label="Pause (SPACE)")
            dpg.bind_item_theme("btn_pause", 0)  # Reset to default theme
            # Hide "Step Once" button when not paused
            dpg.configure_item("btn_step_once", show=False)

    def _update_ff_button(self):
        """Update fast forward button appearance based on state."""
        if self.fast_forward:
            # Yellow color when fast forwarding
            dpg.configure_item("btn_ff", label="Normal Speed (F)")
            dpg.bind_item_theme("btn_ff", self._ff_theme if hasattr(self, '_ff_theme') else 0)
        else:
            dpg.configure_item("btn_ff", label="Fast Forward (F)")
            dpg.bind_item_theme("btn_ff", 0)  # Reset to default theme

    # =============================================================================
    # PUBLIC UPDATE METHODS
    # =============================================================================

    def update_all_data(self, episode_num: int, step_count: int, episode_reward: float,
                       env, checkpoint_info: dict, recent_rewards: deque, obs, last_action: Optional[int] = None):
        """
        Update all GUI data in one call - called after each env step for efficiency.

        Args:
            episode_num: Current episode number
            step_count: Steps in current episode
            episode_reward: Cumulative reward for current episode
            env: Environment instance
            checkpoint_info: Dictionary with checkpoint information
            recent_rewards: Deque of recent episode rewards
            obs: Current observation from environment
            last_action: Last action taken (0-7)
        """
        import time
        current_time = time.time()

        # Always update overview (lightweight)
        self.update_overview(episode_num, step_count, episode_reward, env,
                           checkpoint_info, recent_rewards,
                           paused=self.paused, fast_forward=self.fast_forward)

        # Update framestack (medium weight)
        self.update_framestack(obs, env, last_action)

        # Update advanced tab (heavy) - only if visible and throttled
        if current_time - self._last_ram_update >= self._ram_update_interval:
            if self._is_advanced_tab_visible():
                self.update_advanced(env, step_count, episode_reward, last_action)
            self._last_ram_update = current_time

    def update_overview(self, episode_num: int, step_count: int, episode_reward: float,
                       env, checkpoint_info: dict, recent_rewards: deque,
                       paused: bool = False, fast_forward: bool = False):
        """
        Update the Overview tab with current statistics.

        Args:
            episode_num: Current episode number
            step_count: Steps in current episode
            episode_reward: Cumulative reward for current episode
            env: Environment instance for reading game state
            checkpoint_info: Dictionary with checkpoint information
            recent_rewards: Deque of recent episode rewards
            paused: Whether the viewer is paused
            fast_forward: Whether fast forward is enabled
        """
        # Update internal state
        self.paused = paused
        self.fast_forward = fast_forward

        # Model Information (use cached setters to avoid unnecessary updates)
        self._set_value_cached("txt_checkpoint", checkpoint_info.get('filename', 'Unknown'))
        self._set_value_cached("txt_training_steps", f"{checkpoint_info.get('steps', 0):,}")

        # Episode Information
        status = "PAUSED" if paused else ("FAST FORWARD" if fast_forward else "RUNNING")
        status_color = (255, 165, 0) if paused else ((255, 255, 0) if fast_forward else (0, 255, 0))
        self._set_value_cached("txt_status", status)
        self._configure_item_cached("txt_status", color=status_color)

        self._set_value_cached("txt_episode", str(episode_num))
        self._set_value_cached("txt_steps", f"{step_count} / {config.MAX_STEPS_PER_EPISODE}")
        self._set_value_cached("txt_reward", f"{episode_reward:.3f}")

        # Calculate step progress with visual bar
        progress_pct = (step_count / config.MAX_STEPS_PER_EPISODE) * 100
        filled_bars = int(progress_pct / 10)  # 10 bars total
        empty_bars = 10 - filled_bars
        progress_bar = "▓" * filled_bars + "░" * empty_bars
        self._set_value_cached("txt_step_progress", f"{progress_bar} {progress_pct:.1f}%")

        # Game State
        try:
            # Access unwrapped environment (through VecFrameStack and VecEnv wrappers)
            unwrapped_env = unwrap_env(env)

            xy = unwrapped_env.xy()
            money = unwrapped_env.money()
            map_bank = unwrapped_env.mem8(unwrapped_env.RAM["map_bank"])
            map_number = unwrapped_env.mem8(unwrapped_env.RAM["map_number"])
            bike_flag = unwrapped_env.mem8(unwrapped_env.RAM["bike_flag"])

            self._set_value_cached("txt_position", f"X={xy[0]}, Y={xy[1]}")
            self._set_value_cached("txt_map", f"Bank={map_bank}, Number={map_number}")
            self._set_value_cached("txt_money", f"${money}")
            self._set_value_cached("txt_bike", str(bool(bike_flag)))
        except Exception as e:
            print(f"[Warning] Failed to read game state: {e}")

        # Statistics
        if len(recent_rewards) > 0:
            avg_reward = np.mean(recent_rewards)
            best_reward = max(recent_rewards)
            worst_reward = min(recent_rewards)
            self._set_value_cached("txt_avg_reward", f"{avg_reward:.3f}")
            self._set_value_cached("txt_best_reward", f"{best_reward:.3f}")
            self._set_value_cached("txt_worst_reward", f"{worst_reward:.3f}")

        # FPS tracking (update periodically)
        import time
        current_time = time.time()

        if current_time - self._last_fps_update >= self._fps_update_interval:
            elapsed = current_time - self._last_fps_update

            # Display ENV FPS (set externally by watch_agent.py based on actual step timestamps)
            self._set_value_cached("txt_env_fps", f"{self._env_fps:.1f}")

            # Calculate GUI FPS (frames per second)
            gui_frames_delta = self._gui_frame_count - self._last_gui_frame_count
            gui_fps = gui_frames_delta / elapsed if elapsed > 0 else 0.0
            self._set_value_cached("txt_gui_fps", f"{gui_fps:.1f}")

            # Update counters
            self._last_gui_frame_count = self._gui_frame_count
            self._last_fps_update = current_time

    def update_framestack(self, obs, env, last_action: Optional[int] = None):
        """
        Update the Framestack tab with current stacked frames and action.

        Args:
            obs: Current observation from the environment
            env: Environment instance (used as fallback)
            last_action: Last action taken (0-7)
        """
        try:
            frames = extract_frames_from_stack(obs, env, n_frames=4)

            # Update each frame texture
            for i, frame in enumerate(frames):
                # Scale up the frame for better visibility
                scaled = scale_image_for_display(frame, scale=3)
                # Convert to DPG texture format
                texture_data = numpy_to_dpg_texture(scaled)
                # Update the texture
                update_texture(f"tex_frame_{i}", texture_data)

            # Update current action display
            if last_action is not None and 0 <= last_action < 7:
                action_names = ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START"]
                action_arrows = ["↑", "↓", "←", "→", "Ⓐ", "Ⓑ", "▶"]
                action_text = f"{action_names[last_action]} {action_arrows[last_action]}"
                self._set_value_cached("txt_current_action", action_text)

        except Exception as e:
            print(f"[Warning] Failed to update framestack: {e}")

    def update_advanced(self, env, step_count: int, episode_reward: float, last_action: Optional[int] = None):
        """
        Update the Advanced tab with reward graphs, action distribution, and RAM data.

        Args:
            env: Environment instance for reading RAM
            step_count: Current step count
            episode_reward: Current episode reward
            last_action: Last action taken (0-7)
        """
        # Update reward history
        self.step_counter += 1
        self.reward_history_x.append(self.step_counter)
        self.reward_history_y.append(episode_reward)
        update_reward_plot("reward_plot", self.reward_history_x, self.reward_history_y)

        # Update action distribution
        if last_action is not None and 0 <= last_action < 7:
            self.action_counts[last_action] += 1
            update_action_bar_chart("action_plot_series", self.action_counts)

        # Update RAM inspector only only if Advanced tab is visible (throttled - most expensive operation)
        import time
        current_time = time.time()
        if current_time - self._last_ram_update >= self._ram_update_interval:
            # Check if Advanced tab is currently visible
            if self._is_advanced_tab_visible():
                self._update_ram_inspector(env)
            self._last_ram_update = current_time

    def _is_advanced_tab_visible(self) -> bool:
        """Check if the Advanced tab is currently visible."""
        try:
            # Get the currently active tab value
            active_tab = dpg.get_value("tab_bar")
            # dpg.get_value returns the tag of the active tab
            return active_tab == "tab_advanced"
        except:
            # If we can't determine, assume it's visible (fail-safe)
            return True

    def _update_ram_inspector(self, env):
        """
        Update RAM inspector with current values.

        Args:
            env: Environment instance for reading RAM
        """
        try:
            # Access unwrapped environment (through VecFrameStack and VecEnv wrappers)
            unwrapped_env = unwrap_env(env)

            # Player Position - use bulk read for contiguous regions
            # Read player position data (2 bytes at 0xD20D-0xD20E for x,y)
            player_pos_data = unwrapped_env.mem8_range(0xD20D, 2, bank=1)
            self._set_value_cached("ram_player_x_hex", f"0x{player_pos_data[0]:02X}")
            self._set_value_cached("ram_player_x_dec", str(player_pos_data[0]))
            self._set_value_cached("ram_player_y_hex", f"0x{player_pos_data[1]:02X}")
            self._set_value_cached("ram_player_y_dec", str(player_pos_data[1]))

            # Read map data (2 bytes at 0xDA00-0xDA01 for bank,number)
            map_data = unwrapped_env.mem8_range(0xDA00, 2, bank=1)
            self._set_value_cached("ram_map_bank_hex", f"0x{map_data[0]:02X}")
            self._set_value_cached("ram_map_bank_dec", str(map_data[0]))
            self._set_value_cached("ram_map_number_hex", f"0x{map_data[1]:02X}")
            self._set_value_cached("ram_map_number_dec", str(map_data[1]))

            # Read world position (2 bytes at 0xDA00, 0xDA02)
            world_x = unwrapped_env.mem8(unwrapped_env.RAM["world_x"])
            world_y = unwrapped_env.mem8(unwrapped_env.RAM["world_y"])
            self._set_value_cached("ram_world_x_hex", f"0x{world_x:02X}")
            self._set_value_cached("ram_world_x_dec", str(world_x))
            self._set_value_cached("ram_world_y_hex", f"0x{world_y:02X}")
            self._set_value_cached("ram_world_y_dec", str(world_y))

            # Custom Watch Addresses
            for i, watch_entry in enumerate(config.CUSTOM_WATCH_ADDRESSES):
                addr = watch_entry.get("addr", 0)
                bank = watch_entry.get("bank", None)
                try:
                    value = unwrapped_env.mem8({"addr": addr, "bank": bank})
                    self._set_value_cached(f"custom_watch_{i}_hex", f"0x{value:02X}")
                    self._set_value_cached(f"custom_watch_{i}_dec", str(value))
                except Exception as e:
                    self._set_value_cached(f"custom_watch_{i}_hex", "ERROR")
                    self._set_value_cached(f"custom_watch_{i}_dec", "ERROR")

            # Plot Flags
            for flag_name in config.PLOT_FLAG_REWARDS.keys():
                if flag_name in config.RAM_MAP:
                    is_set = unwrapped_env.check_plot_flag(flag_name)
                    status = "SET" if is_set else "Not Set"
                    color = (0, 255, 0) if is_set else (150, 150, 150)
                    self._set_value_cached(f"flag_{flag_name}", status)
                    self._configure_item_cached(f"flag_{flag_name}", color=color)

            # Pokemon Party
            party_data = unwrapped_env.read_party_pokemon()
            self._set_value_cached("ram_party_count", f"Party Count: {len(party_data)}")

            for i in range(6):
                if i < len(party_data):
                    pokemon = party_data[i]
                    self._set_value_cached(f"ram_party_{i}_species", str(pokemon['species']))
                    self._set_value_cached(f"ram_party_{i}_level", str(pokemon['level']))
                    self._set_value_cached(f"ram_party_{i}_hp", f"{pokemon['hp_current']}/{pokemon['hp_max']}")
                else:
                    self._set_value_cached(f"ram_party_{i}_species", "---")
                    self._set_value_cached(f"ram_party_{i}_level", "---")
                    self._set_value_cached(f"ram_party_{i}_hp", "---")

            # Gym Badges - use bulk read for both badge bytes (consecutive at 0xD57C-0xD57D)
            badge_data = unwrapped_env.mem8_range(0xD57C, 2, bank=1)
            johto_badges = badge_data[0]
            kanto_badges = badge_data[1]

            johto_count = bin(johto_badges).count('1')
            kanto_count = bin(kanto_badges).count('1')

            self._set_value_cached("ram_johto_badges", f"Johto Badges ({johto_count}/8):")
            self._set_value_cached("ram_kanto_badges", f"Kanto Badges ({kanto_count}/8):")

            for i in range(8):
                johto_has = bool(johto_badges & (1 << i))
                kanto_has = bool(kanto_badges & (1 << i))
                self._set_value_cached(f"badge_johto_{i}", "☑" if johto_has else "☐")
                self._set_value_cached(f"badge_kanto_{i}", "☑" if kanto_has else "☐")

            # Pokedex
            seen = unwrapped_env.get_seen_species()
            caught = unwrapped_env.get_caught_species()
            self._set_value_cached("ram_pokedex_seen", f"Species Seen: {len(seen)}")
            self._set_value_cached("ram_pokedex_caught", f"Species Caught: {len(caught)}")

            # Event Flags (Custom)
            for i, flag_entry in enumerate(config.CUSTOM_EVENT_FLAGS):
                try:
                    # Check if flag is specified by name or by index
                    if "index" in flag_entry:
                        # Read by index
                        flag_index = flag_entry["index"]
                        is_set = unwrapped_env.read_gold_event_flag_by_index(flag_index)
                    else:
                        # Read by name (from event_flags.asm)
                        flag_name = flag_entry["name"]
                        is_set = unwrapped_env.read_gold_event_map(flag_name)

                    status = "SET" if is_set else "Not Set"
                    color = (0, 255, 0) if is_set else (150, 150, 150)
                    self._set_value_cached(f"event_flag_{i}", status)
                    self._configure_item_cached(f"event_flag_{i}", color=color)
                except KeyError as e:
                    # Event flag name not found in event_flags.asm
                    self._set_value_cached(f"event_flag_{i}", "NOT FOUND")
                    self._configure_item_cached(f"event_flag_{i}", color=(255, 0, 0))
                except Exception as e:
                    # Other error
                    self._set_value_cached(f"event_flag_{i}", "ERROR")
                    self._configure_item_cached(f"event_flag_{i}", color=(255, 0, 0))

        except Exception as e:
            print(f"[Warning] Failed to update RAM inspector: {e}")



    def update_status_bar(self, message: str):
        """
        Update the status bar text.

        Args:
            message: Status message to display
        """
        self._set_value_cached("status_bar", f"Status: {message}")

    def set_env_fps(self, fps: float):
        """
        Set the environment FPS value (called from watch_agent.py).

        Args:
            fps: Environment steps per second
        """
        self._env_fps = fps

    def reset_episode_data(self):
        """Reset per-episode tracking data (reward history, action counts)."""
        self.reward_history_x.clear()
        self.reward_history_y.clear()
        self.step_counter = 0
        update_reward_plot("reward_plot", [], [])

    # =============================================================================
    # WINDOW MANAGEMENT
    # =============================================================================

    def setup_viewport(self):
        """Set up the Dear PyGui viewport and prepare for rendering."""
        dpg.create_viewport(
            title="Pokemon Gold RL Agent Viewer",
            width=self.width,
            height=self.height
        )
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)

        # Set up keyboard handlers (replaces global keyboard module)
        with dpg.handler_registry():
            dpg.add_key_press_handler(dpg.mvKey_Q, callback=self._on_quit_clicked)
            dpg.add_key_press_handler(dpg.mvKey_R, callback=self._on_reset_clicked)
            dpg.add_key_press_handler(dpg.mvKey_Spacebar, callback=self._on_pause_clicked)
            dpg.add_key_press_handler(dpg.mvKey_F, callback=self._on_ff_clicked)
            dpg.add_key_press_handler(dpg.mvKey_S, callback=self._on_save_clicked)
            dpg.add_key_press_handler(dpg.mvKey_L, callback=self._on_load_clicked)
            dpg.add_key_press_handler(dpg.mvKey_Back, callback=self._on_rewind_clicked)  # Backspace

    def render(self):
        """Render one frame of the GUI. Call this in your main loop."""
        dpg.render_dearpygui_frame()
        self._gui_frame_count += 1  # Track GUI frames for FPS calculation

    def should_close(self) -> bool:
        """Check if the GUI window should close."""
        return not dpg.is_dearpygui_running()

    def cleanup(self):
        """Clean up Dear PyGui resources."""
        dpg.destroy_context()

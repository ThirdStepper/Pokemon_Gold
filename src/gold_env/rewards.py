# rewards.py
"""
Reward Calculation for Pokemon Gold Environment

This mixin contains the comprehensive reward function that drives agent learning.
Rewards encourage exploration, Pokemon collection, story progression, and badges.
"""

import numpy as np
import config
from collections import deque


class RewardsMixin:
    """
    Mixin providing reward calculation for the Pokemon Gold environment.

    The reward function encourages:
    1. Story milestones (plot flags)
    2. Gym badges (highest priority)
    3. Area exploration (world tiles, map banks, local tiles)
    4. Pokemon collection (catching and training)
    5. Diverse behavior (anti-stuck mechanisms)

    Required attributes from parent class:
    - self._visited_tiles: Set of visited (x, y) coordinates
    - self._visited_map_banks: Set of visited map banks
    - self._visited_map_numbers: Set of visited map numbers
    - self._visited_world_tiles: Set of visited world coordinates
    - self._completed_plot_flags: Set of completed plot flags
    - self._seen_species: Set of seen Pokemon species
    - self._caught_species: Set of caught Pokemon species
    - self._last_party_count: Previous party Pokemon count
    - self._last_total_levels: Previous total Pokemon levels
    - self._last_party_data: Previous party data
    - self._last_badge_count: Previous badge count
    - self._last_money: Previous money amount
    - self._collision_count: Current collision count
    - self._step_count: Current step in episode
    - self._plot_flag_last_values: Dict tracking plot flag values (if DEBUG_PLOT_FLAGS)
    - self._plot_flag_initial_values: Dict of initial plot flag values

    Required methods from parent class:
    - self.xy(): Get current position
    - self.get_world_xy(): Get world position
    - self.mem8(): Read byte from memory
    - self.check_plot_flag(): Check if plot flag is set
    - self.get_seen_species(): Get Pokedex seen species
    - self.get_caught_species(): Get Pokedex caught species
    - self.read_party_pokemon(): Read party Pokemon data
    - self.get_badge_count(): Get badge count
    - self.money(): Get current money
    - self._calculate_novelty_bonus(): Calculate novelty bonus
    - self._detect_stuck(): Detect stuck behavior
    - self._calculate_action_diversity(): Calculate action diversity
    - self._calculate_distance_reward(): Calculate distance reward
    - self._log_plot_flag_state(): Log plot flag changes
    """

    def near_npc_check(self) -> float:
        # True only on the exact step that IsNPCAtCoord.yes fired
        return getattr(self, "_npc_front_seen_step", -999) == getattr(self, "_step_count", -1)
    
    def bumped_this_step(self) -> bool:
        # True only on the exact step that DoPlayerMovement.bump fired
        return getattr(self, "_bump_seen_step", -999) == getattr(self, "_step_count", -1)
    
    def in_dialog(self) -> bool:
        # Prefer your pause flag; optionally double-check textbox flags
        paused = getattr(self, "_pause_penalties", False)
        if config.VERIFY_PAUSE_STATE:
            paused = paused or (self.mem8(self.RAM["textbox_flags"]) != 0)
        return paused


    def _calculate_reward(self) -> float:
        """
        Calculate comprehensive reward based on exploration, Pokemon, and progression.

        This reward function encourages the agent to:
        1. Complete story milestones (plot flags)
        2. Earn gym badges (highest priority)
        3. Explore new areas (world tiles, map banks, local tiles)
        4. Catch and train Pokemon
        5. Store Pokemon in PC

        Reward Hierarchy (highest to lowest):
        - Badges: 50-250 per badge (MASSIVE)
        - Story events: 10-40 per milestone (Team Rocket, Rivals, Pokedex, etc.)
        - Pokemon caught: 8.0 per new Pokemon
        - New map bank: 3.0
        - World tile (overworld): 2.0
        - New map number: 1.0
        - Pokemon level up: 0.5 per level
        - PC storage: 0.5 per Pokemon
        - New tile (local): 0.02
        - Money gained: 0.1 per unit
        - Step penalty: -0.0001
        - Repeated tile: -0.005
        - Collision: -0.01 (escalates with consecutive collisions)

        NEW ANTI-STUCK MECHANISMS:
        - Novelty bonus: +0.0 to +0.5 (tiles visited long ago)
        - Stuck penalty: -0.5 (too few unique tiles in recent history)
        - Action diversity: +0.05 (diverse) or -0.1 to -0.2 (repetitive/wall-running)
        - Distance reward: +0.01 per tile from spawn (periodic check)

        PAUSE BEHAVIOR:
        - During dialogs/battles, all movement and anti-stuck rewards are paused
        - One-shot events (badges, Pokedex, world tiles, etc.) continue to trigger

        Returns:
            The reward value for this step
        """
        reward = 0.0

        # Get current position (used by both paused and unpaused logic)
        cur_xy = self.xy()

        # ========================================================================
        # COMPUTE PAUSE STATE
        # ========================================================================
        # Check if we're in a dialog or battle (pause movement/anti-stuck rewards)
        is_dialog = self.in_dialog()
        

        # ========================================================================
        # MOVEMENT & ANTI-STUCK REWARDS (PAUSED DURING DIALOG/BATTLE)
        # ========================================================================
        # When dialogue boxes or battles are active, skip all movement-based rewards
        # and anti-stuck penalties. This prevents the agent from being punished for
        # "stuck" behavior when the game has taken control.
        # One-shot events (badges, Pokedex, world tiles, etc.) continue below.
        
        # Step penalty (encourages efficiency and is enabled at all times as a constant time pressure)
        reward += config.STEP_PENALTY

        # Collision/door code sampled once here for shaping below
        collision_code = int(self.mem8(self.RAM["walking_tile_collision"]))

        if not is_dialog:
            # Movement rewards (new vs repeated tiles)
            if cur_xy not in self._visited_tiles:
                # First time visiting this tile - positive reward
                reward += config.NEW_TILE_REWARD
                self._visited_tiles.add(cur_xy)
            else:
                # Revisiting tile - small penalty to discourage loops
                reward += config.REPEATED_TILE_PENALTY

            # Novelty bonus (visit recency)
            novelty_bonus = self._calculate_novelty_bonus(cur_xy)
            reward += novelty_bonus

            # Update tile visit tracking
            self._tile_visit_counts[cur_xy] = self._tile_visit_counts.get(cur_xy, 0) + 1
            self._tile_last_visit[cur_xy] = self._step_count

            # Collision penalties (escalate for consecutive collisions)
            if self._collision_count > 0:
                # Cap the exponential growth to avoid numeric overflow/NaNs
                exp_n = min(self._collision_count - 1, config.MAX_CONSECUTIVE_COLLISION_EXP)
                collision_penalty = config.COLLISION_PENALTY * (
                    config.CONSECUTIVE_COLLISION_MULTIPLIER ** exp_n
                )
                # Clamp the penalty floor
                if collision_penalty < config.COLLISION_PENALTY_MIN:
                    collision_penalty = config.COLLISION_PENALTY_MIN
                reward += collision_penalty
            

            # Enhanced anti-stuck mechanisms
            # Detect if agent is stuck in small area
            stuck_penalty = self._detect_stuck()
            reward += stuck_penalty

            # Reward action diversity, penalize repetitive patterns
            diversity_reward = self._calculate_action_diversity()
            # Only count diversity when the last action was a movement (0..3)
            try:
                is_move = 0 <= int(self._last_action) <= 3
            except Exception:
                is_move = False
            if is_move:
                reward += diversity_reward

            # Reward distance from spawn (encourages outward exploration)
            distance_reward = self._calculate_distance_reward()
            reward += distance_reward
        

        # -----------------------------------------------------------------
        # Event-style shaping:
        # - Penalize physical bumps
        # - Reward purposeful entering (door/warp) and discourage exit/edge spam
        # -----------------------------------------------------------------
        # We treat any of the following as a "door event" for spam detection:
        #   - collision codes 112/113/122
        #   - a pulse from EnterMapWarp (any door/warp)
        #   - a pulse from EnterMapConnection (edge connections)
        step = self._step_count

        # One-step pulses from function hooks
        warp_pulse = (
            getattr(self, "_enter_map_warp_seen_step", -1) == step
            or getattr(self, "_enter_map_connection_seen_step", -1) == step
        )

        # Defaults for debug/introspection
        penalty_mag = 0.0
        n_events = 0

        if (collision_code in (112, 113, 122)) or warp_pulse:
            # Track recent door events and penalize only if they look like spam.
            if not hasattr(self, "_door_event_steps"):
                self._door_event_steps = deque()
            self._door_event_steps.append(step)

            # Drop events outside the spam window
            cutoff = step - config.DOOR_SPAM_COOLDOWN_STEPS
            while self._door_event_steps and self._door_event_steps[0] <= cutoff:
                self._door_event_steps.popleft()

            # Penalize only if there are enough toggles in-window (scaled)
            n_events = len(self._door_event_steps)
            if n_events >= config.DOOR_SPAM_MIN_EVENTS:
                # 3 events → 1×, 4 → 2×, etc.
                excess = n_events - (config.DOOR_SPAM_MIN_EVENTS - 1)
                penalty_mag = excess * config.DOOR_SPAM_PENALTY_PER_EVENT
                if penalty_mag > config.DOOR_SPAM_MAX_PENALTY:
                    penalty_mag = config.DOOR_SPAM_MAX_PENALTY
                reward -= penalty_mag

            # Positive shaping ONLY when we can confirm "enter" via collision_code.
            # (EnterMapWarp/Connection fire for both directions and lack polarity.)
            if collision_code in (113, 122):
                reward += config.DOOR_ENTER_REWARD
            elif collision_code == 112:
                reward += getattr(config, "DOOR_EXIT_PENALTY", 0.0)

        # -----------------------------
        # Debug & quick introspection
        # -----------------------------
        # Stash last-step values so you can surface them in info/HUD if desired.
        self._last_door_penalty = penalty_mag
        self._last_door_event_count = n_events
        self._last_warp_pulse = bool(warp_pulse)
        self._last_door_collision_code = int(collision_code)

        if getattr(config, "DOOR_DEBUG_LOG", False):
            # Only print when relevant to keep noise down
            if (collision_code in (112, 113, 122)) or warp_pulse or penalty_mag > 0.0:
                print(
                    f"[DOOR] step={step} "
                    f"code={collision_code} "
                    f"pulse={int(warp_pulse)} "
                    f"events_in_window={n_events} "
                    f"penalty={penalty_mag:+.4f} "
                    f"reward_enter={(config.DOOR_ENTER_REWARD if collision_code in (113,122) else 0.0):+.4f}"
                )

        # ========================================================================
        # A BUTTON BEHAVIORAL SHAPING
        # ========================================================================
        # Reward pressing A button near NPCs and during dialogs to help agent
        # discover interaction mechanics (critical for story progression)
        # Edge detection for A press: only pay on rising edge (A released -> pressed)
        pressed_A = (self._last_action == 4)
        prev_A = getattr(self, "_prev_action_was_A", False)
        rising_A = pressed_A and (not prev_A)

        # Track textbox flag to detect actual dialog state changes
        tb = self.mem8(self.RAM["textbox_flags"])
        prev_tb = getattr(self, "_prev_textbox_flags", tb)
        textbox_edge = (tb != prev_tb)  # open/close/advance edge

        if rising_A:
            # Reward only if an NPC was detected exactly this step AND we're not already in dialog
            # (prevents “face NPC and hold A” farming while idle)
            if self.near_npc_check() and not self.bumped_this_step() and not self.in_dialog():
                reward += config.A_BUTTON_NEAR_NPC_REWARD
            elif self.bumped_this_step():
                reward += getattr(config, "A_BUTTON_NEAR_NPC_WHILE_BUMPING_REWARD", 0.0)

            # Reward A during dialog only when the textbox actually edges (or with a small cooldown)
            dialog_ok = textbox_edge or (
                self._step_count - getattr(self, "_last_dialog_a_reward_step", -999)
                >= getattr(config, "A_DIALOG_COOLDOWN_STEPS", 8)
            )
            if self.in_dialog() and dialog_ok:
                reward += config.A_BUTTON_IN_DIALOG_REWARD
                self._last_dialog_a_reward_step = self._step_count

        # Bookkeeping for next step
        self._prev_action_was_A = pressed_A
        self._prev_textbox_flags = tb

        # ========================================================================
        # EXPLORATION REWARDS
        # ========================================================================
        map_bank = self.mem8(self.RAM["map_bank"])
        map_number = self.mem8(self.RAM["map_number"])

        # Reward entering buildings (map transitions often lead to story events)
        # Track last map number to detect transitions
        if not hasattr(self, '_last_map_number'):
            self._last_map_number = map_number

        if map_number != self._last_map_number:
            # Map changed - check if it was through a warp/door
            if self.mem8(self.RAM["warp_number"]) > 0:
                reward += config.ENTER_BUILDING_REWARD

        self._last_map_number = map_number

        # New map bank discovered (major exploration milestone)
        if map_bank not in self._visited_map_banks:
            reward += config.NEW_MAP_BANK_REWARD
            self._visited_map_banks.add(map_bank)

        # New map number discovered (moderate exploration milestone)
        if map_number not in self._visited_map_numbers:
            reward += config.NEW_MAP_NUMBER_REWARD
            self._visited_map_numbers.add(map_number)

        # ========================================================================
        # WORLD-LEVEL EXPLORATION
        # ========================================================================
        world_xy = self.get_world_xy()
        if world_xy not in self._visited_world_tiles:
            reward += config.NEW_WORLD_TILE_REWARD
            self._visited_world_tiles.add(world_xy)

        # ========================================================================
        # STORY/PLOT PROGRESSION
        # ========================================================================
        # Check each configured plot flag and award one-time rewards when they flip from 0→1
        for flag_name, flag_reward in config.PLOT_FLAG_REWARDS.items():
            # Skip if already completed this episode
            if flag_name in self._completed_plot_flags:
                continue

            # Check if flag is now set
            try:
                if self.read_gold_event_map(flag_name):
                    # Flag is set! Award reward and mark as completed
                    reward += flag_reward
                    self._completed_plot_flags.add(flag_name)
            except KeyError:
                # Flag name not found in event_flags.asm - skip silently
                # (This allows for forward-compatibility with different ROM versions)
                pass

        # ========================================================================
        # POKEDEX PROGRESS (UNIQUE SPECIES)
        # ========================================================================
        # Reward for seeing new species
        current_seen = self.get_seen_species()
        new_seen = current_seen - self._seen_species
        if new_seen:
            reward += config.NEW_SPECIES_SEEN_REWARD * len(new_seen)
            self._seen_species = current_seen

        # Reward for catching new species (higher priority than just seeing)
        current_caught = self.get_caught_species()
        new_caught = current_caught - self._caught_species
        if new_caught:
            reward += config.NEW_SPECIES_CAUGHT_REWARD * len(new_caught)
            self._caught_species = current_caught

        # ========================================================================
        # POKEMON COLLECTION & TRAINING
        # ========================================================================
        party_data = self.read_party_pokemon()
        party_count = len(party_data)

        # Reward for catching new Pokemon (increasing party size)
        if self._last_party_count is not None and party_count > self._last_party_count:
            reward += config.NEW_POKEMON_CAUGHT_REWARD * (party_count - self._last_party_count)
        self._last_party_count = party_count

        # Reward for leveling up Pokemon
        total_levels = sum(p["level"] for p in party_data)
        if self._last_total_levels is not None and total_levels > self._last_total_levels:
            reward += config.POKEMON_LEVEL_UP_REWARD * (total_levels - self._last_total_levels)
        self._last_total_levels = total_levels

        # Reward for improving Pokemon stats (HP/Attack/Defense)
        total_stats = sum(p["hp_max"] + p["attack"] + p["defense"] for p in party_data)
        if self._last_party_data is not None:
            last_stats = sum(p["hp_max"] + p["attack"] + p["defense"] for p in self._last_party_data)
            if total_stats > last_stats:
                reward += config.POKEMON_STAT_REWARD_MULTIPLIER * (total_stats - last_stats)
        self._last_party_data = party_data

        # ========================================================================
        # PC STORAGE (DISABLED - Using Pokedex caught/seen instead)
        # ========================================================================
        # PC box tracking can be unreliable, so we rely on Pokedex progress instead
        # pc_total = self.get_pc_pokemon_count()
        # if self._last_pc_total is not None and pc_total > self._last_pc_total:
        #     reward += config.PC_POKEMON_REWARD * (pc_total - self._last_pc_total)
        # self._last_pc_total = pc_total

        # ========================================================================
        # GYM BADGES (MASSIVE REWARDS)
        # ========================================================================
        badge_count = self.get_badge_count()
        if self._last_badge_count is not None and badge_count > self._last_badge_count:
            # Award escalating rewards for each new badge earned
            for i in range(self._last_badge_count, badge_count):
                if i < len(config.BADGE_REWARDS):
                    reward += config.BADGE_REWARDS[i]
        self._last_badge_count = badge_count

        # ========================================================================
        # MONEY (LEGACY)
        # ========================================================================
        cur_money = self.money()
        if self._last_money is not None and cur_money > self._last_money:
            reward += config.MONEY_REWARD_MULTIPLIER * (cur_money - self._last_money)
        self._last_money = cur_money

        # Final safety clip to keep rewards numerically stable
        reward = float(np.clip(reward, -config.REWARD_CLIP_ABS, config.REWARD_CLIP_ABS))
        return reward

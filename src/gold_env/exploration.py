# exploration.py
"""
Enhanced Exploration Helper Methods for Pokemon Gold Environment

This mixin provides anti-stuck mechanisms to encourage exploration and
prevent the agent from getting stuck in local optima, wall-running, or
pacing in small areas.
"""

from typing import Tuple
import math
from collections import Counter
import config


class ExplorationMixin:
    """
    Mixin providing enhanced exploration and anti-stuck detection.

    These methods implement several mechanisms:
    - Novelty bonuses: Reward tiles not visited recently
    - Stuck detection: Detect when agent loops in small area
    - Action diversity: Reward diverse actions, penalize repetition
    - Distance rewards: Encourage venturing from spawn

    Required attributes from parent class:
    - self._tile_visit_counts: Dict tracking visit counts per tile
    - self._tile_last_visit: Dict tracking last visit step per tile
    - self._step_count: Total steps in current episode
    - self._position_history: Deque of recent positions
    - self._action_history: Deque of recent actions
    - self._spawn_position: Starting position of episode
    - self._last_distance_check_step: Last step when distance was checked
    - self.xy(): Method to get current position
    """

    def _calculate_novelty_bonus(self, tile: Tuple[int, int]) -> float:
        """
        Calculate novelty bonus for visiting a tile based on visit recency.

        Tiles visited long ago get higher bonuses than recently visited tiles.
        Formula: base_bonus * (1 - exp(-steps_since_visit / decay_rate))

        Args:
            tile: (x, y) coordinate tuple

        Returns:
            Novelty bonus value (0.0 to NOVELTY_BONUS_SCALE)
        """
        if not config.NOVELTY_BONUS_ENABLED:
            return 0.0

        # First visit gets full bonus
        if tile not in self._tile_last_visit:
            return config.NOVELTY_BONUS_SCALE

        # Calculate steps since last visit
        steps_since_visit = self._step_count - self._tile_last_visit[tile]

        # Exponential decay: bonus increases as time since visit increases
        # More steps = more novelty = higher bonus
        decay_rate = config.NOVELTY_DECAY_STEPS
        novelty_factor = 1.0 - math.exp(-steps_since_visit / decay_rate)

        return config.NOVELTY_BONUS_SCALE * novelty_factor

    def _detect_stuck(self) -> float:
        """
        Detect if agent is stuck in a small area by analyzing position history.

        Checks recent position history to count unique tiles visited.
        If the agent has only visited a few unique tiles recently, it's stuck.

        Returns:
            Penalty value (0.0 or negative) if stuck detected
        """
        if not config.STUCK_DETECTION_ENABLED:
            return 0.0

        # Need enough history to make determination
        if len(self._position_history) < config.STUCK_DETECTION_WINDOW:
            return 0.0

        # Count unique tiles in recent history
        unique_tiles = len(set(self._position_history))

        # If exploring very few tiles, agent is stuck
        if unique_tiles < config.STUCK_RADIUS_THRESHOLD:
            return config.STUCK_PENALTY

        return 0.0

    def _calculate_action_diversity(self) -> float:
        """
        Calculate reward/penalty based on action diversity.

        Analyzes recent action history:
        - High entropy (diverse actions) → bonus
        - Low entropy (repetitive actions) → penalty
        - Wall-running pattern (4+ identical moves) → extra penalty

        Returns:
            Reward or penalty based on action diversity
        """
        if not config.ACTION_DIVERSITY_ENABLED:
            return 0.0

        # Need enough history
        if len(self._action_history) < config.ACTION_DIVERSITY_WINDOW // 2:
            return 0.0

        actions = list(self._action_history)

        # Check for wall-running: 4+ consecutive identical movement actions (1-4)
        if len(actions) >= config.WALL_RUN_THRESHOLD:
            last_n = actions[-config.WALL_RUN_THRESHOLD:]
            # Check if all are identical and are movement actions (1-4)
            if len(set(last_n)) == 1 and 1 <= last_n[0] <= 4:
                return config.WALL_RUN_PENALTY

        # Calculate Shannon entropy over action distribution
        action_counts = Counter(actions)
        total = len(actions)
        entropy = 0.0

        for count in action_counts.values():
            if count > 0:
                prob = count / total
                entropy -= prob * math.log2(prob)

        # Max entropy for 8 actions is log2(8) = 3.0
        max_entropy = math.log2(8)
        normalized_entropy = entropy / max_entropy  # 0.0 to 1.0

        # High entropy (>0.7) → bonus
        # Low entropy (<0.3) → penalty
        if normalized_entropy > 0.7:
            return config.ACTION_DIVERSITY_BONUS
        elif normalized_entropy < 0.3:
            return config.ACTION_REPETITION_PENALTY

        return 0.0

    def _calculate_distance_reward(self) -> float:
        """
        Calculate reward based on distance from spawn position.

        Encourages agent to venture away from starting area.
        Only calculated periodically for performance.

        Returns:
            Reward based on Manhattan distance from spawn
        """
        if not config.DISTANCE_REWARD_ENABLED:
            return 0.0

        # Only check every N steps for performance
        if self._step_count - self._last_distance_check_step < config.DISTANCE_REWARD_INTERVAL:
            return 0.0

        # Need spawn position set
        if self._spawn_position is None:
            return 0.0

        self._last_distance_check_step = self._step_count

        # Calculate Manhattan distance
        cur_pos = self.xy()
        distance = abs(cur_pos[0] - self._spawn_position[0]) + abs(cur_pos[1] - self._spawn_position[1])

        return distance * config.DISTANCE_REWARD_SCALE

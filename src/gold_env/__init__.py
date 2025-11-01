# __init__.py
"""
Pokemon Gold Environment Mixins Package

This package contains modular mixins that compose the PokemonGoldEnv class.
Each mixin handles a specific aspect of the environment functionality.

Organization:
- ram_helpers: Low-level RAM reading utilities
- event_flags: Event flag parsing and reading
- savestate_utilities: Savestate loading/saving
- exploration: Anti-stuck exploration mechanisms
- rewards: Comprehensive reward calculation

Usage:
    from gold_env import (
        RAMHelpersMixin,
        EventFlagsMixin,
        SavestateUtilsMixin,
        ExplorationMixin,
        RewardsMixin
    )
"""

from .ram_helpers import RAMHelpersMixin
from .event_flags import EventFlagsMixin
from .savestate_utilities import SavestateUtilsMixin
from .exploration import ExplorationMixin
from .rewards import RewardsMixin

__all__ = [
    'RAMHelpersMixin',
    'EventFlagsMixin',
    'SavestateUtilsMixin',
    'ExplorationMixin',
    'RewardsMixin',
]

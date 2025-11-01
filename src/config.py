# config.py
"""
Configuration file for Pokemon Gold RL Training

This file centralizes all paths, hyperparameters, and constants used across
the project. As a beginner, you can tweak these values here without digging
through the code.

ORGANIZATION:
1. Project Setup - File paths and ROM configuration
2. Environment Configuration - How the game environment runs
3. Reward System - What drives the agent's learning
4. Training Configuration - PPO algorithm and training settings
5. Visualization & Monitoring - Viewing the agent in action
6. Technical Settings - CPU/GPU and debug configuration
7. Game Data - Pokemon Gold RAM addresses and action space
"""

from pathlib import Path

# Path to the Pokemon Gold ROM file and .sym file
ROM_PATH = Path(r"C:\_Projects\_Python\ML\Pokemon_Gold\rom\Pokemon_Gold.gbc")

SYM_PATH = Path(r"C:\_Projects\_Python\ML\Pokemon_Gold\rom\pokegold.sym")

EVENTFLAGS_PATH = Path(r"C:\_Projects\_Python\ML\Pokemon_Gold\rom\event_flags.asm")
INIT_STATE_PATH = Path(r"C:\_Projects\_Python\ML\Pokemon_Gold\states\boot_after_intro.state")
MODELS_DIR = Path(r"C:\_Projects\_Python\ML\Pokemon_Gold\_models")
LOGS_DIR = Path(r"C:\_Projects\_Python\ML\Pokemon_Gold\_logs")
REQUIRE_INIT_STATE = True

# Game Boy screen is 160x144 pixels. We downsample to reduce computational cost
# and convert to grayscale (black & white) since color isn't needed for gameplay
ORIGINAL_SCREEN_WIDTH = 160
ORIGINAL_SCREEN_HEIGHT = 144
OBSERVATION_WIDTH = 80
OBSERVATION_HEIGHT = 72

MAX_STEPS_PER_EPISODE = 10000
BOOT_WARMUP_FRAMES = 60

# How many emulator frames to skip between observations
# Higher = faster training but agent sees fewer frames
# Set to 12 to replace sticky D-pad (agent decides every 12 frames)
FRAME_SKIP = 12

VIEWER_SMOOTH_RENDER = True
RENDER_EVERY_N_FRAMES = 1

# Game Boy D-pad requires holding for ~10-15 frames to complete a step
# DISABLED: Using FRAME_SKIP instead to avoid PPO gradient bias
# Sticky D-pad creates 12 training samples per decision, breaking PPO assumptions
# Frame skip provides same movement commitment but cleaner action-observation mapping
STICKY_DPAD_ENABLED = False
MOVE_HOLD_FRAMES = 12
INPUT_HOLD_FRAMES = 12
POST_RELEASE_FRAMES = 6
MENU_OPEN_EXTRA_FRAMES = 60

# Random seed for reproducibility
# Each parallel environment will use RANDOM_SEED + env_index
# This ensures deterministic training across runs while giving each env unique randomness
RANDOM_SEED = 777

NUM_ENVS = 7
N_STEPS = 2048
BATCH_SIZE = 2048
N_EPOCHS = 4

# Total number of steps to train for
# More steps = longer training but better performance
# 10,000 is very short (just for testing), real training needs 1M+ steps
TOTAL_TRAINING_STEPS = 7_000_000

# Time-Boxed Training (Optional)
# For overnight training, use time-based limits instead of step counts
# If OVERNIGHT_TRAIN_SECONDS > 0, it takes precedence over TOTAL_TRAINING_STEPS
# Training happens in chunks to preserve progress and allow graceful interruption
#
# Set to 0 to disable time-boxing and use TOTAL_TRAINING_STEPS instead
# Examples:
#   8 hours  = 8 * 60 * 60 = 28800
#   10 hours = 10 * 60 * 60 = 36000
#   12 hours = 12 * 60 * 60 = 43200
OVERNIGHT_TRAIN_SECONDS = 8 * 60 * 60

CHECKPOINT_SAVE_FREQUENCY = NUM_ENVS * N_STEPS
TRAIN_CHUNK_STEPS = CHECKPOINT_SAVE_FREQUENCY * 10

# Learning rate for the neural network
# Lower = more stable but slower learning
# Higher = faster but potentially unstable
LEARNING_RATE = 0.0003

USE_SUBPROC = True

# Stack N consecutive frames as observation (provides temporal information)
# 0 = disabled (single frame), 4 = stack last 4 frames
# Frame stacking helps agent understand movement without recurrent networks
FRAME_STACK = 4

TRAINING_VERBOSE_LEVEL = 0

# Reinforcement Learning agents learn by receiving rewards.
# These parameters control what behaviors we want to encourage/discourage.
STEP_PENALTY = -0.0001
NEW_TILE_REWARD = 0.001
REPEATED_TILE_PENALTY = -0.0001
COLLISION_PENALTY = -0.01
CONSECUTIVE_COLLISION_MULTIPLIER = 1.03

# Safety caps to prevent numeric blowups
# Limit how fast the exponential collision penalty can grow,
# and clamp the minimum per-step penalty.
MAX_CONSECUTIVE_COLLISION_EXP = 3
COLLISION_PENALTY_MIN = -0.01

REWARD_CLIP_ABS = 20.0

NEW_MAP_BANK_REWARD = 1.5
NEW_MAP_NUMBER_REWARD = 1.0
NEW_WORLD_TILE_REWARD = 0.1

# A Button Behavioral Shaping
# These rewards help the agent learn to interact with NPCs and environment
A_BUTTON_NEAR_NPC_REWARD = 0.01
A_BUTTON_NEAR_NPC_WHILE_BUMPING_REWARD = 0.008
A_BUTTON_IN_DIALOG_REWARD = 0.005
A_DIALOG_COOLDOWN_STEPS = 25

ENTER_BUILDING_REWARD = 0
BUMP_EVENT_PENALTY = -0.05
DOOR_ENTER_REWARD = 0.05
DOOR_EXIT_PENALTY = 0
DOOR_SPAM_COOLDOWN_STEPS = 100
DOOR_SPAM_PENALTY_PER_EVENT = 0.10
DOOR_SPAM_MAX_PENALTY = 0.50
DOOR_SPAM_MIN_EVENTS = 3
DOOR_DEBUG_LOG = False

NEW_POKEMON_CAUGHT_REWARD = 0.2
NEW_SPECIES_SEEN_REWARD = 0.1
NEW_SPECIES_CAUGHT_REWARD = 1.0
POKEMON_LEVEL_UP_REWARD = 0.05
POKEMON_STAT_REWARD_MULTIPLIER = 0.0005
PC_POKEMON_REWARD = 0.0

# Gym Badge Rewards (Escalating)
# These are MASSIVE rewards to prioritize game progression
BADGE_REWARDS = [
    5.0,
    6.5,
    8.0,
    9.5,
    11.0,
    13.0,
    15.0,
    18.0,
]

# Story/Plot Progression Rewards
# Rewards for completing major story milestones
# These are one-shot rewards that trigger when event flags flip from 0->1
# Note: Keep gym-leader "beat" flags small or omit them to avoid double-counting with BADGE_REWARDS
PLOT_FLAG_REWARDS = {
    # Early game progression
    "EVENT_PLAYERS_HOUSE_MOM_2": 1.0,
    "EVENT_MR_POKEMONS_HOUSE_OAK": 0.4,
    "EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON": 1.0,
    "EVENT_TALKED_TO_MOM_AFTER_MYSTERY_EGG_QUEST": 0.4,     # Post-quest confirmation
    "EVENT_GOT_A_POKEMON_FROM_ELM": 1.1,                    # Starter acquired (meaningful early reward)
    "EVENT_LEARNED_TO_CATCH_POKEMON": 1.2,                  # Tutorial completion

    # Egg & Togepi progression
    "EVENT_TOGEPI_HATCHED": 0.8,                            # Modest reward for hatching
    "EVENT_SHOWED_TOGEPI_TO_ELM": 1.2,                      # Small bonus for showing

    # Mobility unlocks (important for exploration)
    "EVENT_GOT_BICYCLE": 2.7,                               # Bicycle is a medium mobility unlock

    # Item unlocks
    "EVENT_GOT_OLD_ROD": 1.0,
    "EVENT_GOT_GOOD_ROD": 1.0,
    "EVENT_GOT_SUPER_ROD": 1.0,

    # HM unlocks
    "EVENT_GOT_HM01_CUT": 3.0,
    "EVENT_GOT_HM02_FLY": 3.0,
    "EVENT_GOT_HM03_SURF": 3.4,                             
    "EVENT_GOT_HM04_STRENGTH": 3.0,
    "EVENT_GOT_HM05_FLASH": 2.5,
    "EVENT_GOT_HM06_WHIRLPOOL": 3.0,

    # Story beats
    "EVENT_CLEARED_SLOWPOKE_WELL": 2.4,                     # Medium story beat
    "EVENT_MADE_UNOWN_APPEAR_IN_RUINS": 1.4,                # Small-medium puzzle unlock
    "EVENT_USED_THE_CARD_KEY_IN_THE_RADIO_TOWER": 2.0,      # Medium step
    "EVENT_CLEARED_RADIO_TOWER": 3.3,                       # Large story beat
    "EVENT_HERDED_FARFETCHD": 1.0,
    "EVENT_FOUGHT_SUDOWOODO": 1.4,
    "EVENT_HEALED_MOOMOO": 1.0,

    # Rival Events
    "EVENT_RIVAL_NEW_BARK_TOWN": 0.4,
    "EVENT_RIVAL_CHERRYGROVE_CITY": 1.5,
    "EVENT_RIVAL_AZALEA_TOWN": 0.7,
    "EVENT_RIVAL_SPROUT_TOWER": 1.0,
    "EVENT_RIVAL_BURNED_TOWER": 1.0,


    

    # Gym leader beats (small tokens - badges pay the big reward)
    "EVENT_BEAT_FALKNER": 1.1,  # Token reward (badge pays 15.0)
    "EVENT_BEAT_BUGSY": 1.2,   # Token reward (badge pays 20.0)
    "EVENT_BEAT_WHITNEY": 1.3,  # Token reward (badge pays 25.0)
    "EVENT_BEAT_MORTY": 1.4,    # Token reward (badge pays 30.0)
    "EVENT_BEAT_CHUCK": 1.5,    # Token reward (badge pays 35.0)
    "EVENT_BEAT_JASMINE": 1.6,  # Token reward (badge pays 40.0)
    "EVENT_BEAT_PRYCE": 1.7,    # Token reward (badge pays 45.0)
    "EVENT_BEAT_CLAIR": 1.8,    # Token reward (badge pays 50.0)
}

# Legacy Rewards
MONEY_REWARD_MULTIPLIER = 0.001           # Reward for gaining money (kept for compatibility)

# --- 3B. ANTI-STUCK SYSTEMS ---
# Advanced mechanisms to prevent the agent from getting stuck in local optima,
# running into walls, or pacing back and forth in small areas.

# Novelty Bonus System
# Rewards visiting tiles that haven't been visited recently
NOVELTY_BONUS_ENABLED = False
NOVELTY_BONUS_SCALE = 0.008
NOVELTY_DECAY_STEPS = 250

# Stuck Detection System
# Detects when agent is stuck in small areas and applies penalties
STUCK_DETECTION_ENABLED = False
STUCK_DETECTION_WINDOW = 200
STUCK_RADIUS_THRESHOLD = 9
STUCK_PENALTY = -0.005

# Movement Diversity System
# Rewards diverse action patterns, penalizes repetitive movements
ACTION_DIVERSITY_ENABLED = False
ACTION_DIVERSITY_WINDOW = 24
ACTION_DIVERSITY_BONUS = 0.02
ACTION_REPETITION_PENALTY = -0.025
WALL_RUN_PENALTY = -0.015
WALL_RUN_THRESHOLD = 8

# Distance-from-Spawn Rewards
# Encourages agent to venture away from starting position
DISTANCE_REWARD_ENABLED = True
DISTANCE_REWARD_SCALE = 0.005
DISTANCE_REWARD_INTERVAL = 25

# Penalty Pausing System
# Pauses collision/stuck penalties during dialogues and battles
# The agent shouldn't be punished for "stuck" behavior when the game has taken control
VERIFY_PAUSE_STATE = True

# Render mode for the emulator
# "headless" = No window, fastest (use for training)
# "window" = Show game window (use for visualization/debugging)
RENDER_MODE = "headless"

# Target frames per second when in window mode
# Game Boy runs at ~59.7 FPS, we use 60 for simplicity
# This only applies when RENDER_MODE = "window"
# In headless mode, the emulator runs as fast as possible
TARGET_FPS = 60

# Device to run viewer inference on
# "cpu" = Keep viewer off GPU (recommended - leaves GPU free for training)
# "cuda" = Use GPU for viewer (only if not training simultaneously)
VIEWER_DEVICE = "cuda"

# How often to refresh the GUI data updates (in seconds)
# Lower = more responsive display, but more CPU overhead for data fetching
# Higher = less overhead, but less responsive stats
# Note: This controls data fetching rate (RAM reads, env queries), not GUI render FPS (which stays at 60)
# Optimized to 20-30 Hz (0.033-0.050s) to reduce expensive RAM reads while keeping GUI smooth
VIEWER_DISPLAY_REFRESH_INTERVAL = 0.040

VIEWER_FRAMESTACK_REFRESH_INTERVAL = 0.033

# Whether to automatically reload newer model checkpoints during viewing
# If True: viewer will check for new models every VIEWER_RELOAD_CHECK_INTERVAL seconds
# If False: viewer loads the latest model once at startup
AUTO_RELOAD_MODELS = True

VIEWER_RELOAD_CHECK_INTERVAL = 10

VIEWER_EPISODE_HISTORY = 10

VIEWER_STATES_DIR = Path(r"C:\_Projects\_Python\ML\Pokemon_Gold\_viewer_states")
VIEWER_REWIND_SLOTS = 10
VIEWER_REWIND_INTERVAL_SEC = 1.0

# Maximum number of CPU threads PyTorch can use
# Set to None to use all available cores (default PyTorch behavior)
# Set to a number to limit CPU usage (e.g., 12 on a 16-core system = 75%)
#
# Why limit CPU threads?
# - Prevents training from monopolizing 100% of CPU
# - Allows running viewer + training simultaneously
# - Useful in shared/multi-user environments
# - Still applies when using CUDA/GPU (CPU threads used for data loading)
#
# Recommended values for a 16-core system:
#   12 = 75% (leaves 4 cores free) - recommended for running viewer + training
#   8  = 50% (leaves 8 cores free) - good for heavy multitasking
#   None = 100% (uses all cores) - when training is the only priority
TORCH_NUM_THREADS = 8

DEBUG_PLOT_FLAGS = False
PLOT_FLAGS_LOG_PATH = Path(r"C:\_Projects\_Python\ML\Pokemon_Gold\_logs\plot\plot_flags_debug.log")

# These sections contain Pokemon Gold-specific data structures and memory
# addresses. modifying these isnt generally necessary unless  extending
# the reward system or adding new game state tracking.
#
# these memory addresses tell us where important game data is stored in RAM
# source: https://github.com/pret/pokegold/blob/symbols/pokegold.sym
#
# Structured symbols: { "addr": ..., "bank": Optional[int] }
# bank=None -> unbanked read (pyboy.memory[addr])
# bank=int  -> banked read   (pyboy.memory[bank, addr])
RAM_MAP = {
    # === RAM Bank Stuff
    "bank_register": 0xFF70,

    # === Player Position ===
    "player_x": { "bank": 1, "addr": 0xD20D },          # Player's X coordinate on the current map
    "player_y": { "bank": 1, "addr": 0xD20E },          # Player's Y coordinate on the current map
    "map_bank": { "bank": 1, "addr": 0xDA00 },          # Which group of maps we're in
    "map_number": { "bank": 1, "addr": 0xDA01 },        # Which specific map within the bank
    "world_x": { "bank": 1, "addr": 0xDA02 },           # X position in the overworld
    "world_y": { "bank": 1, "addr": 0xDA03 },           # Y position in the overworld

    # === Pokemon Gold Flags
    "event_flags_base": { "bank": 1, "addr": 0xD7B7 },  # end is d7b6, wram bank 1 bitfield
    "textbox_flags": { "bank": 1, "addr": 0xD19C },     # wTextboxFlags - nonzero when textbox is active

    # === Scripts ===
    "script_running": { "bank": 1, "addr": 0xD15F },       # wScriptRunning - nonzero when script/dialog active

    # === Movement & Direction ===
    "player_step_direction": { "bank": None, "addr": 0xCE86 },  # wPlayerStepDirection (0=S,1=N,2=W,3=E,255=idle) 
    "player_direction": { "bank": 1, "addr": 0xD205 },          # wPlayerDirection (0=down,4=up,8=left,12=right)
    #"facing_direction": { "bank": None, "addr": 0xCF2F },       # wFacingDirection kept as ref but worse version of wPlayerDirection
    #"walking_direction": { "bank": None, "addr": 0xCF2E },      # wWalkingDirection, duplicate of player_step_direction
    #"player_walking": { "bank": 1, "addr": 0xD204 },            # wPlayerWalking - misleading

    # === Collision Detection ===
    "walking_tile_collision": { "bank": None, "addr": 0xCF32 },  # wWalkingTileCollision
    "walking_into_land": { "bank": None, "addr": 0xCF2B },       # wWalkingIntoLand
    "walking_into_edge_warp": { "bank": None, "addr": 0xCF2C },  # wWalkingIntoEdgeWarp - at door/warp, doesn't catch all door transitions

    # === Game State ===
    "battle_mode": { "bank": 1, "addr": 0xD116 },  # wBattleMode - battle active flag
    "warp_number": { "bank": 1, "addr": 0xD9FF },  # wWarpNumber - current warp/door number
    

    # === Player Resources ===
    "money_3be": {"bank": 1, "addr": 0xD573, "len": 3},  # Player's money (3 bytes, big-endian format) - uses banked format
    "bike_flag": {"bank": 1, "addr": 0xD682},  # 1 if player is on bike, 0 otherwise

    # === Pokemon Party Data ===
    "party_count": {"bank": 1, "addr": 0xDA22},  # Number of Pokemon in party (0-6)
    "party_species": [           # Species IDs for each party Pokemon (6 slots)
        {"bank": 1, "addr": 0xDA2A}, {"bank": 1, "addr": 0xDA5A}, {"bank": 1, "addr": 0xDA8A},
        {"bank": 1, "addr": 0xDABA}, {"bank": 1, "addr": 0xDAEA}, {"bank": 1, "addr": 0xDB1A}
    ],
    "party_level": [             # Level for each party Pokemon (6 slots)
        {"bank": 1, "addr": 0xDA49}, {"bank": 1, "addr": 0xDA79}, {"bank": 1, "addr": 0xDAA9},
        {"bank": 1, "addr": 0xDAD9}, {"bank": 1, "addr": 0xDB09}, {"bank": 1, "addr": 0xDB39}
    ],
    "party_hp_current": [        # Current HP for each Pokemon (2 bytes each, big-endian)
        {"bank": 1, "addr1": 0xDA4C, "addr2": 0xDA4D},  # Pokemon 1
        {"bank": 1, "addr1": 0xDA7C, "addr2": 0xDA7D},  # Pokemon 2
        {"bank": 1, "addr1": 0xDAAC, "addr2": 0xDAAD},  # Pokemon 3
        {"bank": 1, "addr1": 0xDADC, "addr2": 0xDADD},  # Pokemon 4
        {"bank": 1, "addr1": 0xDB0C, "addr2": 0xDB0D},  # Pokemon 5
        {"bank": 1, "addr1": 0xDB3C, "addr2": 0xDB3D},  # Pokemon 6
    ],
    "party_hp_max": [            # Max HP for each Pokemon (2 bytes each, big-endian)
        {"bank": 1, "addr1": 0xDA4E, "addr2": 0xDA4F},  # Pokemon 1
        {"bank": 1, "addr1": 0xDA7E, "addr2": 0xDA7F},  # Pokemon 2
        {"bank": 1, "addr1": 0xDAAE, "addr2": 0xDAAF},  # Pokemon 3
        {"bank": 1, "addr1": 0xDADE, "addr2": 0xDADF},  # Pokemon 4
        {"bank": 1, "addr1": 0xDB0E, "addr2": 0xDB0F},  # Pokemon 5
        {"bank": 1, "addr1": 0xDB3E, "addr2": 0xDB3F},  # Pokemon 6
    ],
    "party_attack": [            # Attack stat (2 bytes each, big-endian)
        {"bank": 1, "addr1": 0xDA50, "addr2": 0xDA51},  # Pokemon 1
        {"bank": 1, "addr1": 0xDA80, "addr2": 0xDA81},  # Pokemon 2
        {"bank": 1, "addr1": 0xDAB0, "addr2": 0xDAB1},  # Pokemon 3
        {"bank": 1, "addr1": 0xDAE0, "addr2": 0xDAE1},  # Pokemon 4
        {"bank": 1, "addr1": 0xDB10, "addr2": 0xDB11},  # Pokemon 5
        {"bank": 1, "addr1": 0xDB40, "addr2": 0xDB41},  # Pokemon 6
    ],
    "party_defense": [           # Defense stat (2 bytes each, big-endian)
        {"bank": 1, "addr1": 0xDA52, "addr2": 0xDA53},  # Pokemon 1
        {"bank": 1, "addr1": 0xDA82, "addr2": 0xDA83},  # Pokemon 2
        {"bank": 1, "addr1": 0xDAB2, "addr2": 0xDAB3},  # Pokemon 3
        {"bank": 1, "addr1": 0xDAE2, "addr2": 0xDAE3},  # Pokemon 4
        {"bank": 1, "addr1": 0xDB12, "addr2": 0xDB13},  # Pokemon 5
        {"bank": 1, "addr1": 0xDB42, "addr2": 0xDB43},  # Pokemon 6
    ],

    # === Pokemon PC Storage ===
    # Pokemon Gold has 14 PC boxes, each can hold up to 20 Pokemon (max 280 total)
    # SRAM addresses (0xA000-0xBFFF) commented out - need MBC bank switching support
    # "pc_box_total": 0xAD6C,      # Total Pokemon count across all boxes (may need verification)

    # "pc_current_box_species": [  # Pokemon presence in slots 1-20 of CURRENT selected box only
    #     0xAD6D, 0xAD6E, 0xAD6F, 0xAD70, 0xAD71, # Slot 1-5
    #     0xAD72, 0xAD73, 0xAD74, 0xAD75, 0xAD76, # Slot 6-10
    #     0xAD77, 0xAD78, 0xAD79, 0xAD7A, 0xAD7B, # Slot 11-15
    #     0xAD7C, 0xAD7D, 0xAD7E, 0xAD7F, 0xAD80  # Slot 16-20
    # ],

    "pc_box_selected": {"bank": 1, "addr": 0xD8BC},  # Currently selected PC box (0-13 for 14 boxes)

    "pc_box_pokemon_count": [    # Pokemon count for each of 14 PC boxes
        {"bank": 1, "addr": 0xD8BF},  # Box 1 count  - $D8BF to $D8C7
        {"bank": 1, "addr": 0xD8C8},  # Box 2 count  - $D8C8 to $D8D0
        {"bank": 1, "addr": 0xD8D1},  # Box 3 count  - $D8D1 to $D8D9
        {"bank": 1, "addr": 0xD8DA},  # Box 4 count  - $D8DA to $D8E2
        {"bank": 1, "addr": 0xD8E3},  # Box 5 count  - $D8E3 to $D8EB
        {"bank": 1, "addr": 0xD8EC},  # Box 6 count  - $D8EC to $D8F4
        {"bank": 1, "addr": 0xD8F5},  # Box 7 count  - $D8F5 to $D8FD
        {"bank": 1, "addr": 0xD8FE},  # Box 8 count  - $D8FE to $D906
        {"bank": 1, "addr": 0xD907},  # Box 9 count  - $D907 to $D90F
        {"bank": 1, "addr": 0xD910},  # Box 10 count - $D910 to $D918
        {"bank": 1, "addr": 0xD919},  # Box 11 count - $D919 to $D921
        {"bank": 1, "addr": 0xD922},  # Box 12 count - $D922 to $D92A
        {"bank": 1, "addr": 0xD92B},  # Box 13 count - $D92B to $D933
        {"bank": 1, "addr": 0xD934},  # Box 14 count - $D934 to $D93C
    ],

    # "pc_box_level": [
    #     0xADA1, 0xADC1, 0xADE1, 0xAE01, 0xAE21, # Box slot 1-5   levels
    #     0xAE41, 0xAE61, 0xAE81, 0xAEA1, 0xAEC1, # Box slot 6-10  Levels
    #     0xAEE1, 0xAF01, 0xAF21, 0xAF41, 0xAF61, # Box slot 11-15 Levels
    #     0xAF81, 0xAFA1, 0xAFC1, 0xAFE1, 0xB001  # Box slot 16-20 levels
    # ],

    # === Gym Badges ===
    "johto_badges": {"bank": 1, "addr": 0xD57C},  # Bitfield: each bit = 1 Johto badge (8 total)
    "kanto_badges": {"bank": 1, "addr": 0xD57D},  # Bitfield: each bit = 1 Kanto badge (8 total)

    # === Pokedex Tracking ===
    # Each byte represents 8 Pokemon species using bit flags (bit 0 = first species in range)
    # Total coverage: Pokemon species 1-256
    "pokedex_owned": [           # Caught Pokemon (32 bytes, 256 species)
        {"bank": 1, "addr": 0xDBE4}, {"bank": 1, "addr": 0xDBE5}, {"bank": 1, "addr": 0xDBE6}, {"bank": 1, "addr": 0xDBE7},
        {"bank": 1, "addr": 0xDBE8}, {"bank": 1, "addr": 0xDBE9}, {"bank": 1, "addr": 0xDBEA}, {"bank": 1, "addr": 0xDBEB},  # Species 1-64
        {"bank": 1, "addr": 0xDBEC}, {"bank": 1, "addr": 0xDBED}, {"bank": 1, "addr": 0xDBEE}, {"bank": 1, "addr": 0xDBEF},
        {"bank": 1, "addr": 0xDBF0}, {"bank": 1, "addr": 0xDBF1}, {"bank": 1, "addr": 0xDBF2}, {"bank": 1, "addr": 0xDBF3},  # Species 65-128
        {"bank": 1, "addr": 0xDBF4}, {"bank": 1, "addr": 0xDBF5}, {"bank": 1, "addr": 0xDBF6}, {"bank": 1, "addr": 0xDBF7},
        {"bank": 1, "addr": 0xDBF8}, {"bank": 1, "addr": 0xDBF9}, {"bank": 1, "addr": 0xDBFA}, {"bank": 1, "addr": 0xDBFB},  # Species 129-192
        {"bank": 1, "addr": 0xDBFC}, {"bank": 1, "addr": 0xDBFD}, {"bank": 1, "addr": 0xDBFE}, {"bank": 1, "addr": 0xDBFF},
        {"bank": 1, "addr": 0xDC00}, {"bank": 1, "addr": 0xDC01}, {"bank": 1, "addr": 0xDC02}, {"bank": 1, "addr": 0xDC03},  # Species 193-256
    ],
    "pokedex_seen": [            # Seen Pokemon (32 bytes, 256 species)
        {"bank": 1, "addr": 0xDC04}, {"bank": 1, "addr": 0xDC05}, {"bank": 1, "addr": 0xDC06}, {"bank": 1, "addr": 0xDC07},
        {"bank": 1, "addr": 0xDC08}, {"bank": 1, "addr": 0xDC09}, {"bank": 1, "addr": 0xDC0A}, {"bank": 1, "addr": 0xDC0B},  # Species 1-64
        {"bank": 1, "addr": 0xDC0C}, {"bank": 1, "addr": 0xDC0D}, {"bank": 1, "addr": 0xDC0E}, {"bank": 1, "addr": 0xDC0F},
        {"bank": 1, "addr": 0xDC10}, {"bank": 1, "addr": 0xDC11}, {"bank": 1, "addr": 0xDC12}, {"bank": 1, "addr": 0xDC13},  # Species 65-128
        {"bank": 1, "addr": 0xDC14}, {"bank": 1, "addr": 0xDC15}, {"bank": 1, "addr": 0xDC16}, {"bank": 1, "addr": 0xDC17},
        {"bank": 1, "addr": 0xDC18}, {"bank": 1, "addr": 0xDC19}, {"bank": 1, "addr": 0xDC1A}, {"bank": 1, "addr": 0xDC1B},  # Species 129-192
        {"bank": 1, "addr": 0xDC1C}, {"bank": 1, "addr": 0xDC1D}, {"bank": 1, "addr": 0xDC1E}, {"bank": 1, "addr": 0xDC1F},
        {"bank": 1, "addr": 0xDC20}, {"bank": 1, "addr": 0xDC21}, {"bank": 1, "addr": 0xDC22}, {"bank": 1, "addr": 0xDC23},  # Species 193-256
    ],
}

# --- Custom Watch Addresses ---
# User-defined memory addresses to display in the RAM inspector
# Each entry should have: "name" (display label), "addr" (hex address), "bank" (memory bank)
# Example: Watch player map coordinates
CUSTOM_WATCH_ADDRESSES = [
    { "name": "wPlayerMapX",        "addr": 0xD20D, "bank": 1 },
    { "name": "wPlayerMapY",        "addr": 0xD20E, "bank": 1 },
    { "name": "wScriptRunning",     "addr": 0xD15F, "bank": 1 },
    { "name": "wScriptTextBank",    "addr": 0xD175, "bank": 1 },
    { "name": "wScriptTextAddr",    "addr": 0xD176, "bank": 1 },
    { "name": "wTextBoxFlags",      "addr": 0xD19C, "bank": 1 },
    { "name": "wTextBoxFrame",      "addr": 0xD19B, "bank": 1 },
    
]

# --- Custom Event Flags ---
# User-defined event flags to display in the RAM inspector
# Two formats supported:
#   1. By name (from event_flags.asm): {"name": "EVENT_GOT_TOTODILE"}
#   2. By index: {"name": "Custom Flag Name", "index": 42}
# Event flags are stored as bitfields in wEventFlags (WRAM bank 1)
CUSTOM_EVENT_FLAGS = [
    # Add event flags here, examples:
    # {"name": "EVENT_GOT_TOTODILE"},
    # {"name": "Custom Flag 100", "index": 100},
    {"name": "EVENT_GOT_A_POKEMON_FROM_ELM"},
    {"name": "EVENT_GOT_BICYCLE"},
]
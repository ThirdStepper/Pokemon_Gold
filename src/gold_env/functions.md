# Pokemon Gold Environment - Module Usage Guide

This guide explains how to use the `gold_env` mixin modules that compose the `PokemonGoldEnv` class.

## Table of Contents
- [Introduction](#introduction)
- [ram_helpers.py - RAM Reading Utilities](#ram_helperspy---ram-reading-utilities)
- [event_flags.py - Event Flag System](#event_flagspy---event-flag-system)
- [savestate_utilities.py - Savestate Management](#savestate_utilitiespy---savestate-management)
- [exploration.py - Anti-Stuck Mechanisms](#explorationpy---anti-stuck-mechanisms)
- [rewards.py - Reward Calculation](#rewardspy---reward-calculation)
- [Common Patterns & Best Practices](#common-patterns--best-practices)
- [Appendix](#appendix)

---

## Introduction

The `gold_env` package uses **mixin classes** to organize Pokemon Gold environment functionality. Each mixin handles a specific aspect:

```python
class PokemonGoldEnv(gym.Env, RAMHelpersMixin, EventFlagsMixin,
                     SavestateUtilsMixin, ExplorationMixin, RewardsMixin):
    """Main environment - inherits all mixin methods"""
```

All mixin methods are available directly on `PokemonGoldEnv` instances via multiple inheritance.

---

## ram_helpers.py - RAM Reading Utilities

### Overview
This module provides low-level memory reading for accessing Pokemon Gold game state from PyBoy emulator RAM.

### Address Formats

**IMPORTANT:** Most methods accept **two address formats**:

1. **Bare integer** (unbanked/legacy):
   ```python
   value = self.mem8(0xD573)  # Direct WRAM0 address
   ```

2. **Banked dictionary** (WRAM1/SRAM/VRAM):
   ```python
   value = self.mem8({"bank": 1, "addr": 0xD20D})  # Banked WRAM1
   ```

The `RAM_MAP` configuration uses both formats:
```python
RAM_MAP = {
    "money_3be": (0xD573, 3),              # Legacy tuple format
    "player_x": {"bank": 1, "addr": 0xD20D},  # Banked dict format
}
```

---

### `mem8(addr_or_sym) -> int`

Read a single byte (0-255) from memory.

**Parameters:**
- `addr_or_sym`: Either:
  - `int` - Direct address (unbanked)
  - `dict` - `{"addr": int, "bank": int or None}` (banked)

**Examples:**
```python
# Unbanked read (WRAM0)
value = self.mem8(0xD573)

# Banked read (WRAM1)
value = self.mem8({"bank": 1, "addr": 0xD20D})

# Using RAM_MAP symbol (handles both formats automatically)
value = self.mem8(self.RAM["player_x"])  # Works regardless of format!
```

**When to use:**
- Direct byte access for custom RAM addresses
- Quick debugging/inspection
- Low-level memory manipulation

---

### `mem8_range(start_addr, length, bank=None) -> bytes`

Read multiple consecutive bytes from memory.

**Parameters:**
- `start_addr`: Starting address (int)
- `length`: Number of bytes to read
- `bank`: Optional bank number (None = unbanked)

**Examples:**
```python
# Read 6 bytes from unbanked address
data = self.mem8_range(0xDA22, 6, bank=None)

# Read 256 bytes from banked WRAM1 (event flags)
base = self.RAM["event_flags_base"]["addr"]
bank = self.RAM["event_flags_base"]["bank"]
bitfield = self.mem8_range(base, 0x100, bank=bank)

# Read party species IDs
party_data = self.mem8_range(0xDA2A, 6, bank=None)
```

**When to use:**
- Reading bitfields (event flags, Pokedex)
- Bulk data reads
- Snapshot comparisons

---

### `read_sym8(name) -> int`

Read a byte using a RAM_MAP symbol name (handles banking automatically).

**Parameters:**
- `name`: Symbol name from `config.RAM_MAP`

**Examples:**
```python
# Reads from RAM_MAP["player_x"] (banked or unbanked)
x = self.read_sym8("player_x")

# Handles both formats transparently
money_byte = self.read_sym8("money_3be")  # Legacy format
world_x = self.read_sym8("world_x")      # Banked format
```

**When to use:**
- Reading configured RAM addresses
- Avoiding hardcoded addresses
- Future-proofing against RAM map changes

---

### `mem16be(addr_hi_lo) -> int`

Read a 16-bit big-endian value (0-65535).

**Parameters:**
- `addr_hi_lo`: Address of the **high byte** (most significant byte first)

**Examples:**
```python
# Read Pokemon HP (stored as 2 bytes)
hp_addr = self.RAM["party_hp_max"][0][0]  # First Pokemon's max HP
hp = self.mem16be(hp_addr)

# Read attack stat
attack_addr = self.RAM["party_attack"][0][0]
attack = self.mem16be(attack_addr)
```

**When to use:**
- Reading Pokemon stats (HP, Attack, Defense, etc.)
- Any 16-bit game values

---

### `money() -> int`

Read player's current money (0-999,999).

**Returns:** Money amount as integer

**Examples:**
```python
current_money = self.money()
print(f"Player has ${current_money}")

# Track money changes
old_money = self.money()
# ... some game actions ...
new_money = self.money()
earned = new_money - old_money
```

**When to use:**
- Reward calculations
- Progress tracking
- Economy analysis

---

### `xy() -> Tuple[int, int]`

Read player's current (x, y) tile coordinates on the map.

**Returns:** `(x, y)` tuple

**Examples:**
```python
x, y = self.xy()
print(f"Player at position ({x}, {y})")

# Track movement
old_pos = self.xy()
# ... player takes action ...
new_pos = self.xy()
if old_pos == new_pos:
    print("Collision detected!")
```

**When to use:**
- Movement tracking
- Collision detection
- Exploration rewards

---

### `read_party_pokemon() -> List[dict]`

Read all Pokemon in the player's party (up to 6).

**Returns:** List of dicts with keys:
- `species`: Pokemon species ID (1-251)
- `level`: Pokemon level (1-100)
- `hp_current`: Current HP
- `hp_max`: Max HP
- `attack`: Attack stat
- `defense`: Defense stat

**Examples:**
```python
party = self.read_party_pokemon()
print(f"Party size: {len(party)}")

for i, pokemon in enumerate(party, 1):
    print(f"Pokemon {i}: Species {pokemon['species']}, Level {pokemon['level']}")
    print(f"  HP: {pokemon['hp_current']}/{pokemon['hp_max']}")
    print(f"  Attack: {pokemon['attack']}, Defense: {pokemon['defense']}")

# Calculate total party levels
total_levels = sum(p["level"] for p in party)

# Check if any Pokemon fainted
any_fainted = any(p["hp_current"] == 0 for p in party)
```

**When to use:**
- Reward calculations (level ups, stat gains)
- Party composition analysis
- Battle state tracking

---

### `get_pc_pokemon_count() -> int`

Get total number of Pokemon stored in PC boxes (0-280).

**Returns:** Total Pokemon count across all 14 PC boxes

**Examples:**
```python
pc_count = self.get_pc_pokemon_count()
print(f"PC Storage: {pc_count}/280 Pokemon")

# Track PC deposits
old_count = self.get_pc_pokemon_count()
# ... player deposits Pokemon ...
new_count = self.get_pc_pokemon_count()
deposited = new_count - old_count
```

**When to use:**
- Collection tracking
- Storage management analysis
- (Note: Currently disabled in rewards - use Pokedex instead)

---

### `get_badge_count() -> int`

Get total number of gym badges earned (0-16).

**Returns:** Badge count (8 Johto + 8 Kanto)

**Examples:**
```python
badges = self.get_badge_count()
print(f"Badges: {badges}/16")

if badges >= 8:
    print("Johto League complete!")
```

**When to use:**
- Progress tracking
- Reward calculation (badges = huge rewards!)
- Victory condition checks

---

### `get_world_xy() -> Tuple[int, int]`

Get player's world-level coordinates (overworld map).

**Returns:** `(world_x, world_y)` tuple

**Note:** Different from `xy()` which gives position within a specific map.

**Examples:**
```python
world_x, world_y = self.get_world_xy()
print(f"World position: ({world_x}, {world_y})")

# Track overworld exploration
visited_world_tiles = set()
visited_world_tiles.add(self.get_world_xy())
```

**When to use:**
- Large-scale exploration tracking
- Region movement analysis
- Map-to-map transitions

---

### `check_plot_flag(flag_name) -> bool`

Check if a story/plot flag is set.

**Parameters:**
- `flag_name`: Flag name from `config.RAM_MAP`

**Returns:** `True` if flag is set (event completed), `False` otherwise

**Examples:**
```python
if self.check_plot_flag("player_rcv_pkdex"):
    print("Player has Pokedex")

if self.check_plot_flag("starter_acquired"):
    print("Player chose starter Pokemon")

# Track newly completed flags
completed_flags = set()
for flag_name in config.PLOT_FLAG_REWARDS.keys():
    if self.check_plot_flag(flag_name):
        completed_flags.add(flag_name)
```

**When to use:**
- Story progression tracking
- Conditional rewards
- Event completion detection

**Note:** ⚠️ Plot flags currently disabled in config due to incorrect addresses. Use event flags instead (see next section).

---

### `get_seen_species() -> Set[int]`

Get set of Pokemon species that have been seen (Pokedex).

**Returns:** Set of species IDs (e.g., `{1, 4, 7, 152}`)

**Examples:**
```python
seen = self.get_seen_species()
print(f"Seen {len(seen)} species")

# Check for specific Pokemon
if 6 in seen:  # Charizard
    print("Seen Charizard!")

# Calculate new sightings
old_seen = self.get_seen_species()
# ... battle/encounter ...
new_seen = self.get_seen_species()
newly_seen = new_seen - old_seen
```

**When to use:**
- Pokedex completion tracking
- Discovery rewards
- Encounter diversity analysis

---

### `get_caught_species() -> Set[int]`

Get set of Pokemon species that have been caught (Pokedex).

**Returns:** Set of species IDs

**Examples:**
```python
caught = self.get_caught_species()
print(f"Caught {len(caught)}/251 species")

# Calculate Pokedex completion
completion_rate = len(caught) / 251 * 100
print(f"Pokedex: {completion_rate:.1f}% complete")

# Reward new catches
old_caught = self.get_caught_species()
# ... catch Pokemon ...
new_caught = self.get_caught_species()
newly_caught = new_caught - old_caught
reward = len(newly_caught) * 10.0
```

**When to use:**
- Collection rewards
- Pokedex tracking
- Catch diversity analysis

---

### `_log_plot_flag_state(flag_name, old_value, new_value, is_first_change)`

Log plot flag changes for debugging (internal method).

**Parameters:**
- `flag_name`: Name of the flag
- `old_value`: Previous byte value (0-255)
- `new_value`: Current byte value (0-255)
- `is_first_change`: `True` if first change from initial state

**When to use:**
- Debugging flag behavior
- Identifying bitfield vs byte flags
- Automatically called when `config.DEBUG_PLOT_FLAGS = True`

---

## event_flags.py - Event Flag System

### Overview

Pokemon Gold stores event state in `wEventFlags` - a 256-byte bitfield in WRAM bank 1 (0xD7B7-0xD8B6). Each event is a single bit defined in `event_flags.asm`.

**Architecture:**
- 256 bytes × 8 bits = 2,048 possible event flags
- Events defined sequentially in `event_flags.asm` using RGBDS `const` syntax
- Each event has a unique name (e.g., `EVENT_BEAT_FALKNER`)

---

### `read_gold_event_map(name) -> int`

**⭐ PRIMARY METHOD** - Read an event flag by name.

**Parameters:**
- `name`: Event name from `event_flags.asm` (e.g., `"EVENT_BEAT_FALKNER"`)

**Returns:** `1` if flag is set, `0` if not set

**Examples:**

```python
# 1. Basic reading (RECOMMENDED)
if self.read_gold_event_map("EVENT_BEAT_FALKNER"):
    print("Falkner (Gym 1) defeated!")

if self.read_gold_event_map("EVENT_GOT_BICYCLE"):
    print("Player has bicycle")

# 2. Check multiple events
gym_events = [
    "EVENT_BEAT_FALKNER",
    "EVENT_BEAT_BUGSY",
    "EVENT_BEAT_WHITNEY",
    "EVENT_BEAT_MORTY",
]

defeated_gyms = [name for name in gym_events if self.read_gold_event_map(name)]
print(f"Defeated {len(defeated_gyms)}/8 Johto gyms")

# 3. Track event completion
old_event_state = self.read_gold_event_map("EVENT_BEAT_FALKNER")
# ... battle happens ...
new_event_state = self.read_gold_event_map("EVENT_BEAT_FALKNER")
if new_event_state and not old_event_state:
    print("Just defeated Falkner!")
    reward = 50.0
```

**When to use:**
- Checking specific events by name
- Story progression tracking
- Conditional rewards
- Most common use case!

---

### `gold_event_flag_addr_mask(name) -> Tuple[int, int]`

Get the raw memory address and bitmask for an event flag.

**Parameters:**
- `name`: Event name from `event_flags.asm`

**Returns:** `(address, mask)` tuple
- `address`: Memory address (int)
- `mask`: Bitmask for the specific bit (int, power of 2)

**Raises:** `KeyError` if event name doesn't exist

**Examples:**

```python
# 1. Debug printing
addr, mask = self.gold_event_flag_addr_mask("EVENT_BEAT_FALKNER")
print(f"EVENT_BEAT_FALKNER at {hex(addr)}, mask {hex(mask)} (bit {mask.bit_length()-1})")
# Output: EVENT_BEAT_FALKNER at 0xd7c3, mask 0x20 (bit 5)

# 2. Manual bit checking
addr, mask = self.gold_event_flag_addr_mask("EVENT_GOT_BICYCLE")
byte_value = self.mem8({"bank": 1, "addr": addr})
is_set = bool(byte_value & mask)
print(f"Bicycle flag: {is_set}")

# 3. Get all addresses for a set of events
events = ["EVENT_BEAT_FALKNER", "EVENT_BEAT_BUGSY", "EVENT_BEAT_WHITNEY"]
for name in events:
    addr, mask = self.gold_event_flag_addr_mask(name)
    print(f"{name:30s} @ {hex(addr)} mask {hex(mask)}")
```

**When to use:**
- Debugging event flags
- UI/visualization tools
- Understanding memory layout
- Low-level event manipulation

---

### `gold_event_map() -> Dict[str, Tuple[int, int]]`

Get the complete mapping of all event names to (address, mask) pairs.

**Returns:** Dictionary: `{event_name: (address, mask), ...}`

**Examples:**

```python
# 1. Get full event map
events = self.gold_event_map()
print(f"Total events defined: {len(events)}")

# 2. List all event names
all_event_names = sorted(events.keys())
print("All events:", all_event_names[:20], "...")

# 3. Scan for ALL currently set flags
events = self.gold_event_map()
set_flags = [name for name, (addr, mask) in events.items()
             if self.read_gold_event_map(name)]
print(f"{len(set_flags)} flags currently set:")
print(set_flags[:20], "..." if len(set_flags) > 20 else "")

# 4. Find events in a specific address range
events = self.gold_event_map()
gym_events = {name: (addr, mask) for name, (addr, mask) in events.items()
              if "BEAT_" in name and "EVENT_" in name}
print(f"Found {len(gym_events)} gym battle events")

# 5. Group events by byte address
from collections import defaultdict
events = self.gold_event_map()
by_address = defaultdict(list)
for name, (addr, mask) in events.items():
    by_address[addr].append((name, mask))

# Show how many events share each byte
for addr, flags in sorted(by_address.items())[:10]:
    print(f"{hex(addr)}: {len(flags)} flags")
```

**When to use:**
- Comprehensive event scanning
- Event discovery/exploration
- Building event browsers/UIs
- Analysis and debugging

**Performance note:** Parsing `event_flags.asm` happens once (cached), so repeated calls are fast.

---

### `read_gold_event_flag_by_index(idx) -> int`

Read an event flag by its sequential index (0-based).

**Parameters:**
- `idx`: Bit index (0 to ~2000)

**Returns:** `1` if flag is set, `0` if not set

**Examples:**

```python
# 1. Read first 100 flags by index
for i in range(100):
    value = self.read_gold_event_flag_by_index(i)
    if value:
        print(f"Flag {i} is set")

# 2. Dump all flags in range
start, end = 0, 256
flags = [self.read_gold_event_flag_by_index(i) for i in range(start, end)]
print(f"Flags {start}-{end}: {flags}")

# 3. Bitwise comparison (before/after)
old_flags = [self.read_gold_event_flag_by_index(i) for i in range(2048)]
# ... play game ...
new_flags = [self.read_gold_event_flag_by_index(i) for i in range(2048)]
changed = [i for i in range(2048) if old_flags[i] != new_flags[i]]
print(f"Changed flag indices: {changed}")
```

**When to use:**
- Brute-force flag scanning
- When you don't have event names
- Debugging unknown flags
- Low-level analysis

**Note:** Less convenient than reading by name, but useful for discovery.

---

### `gold_parse_event_flags(path) -> Dict[str, Tuple[int, int]]`

Parse `event_flags.asm` file to build the event mapping.

**Parameters:**
- `path`: Path to `event_flags.asm` file (default: `config.EVENTFLAGS_PATH`)

**Returns:** Dictionary: `{event_name: (address, mask), ...}`

**When to use:**
- Called automatically by `gold_event_map()`
- Only call directly if you need to parse a custom file
- Cached after first call

**RGBDS Syntax Parsed:**
```asm
const_def           ; Start at index 0
const EVENT_FOOBAR  ; Index 0
const EVENT_BAZQUX  ; Index 1
const_next 100      ; Jump to index 100
const EVENT_JUMP    ; Index 100
```

---

### Advanced Event Flag Patterns

#### Pattern 1: Snapshot & Compare

```python
# Take a snapshot of entire event flag bitfield
def snapshot_events(self):
    base = self.RAM["event_flags_base"]["addr"]
    bank = self.RAM["event_flags_base"]["bank"]
    return self.mem8_range(base, 0x100, bank)  # 256 bytes = 2048 bits

# Compare snapshots
snapshot_before = snapshot_events(self)
# ... play game for 1000 steps ...
snapshot_after = snapshot_events(self)

# Find changed bytes
changed_bytes = [i for i in range(256) if snapshot_before[i] != snapshot_after[i]]
print(f"Changed {len(changed_bytes)} bytes at indices: {changed_bytes}")

# Find exact changed bits
events = self.gold_event_map()
for name, (addr, mask) in events.items():
    base_addr = self.RAM["event_flags_base"]["addr"]
    byte_offset = addr - base_addr

    old_byte = snapshot_before[byte_offset]
    new_byte = snapshot_after[byte_offset]

    old_bit = bool(old_byte & mask)
    new_bit = bool(new_byte & mask)

    if old_bit != new_bit:
        print(f"{name}: {old_bit} -> {new_bit}")
```

#### Pattern 2: Event Flag Watcher

```python
class EventFlagWatcher:
    def __init__(self, env):
        self.env = env
        self.last_state = {}

    def update(self):
        """Check for newly set flags"""
        events = self.env.gold_event_map()
        newly_set = []

        for name in events:
            current = self.env.read_gold_event_map(name)
            previous = self.last_state.get(name, 0)

            if current and not previous:
                newly_set.append(name)

            self.last_state[name] = current

        return newly_set

# Usage:
watcher = EventFlagWatcher(env)

for step in range(1000):
    obs, reward, done, truncated, info = env.step(action)

    new_flags = watcher.update()
    if new_flags:
        print(f"Step {step}: New events: {new_flags}")
```

#### Pattern 3: Event-Based Rewards

```python
# Custom reward function based on specific events
CUSTOM_EVENT_REWARDS = {
    "EVENT_BEAT_FALKNER": 100.0,
    "EVENT_BEAT_BUGSY": 150.0,
    "EVENT_GOT_BICYCLE": 50.0,
    "EVENT_RADIO_TOWER_ROCKET_TAKEOVER": 200.0,
}

def calculate_event_rewards(self):
    """Add to reward calculation"""
    reward = 0.0

    for event_name, event_reward in CUSTOM_EVENT_REWARDS.items():
        if event_name not in self._completed_events:
            try:
                if self.read_gold_event_map(event_name):
                    reward += event_reward
                    self._completed_events.add(event_name)
                    print(f"Event completed: {event_name} (+{event_reward})")
            except KeyError:
                # Event not in event_flags.asm
                pass

    return reward
```

#### Pattern 4: Quick Brute-Force Scan

```python
# Find all currently set flags (quick & dirty)
events = self.gold_event_map()
set_now = [name for name, (addr, mask) in events.items()
           if self.read_gold_event_map(name)]

print(f"\n=== {len(set_now)} FLAGS CURRENTLY SET ===")
for name in sorted(set_now):
    addr, mask = self.gold_event_flag_addr_mask(name)
    print(f"  {name:40s} @ {hex(addr)} mask {hex(mask)}")
```

---

## savestate_utilities.py - Savestate Management

### Overview

Savestates capture the complete emulator state (CPU, RAM, registers) for instant save/load. Used to:
- Skip game intro sequences
- Quickly reset episodes
- Create training checkpoints

---

### `_load_init_state_if_any() -> bool`

Load a savestate at episode start (internal method).

**Returns:** `True` if savestate was loaded, `False` otherwise

**Priority:**
1. `self.boot_state` (in-memory BytesIO) - fastest
2. `self.init_state_path` (disk file) - persistent

**Examples:**
```python
# Called automatically in reset()
# You typically don't call this directly

# In reset():
if not self._load_init_state_if_any():
    if self.require_init_state:
        raise RuntimeError("No savestate found!")
```

**When to use:**
- Automatically called by `reset()`
- Don't call directly unless customizing reset logic

---

### `capture_boot_state(to_path=None)`

Save current emulator state for fast episode restarts.

**Parameters:**
- `to_path`: Optional path to save to disk (Path object)

**Examples:**

```python
# 1. Create in-memory savestate only
env.capture_boot_state()
# Now env.boot_state contains the state (BytesIO)

# 2. Save to disk for persistence
from pathlib import Path
save_path = Path("states/after_intro.state")
env.capture_boot_state(to_path=save_path)

# 3. Create savestate at specific game location
# Play manually to desired location:
for _ in range(1000):
    env.step(5)  # Press A button

# Capture state when ready
env.capture_boot_state(to_path=Path("states/custom_start.state"))
print("Savestate captured!")
```

**When to use:**
- First-time setup (run `make_boot_state.py`)
- Creating custom starting positions
- Saving interesting game states
- Building a library of savestates

**Workflow:**
1. Boot environment
2. Manually play to desired state (skip intro, etc.)
3. Call `capture_boot_state()` with a path
4. Set `config.INIT_STATE_PATH` to that file
5. Future episodes start from that state

---

## exploration.py - Anti-Stuck Mechanisms

### Overview

These methods detect and discourage "stuck" behavior:
- Wall-running (pressing same direction repeatedly)
- Pacing in small areas
- Repetitive action patterns
- Never leaving spawn area

All methods are called automatically by `_calculate_reward()`.

---

### `_calculate_novelty_bonus(tile) -> float`

Calculate reward bonus for visiting tiles based on recency.

**Parameters:**
- `tile`: `(x, y)` coordinate tuple

**Returns:** Bonus value (0.0 to `config.NOVELTY_BONUS_SCALE`)

**Formula:** `SCALE * (1 - exp(-steps_since_visit / decay_rate))`

**Examples:**
```python
# Called automatically in _calculate_reward()
tile = self.xy()
novelty = self._calculate_novelty_bonus(tile)
reward += novelty

# First visit: full bonus (e.g., 0.5)
# Recent visit: small bonus (e.g., 0.05)
# Old visit: high bonus again (e.g., 0.4)
```

**Configuration:**
```python
config.NOVELTY_BONUS_ENABLED = True
config.NOVELTY_BONUS_SCALE = 0.5      # Max bonus
config.NOVELTY_DECAY_STEPS = 100      # Freshness decay rate
```

**When to use:**
- Automatically used in reward calculation
- Encourages revisiting old areas
- Prevents local minima

---

### `_detect_stuck() -> float`

Detect if agent is stuck in a small area.

**Returns:** Penalty (0.0 or negative, e.g., -0.5)

**Logic:**
- Analyzes recent position history (last N steps)
- Counts unique tiles visited
- If too few unique tiles → stuck penalty

**Examples:**
```python
# Called automatically in _calculate_reward()
stuck_penalty = self._detect_stuck()
reward += stuck_penalty  # Adds negative value if stuck

# Example: Agent visited only 5 unique tiles in last 50 steps
# Result: -0.5 penalty applied
```

**Configuration:**
```python
config.STUCK_DETECTION_ENABLED = True
config.STUCK_DETECTION_WINDOW = 50    # Analyze last 50 steps
config.STUCK_RADIUS_THRESHOLD = 8     # Min unique tiles to avoid penalty
config.STUCK_PENALTY = -0.5           # Penalty when stuck
```

**When to use:**
- Automatically used in reward calculation
- Discourages pacing/looping behavior
- Forces exploration

---

### `_calculate_action_diversity() -> float`

Reward diverse actions, penalize repetitive patterns.

**Returns:** Bonus/penalty based on action entropy

**Logic:**
- Calculates Shannon entropy of recent actions
- High entropy (diverse) → bonus
- Low entropy (repetitive) → penalty
- Wall-running detection (4+ identical moves) → extra penalty

**Examples:**
```python
# Called automatically in _calculate_reward()
diversity = self._calculate_action_diversity()
reward += diversity

# Diverse actions (entropy > 0.7): +0.05 bonus
# Repetitive actions (entropy < 0.3): -0.1 penalty
# Wall-running (UP, UP, UP, UP): -0.2 penalty
```

**Configuration:**
```python
config.ACTION_DIVERSITY_ENABLED = True
config.ACTION_DIVERSITY_WINDOW = 20        # Analyze last 20 actions
config.ACTION_DIVERSITY_BONUS = 0.05       # Reward for high entropy
config.ACTION_REPETITION_PENALTY = -0.1    # Penalty for low entropy
config.WALL_RUN_PENALTY = -0.2             # Extra penalty for wall-running
config.WALL_RUN_THRESHOLD = 4              # Consecutive moves for wall-run
```

**When to use:**
- Automatically used in reward calculation
- Prevents action spam
- Encourages intelligent behavior

---

### `_calculate_distance_reward() -> float`

Reward distance from spawn position (encourages outward exploration).

**Returns:** Reward based on Manhattan distance from spawn

**Logic:**
- Calculated periodically (every N steps for performance)
- Manhattan distance: `|x - spawn_x| + |y - spawn_y|`
- Larger distance → larger reward

**Examples:**
```python
# Called automatically in _calculate_reward()
distance_reward = self._calculate_distance_reward()
reward += distance_reward

# Spawn at (10, 10), current at (20, 25)
# Distance = |20-10| + |25-10| = 25 tiles
# Reward = 25 * 0.01 = 0.25
```

**Configuration:**
```python
config.DISTANCE_REWARD_ENABLED = True
config.DISTANCE_REWARD_SCALE = 0.01        # Reward per tile
config.DISTANCE_REWARD_INTERVAL = 10       # Check every 10 steps
```

**When to use:**
- Automatically used in reward calculation
- Encourages leaving starting area
- Balances with local exploration

---

## rewards.py - Reward Calculation

### Overview

The `_calculate_reward()` method computes comprehensive rewards based on all game state changes. This drives agent learning.

---

### `_calculate_reward() -> float`

Calculate reward for the current step.

**Returns:** Reward value (clipped to ±`config.REWARD_CLIP_ABS`)

**Reward Hierarchy (highest → lowest):**

| Category | Reward | Description |
|----------|--------|-------------|
| **Badges** | 15-50 per badge | Gym victories (MASSIVE rewards) |
| **Story Events** | 10-40 per event | Major milestones (currently disabled) |
| **Species Caught** | 10.0 per species | New Pokedex entry |
| **Party Pokemon** | 8.0 per Pokemon | Catching new Pokemon |
| **Species Seen** | 2.0 per species | First encounter |
| **Map Bank** | 5.0 | New region discovered |
| **World Tile** | 2.0 | New overworld tile |
| **Map Number** | 1.0 | New map within region |
| **Pokemon Level** | 0.5 per level | Training Pokemon |
| **Novelty Bonus** | 0.0-0.5 | Revisiting old tiles |
| **Action Diversity** | ±0.05 to ±0.2 | Action patterns |
| **New Tile** | 0.02 | First visit to tile |
| **Money** | 0.01 per unit | Gaining money |
| **Step Penalty** | -0.0001 | Efficiency incentive |
| **Repeated Tile** | -0.005 | Discourages loops |
| **Collision** | -0.01 (escalating) | Running into walls |
| **Stuck Penalty** | -0.5 | Pacing in small area |

**Examples:**

```python
# Called automatically in step()
reward = self._calculate_reward()

# Typical reward breakdown:
# - Visited new tile: +0.02
# - Novelty bonus: +0.15
# - Step penalty: -0.0001
# - Total: +0.1699

# Big event:
# - Defeated gym leader: +50.0 (badge)
# - Leveled up Pokemon: +2.5 (5 level ups)
# - New map explored: +5.0
# - Total: +57.5 (huge!)
```

**Configuration:**

All reward values are in `config.py`:

```python
# Movement
config.NEW_TILE_REWARD = 0.02
config.REPEATED_TILE_PENALTY = -0.005
config.COLLISION_PENALTY = -0.01

# Exploration
config.NEW_MAP_BANK_REWARD = 5.0
config.NEW_MAP_NUMBER_REWARD = 1.0
config.NEW_WORLD_TILE_REWARD = 2.0

# Pokemon
config.NEW_POKEMON_CAUGHT_REWARD = 8.0
config.NEW_SPECIES_CAUGHT_REWARD = 10.0
config.POKEMON_LEVEL_UP_REWARD = 0.5

# Badges (escalating)
config.BADGE_REWARDS = [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]

# Anti-stuck
config.NOVELTY_BONUS_SCALE = 0.5
config.STUCK_PENALTY = -0.5
config.WALL_RUN_PENALTY = -0.2
```

**When to use:**
- Automatically called in `step()`
- Modify values in `config.py` to tune agent behavior
- Don't call directly unless debugging

**State Variables Tracked:**
```python
self._visited_tiles              # Set of (x,y) coordinates
self._visited_map_banks          # Set of map bank IDs
self._visited_map_numbers        # Set of map number IDs
self._visited_world_tiles        # Set of (world_x, world_y)
self._seen_species               # Set of Pokemon species IDs
self._caught_species             # Set of caught species IDs
self._last_party_count           # Previous party size
self._last_total_levels          # Previous total levels
self._last_badge_count           # Previous badge count
self._collision_count            # Consecutive collisions
# ... and more
```

---

## Common Patterns & Best Practices

### Pattern 1: Tracking Game Progress

```python
# In your training loop or custom environment wrapper
class ProgressTracker:
    def __init__(self, env):
        self.env = env
        self.episode_start_badges = 0
        self.episode_start_caught = set()

    def on_reset(self):
        """Called at episode start"""
        self.episode_start_badges = self.env.get_badge_count()
        self.episode_start_caught = self.env.get_caught_species()

    def on_step(self):
        """Called after each step"""
        # Check for badge progress
        current_badges = self.env.get_badge_count()
        if current_badges > self.episode_start_badges:
            print(f"🏆 Badge earned! Total: {current_badges}/16")

        # Check for new catches
        current_caught = self.env.get_caught_species()
        new_catches = current_caught - self.episode_start_caught
        if new_catches:
            print(f"✨ Caught {len(new_catches)} new species!")
```

### Pattern 2: Custom Event-Based Logic

```python
def check_story_progress(env):
    """Check major story milestones using event flags"""
    milestones = {
        "Started Journey": "EVENT_PLAYER_LEFT_NEW_BARK",
        "Got Starter": "EVENT_GOT_STARTER",
        "First Gym": "EVENT_BEAT_FALKNER",
        "Got Bicycle": "EVENT_GOT_BICYCLE",
        "Radio Tower": "EVENT_RADIO_TOWER_ROCKET_TAKEOVER",
    }

    completed = []
    for milestone, event_name in milestones.items():
        try:
            if env.read_gold_event_map(event_name):
                completed.append(milestone)
        except KeyError:
            pass  # Event not defined

    return completed
```

### Pattern 3: Debugging RAM Addresses

```python
def debug_ram_addresses(env):
    """Print useful RAM values for debugging"""
    print("=== RAM State ===")
    print(f"Position: {env.xy()}")
    print(f"World: {env.get_world_xy()}")
    print(f"Money: ${env.money()}")
    print(f"Badges: {env.get_badge_count()}/16")

    # Map info
    map_bank = env.mem8(env.RAM["map_bank"])
    map_num = env.mem8(env.RAM["map_number"])
    print(f"Map: Bank {map_bank}, Number {map_num}")

    # Party info
    party = env.read_party_pokemon()
    print(f"Party: {len(party)} Pokemon")
    for i, p in enumerate(party, 1):
        print(f"  {i}. Species {p['species']}, Lv{p['level']}, "
              f"HP {p['hp_current']}/{p['hp_max']}")

    # Pokedex
    seen = env.get_seen_species()
    caught = env.get_caught_species()
    print(f"Pokedex: {len(caught)} caught, {len(seen)} seen")
```

### Pattern 4: When to Use Banked vs Unbanked Reads

```python
# UNBANKED (bank=None or bare int)
# - WRAM0 (0xC000-0xCFFF)
# - Some WRAM1 (0xD000-0xDFFF if not banked)
money = self.mem8(0xD573)                    # Unbanked
party_count = self.mem8(0xDA22)              # Unbanked

# BANKED (bank=1 or other)
# - WRAM1 bank-switched regions (some 0xDxxx addresses)
# - SRAM (save data, bank 0-3)
# - VRAM (video RAM, bank 0-1)
player_x = self.mem8({"bank": 1, "addr": 0xD20D})  # WRAM1

# AUTOMATIC (using RAM_MAP symbols)
# Let the RAM_MAP handle it!
x = self.mem8(self.RAM["player_x"])  # Automatically handles banking
```

**Rule of thumb:**
- Use RAM_MAP symbols when possible (safest, most maintainable)
- Use bare int for quick debugging of WRAM0 addresses
- Use banked dict for WRAM1/SRAM/VRAM or when RAM_MAP specifies it

---

## Appendix

### WRAM Banking Explained

Game Boy Color has banked Work RAM (WRAM):

- **WRAM0** (0xC000-0xCFFF): Always accessible, no banking
- **WRAM1** (0xD000-0xDFFF): Can be bank-switched (banks 1-7)

Pokemon Gold uses **bank 1** for most game state (`wEventFlags`, player position, etc.).

**Accessing banked RAM:**
```python
# PyBoy supports: memory[bank, address]
value = self.mem[1, 0xD20D]  # Bank 1, address 0xD20D

# Our wrapper:
value = self.mem8({"bank": 1, "addr": 0xD20D})
```

### Event Flag Naming Conventions

From `event_flags.asm`, events follow patterns:

- `EVENT_BEAT_*` - Gym leaders, rivals, trainers
- `EVENT_GOT_*` - Items received
- `EVENT_*_ROCKET_*` - Team Rocket encounters
- `EVENT_*_GYM_*` - Gym-related events
- `EVENT_TALKED_TO_*` - NPC conversations

**Finding events:**
```python
# List all gym battle events
events = env.gold_event_map()
gym_battles = [name for name in events if "BEAT_" in name and "GYM" not in name]
print(gym_battles)

# List all item events
item_events = [name for name in events if "GOT_" in name or "ITEM_" in name]
print(item_events)
```

### RAM_MAP Structure Reference

Two formats coexist:

```python
RAM_MAP = {
    # Legacy format (tuple for multi-byte values)
    "money_3be": (0xD573, 3),  # (address, byte_count)

    # Banked format (dict with optional bank)
    "player_x": {"bank": 1, "addr": 0xD20D},
    "player_y": {"bank": 1, "addr": 0xD20E},
    "map_bank": {"bank": 1, "addr": 0xDA00},

    # Unbanked format (dict with bank=None)
    "some_flag": {"bank": None, "addr": 0xD500},

    # Array formats
    "party_species": [0xDA2A, 0xDA5A, ...],  # 6 addresses
    "pokedex_owned": [0xDBE4, 0xDBE5, ...],  # 32 addresses
}
```

### Debugging Tips

**1. Verify event flag parsing:**
```python
events = env.gold_event_map()
if not events:
    print("ERROR: No events parsed!")
    print(f"Check file exists: {config.EVENTFLAGS_PATH}")
```

**2. Monitor memory changes:**
```python
# Watch a specific address
addr = {"bank": 1, "addr": 0xD20D}
old_value = env.mem8(addr)
# ... take action ...
new_value = env.mem8(addr)
if old_value != new_value:
    print(f"Address {addr} changed: {old_value} -> {new_value}")
```

**3. Log all set event flags at episode start:**
```python
def log_initial_events(env):
    events = env.gold_event_map()
    set_events = [name for name in events if env.read_gold_event_map(name)]
    print(f"Episode starts with {len(set_events)} events set:")
    for name in sorted(set_events):
        print(f"  - {name}")
```

**4. Reward breakdown:**
```python
# Add detailed logging to _calculate_reward()
def _calculate_reward_verbose(self):
    components = {}

    # Movement
    cur_xy = self.xy()
    if cur_xy not in self._visited_tiles:
        components['new_tile'] = config.NEW_TILE_REWARD
    else:
        components['repeated_tile'] = config.REPEATED_TILE_PENALTY

    # ... calculate all components ...

    total = sum(components.values())
    print(f"Reward: {total:.4f} = {components}")
    return total
```

---

## Quick Reference

### Most Common Operations

```python
# Position
x, y = env.xy()
world_x, world_y = env.get_world_xy()

# Progress
badges = env.get_badge_count()
money = env.money()
party = env.read_party_pokemon()

# Pokedex
seen = env.get_seen_species()
caught = env.get_caught_species()

# Events (MOST USEFUL)
if env.read_gold_event_map("EVENT_BEAT_FALKNER"):
    print("Gym 1 complete!")

# Scan all set events
events = env.gold_event_map()
set_flags = [n for n in events if env.read_gold_event_map(n)]

# RAM reading
value = env.mem8(env.RAM["player_x"])  # Automatic banking
```

---

## Additional Resources

- **Pokemon Gold Disassembly**: https://github.com/pret/pokegold
- **RAM Map**: https://datacrystal.tcrf.net/wiki/Pok%C3%A9mon_Gold_and_Silver/RAM_map
- **Event Flags**: Check `rom/event_flags.asm` in your project
- **Config Reference**: `src/config.py` for all tunable values

---

**Last Updated:** 2025-01-28
**Compatible with:** Pokemon Gold RL Environment v1.0

# ram_helpers.py
"""
RAM Reading Helper Methods for Pokemon Gold Environment

This mixin provides low-level memory reading utilities for accessing
game state data from the Game Boy's RAM via PyBoy emulator.
"""

from typing import Tuple, Set, List
from datetime import datetime


class RAMHelpersMixin:
    """
    Mixin providing RAM reading utilities for Pokemon Gold.

    These methods read game state from the Game Boy's memory (RAM).
    Source: https://datacrystal.tcrf.net/wiki/Pok%C3%A9mon_Gold_and_Silver/RAM_map

    Required attributes from parent class:
    - self.mem: PyBoy memory interface
    - self.RAM: RAM_MAP configuration dictionary
    - self._plot_flags_log_file: Optional file handle for plot flag logging
    """

    def mem8(self, addr_or_sym) -> int:
        """
        Read a single byte (8 bits) from memory at the given address or symbol.
        Supports:
          - int absolute addresses (WRAM0/etc.)
          - dict symbols: {"addr": int, "bank": Optional[int]}
          For banked WRAM (e.g., bank=1), PyBoy supports mem[bank, addr]
        """
        if isinstance(addr_or_sym, dict):
            addr = addr_or_sym["addr"]
            bank = addr_or_sym["bank"]
            return (int(self.mem[addr]) if bank is None else int(self.mem[bank, addr])) & 0xFF
        return int(self.mem[addr_or_sym]) & 0xFF

    def mem8_banked(self, bank: int | None, addr: int) -> int:
        """
        Read a single byte from a specific memory bank.

        Args:
            bank: Memory bank number (None for unbanked)
            addr: Memory address to read

        Returns:
            Byte value (0-255)
        """
        if bank is None:
            return int(self.mem[addr]) & 0xFF
        return int(self.mem[bank, addr]) & 0xFF

    def mem8_range(self, start_addr: int, length: int, bank: int | None = None) -> bytes:
        if bank is None:
            return bytes(int(self.mem[start_addr+i]) & 0xFF for i in range(length))
        return bytes(int(self.mem[bank, start_addr+i]) & 0xFF for i in range(length))

    def read_sym8(self, name: str) -> int:
        """
        Read a single byte using a RAM_MAP symbol that may be banked or unbanked.
        RAM_MAP[name] must be either an int (legacy) or a dict {"addr": int, "bank": Optional[int]}.
        """
        sym = self.RAM[name]
        if isinstance(sym, int):
            # Backward-compatible: treat as unbanked absolute address
            return self.mem8(sym)
        addr = sym["addr"]
        bank = sym.get("bank", None)
        if bank is None:
            # Unbanked or WRAM0
            return self.mem8(addr)
        # Banked (e.g., WRAMX, VRAM, SRAM)
        return self.mem8_banked(bank, addr)

    def mem16be(self, addr_or_dict):
        """
        Read a 16-bit big-endian value from memory.

        Big-endian means the most significant byte comes first.
        For example, if addr has 0x12 and addr+1 has 0x34, this returns 0x1234.

        Supports:
          - int: reads from addr and addr+1 (legacy format)
          - dict: {"bank": int, "addr1": int, "addr2": int} - reads both addresses with banking

        Args:
            addr_or_dict: Either an int address (legacy) or dict with bank, addr1, addr2

        Returns:
            The 16-bit value (0-65535)
        """
        if isinstance(addr_or_dict, dict):
            # New format: {"bank": 1, "addr1": X, "addr2": Y}
            bank = addr_or_dict.get("bank", None)
            addr1 = addr_or_dict["addr1"]
            addr2 = addr_or_dict["addr2"]
            hi = self.mem8({"bank": bank, "addr": addr1})
            lo = self.mem8({"bank": bank, "addr": addr2})
            return (hi << 8) | lo
        else:
            # Legacy format: int address
            hi = self.mem8(addr_or_dict)
            lo = self.mem8(addr_or_dict + 1)
            return (hi << 8) | lo

    def money(self) -> int:
        """
        Read the player's money (3-byte big-endian at money_3be).
        Supports either a legacy tuple (base_addr, length) or a banked dict:
          {"bank": 1, "addr": 0xD573, "len": 3}

        Returns:
            Current money amount (0-999999)
        """
        sym = self.RAM["money_3be"]
        if isinstance(sym, tuple):
            base, n = sym
            assert n == 3
            b0 = self.mem8(base)
            b1 = self.mem8(base + 1)
            b2 = self.mem8(base + 2)
        else:
            bank = sym.get("bank")
            base = sym["addr"]
            # Explicitly read from the specified bank to avoid active-bank drift
            b0 = self.mem8_banked(bank, base)
            b1 = self.mem8_banked(bank, base + 1)
            b2 = self.mem8_banked(bank, base + 2)
        return (b0 << 16) | (b1 << 8) | b2

    def xy(self) -> Tuple[int, int]:
        """
        Read the player's current (x, y) coordinates on the map.

        Returns:
            Tuple of (x, y) coordinates
        """
        return (self.mem8(self.RAM["player_x"]), self.mem8(self.RAM["player_y"]))

    def read_party_pokemon(self) -> List[dict]:
        """
        Read all party Pokemon data.

        This reads species, level, HP, Attack, and Defense for each Pokemon
        in the player's party (up to 6 Pokemon).

        Returns:
            List of dicts with keys: species, level, hp_current, hp_max, attack, defense
            Empty list if no Pokemon in party
        """
        party_count = self.mem8(self.RAM["party_count"])
        party = []

        for i in range(min(party_count, 6)):
            species = self.mem8(self.RAM["party_species"][i])
            if species == 0 or species == 0xFF:  # Empty slot
                continue

            level = self.mem8(self.RAM["party_level"][i])
            hp_current = self.mem16be(self.RAM["party_hp_current"][i])
            hp_max = self.mem16be(self.RAM["party_hp_max"][i])
            attack = self.mem16be(self.RAM["party_attack"][i])
            defense = self.mem16be(self.RAM["party_defense"][i])

            party.append({
                "species": species,
                "level": level,
                "hp_current": hp_current,
                "hp_max": hp_max,
                "attack": attack,
                "defense": defense,
            })

        return party

    def get_pc_pokemon_count(self) -> int:
        """
        Get total number of Pokemon stored in PC boxes.

        Pokemon Gold has 14 PC boxes, each can hold up to 20 Pokemon.
        This sums the count from each individual box for accuracy.

        Returns:
            Total Pokemon count in PC storage (0-280)
        """
        # Sum Pokemon count from all 14 boxes
        total = 0
        for box_addr in self.RAM["pc_box_pokemon_count"]:
            count = self.mem8(box_addr)
            # Sanity check: each box can hold max 20 Pokemon
            if count <= 20:
                total += count
            else:
                # If we get invalid data, skip this box
                continue

        return total if total <= 280 else 0  # Max 14 boxes * 20 per box = 280

    def get_badge_count(self) -> int:
        """
        Get total number of gym badges earned.

        Badges are stored as bitfields where each bit represents one badge.
        Pokemon Gold has 8 Johto badges and 8 Kanto badges.

        Returns:
            Total badge count (0-16)
        """
        johto = self.mem8(self.RAM["johto_badges"])
        kanto = self.mem8(self.RAM["kanto_badges"])

        # Count set bits (each bit = 1 badge)
        return bin(johto).count('1') + bin(kanto).count('1')

    def get_world_xy(self) -> Tuple[int, int]:
        """
        Get the player's world coordinates (overworld map level).

        This is different from xy() which gives coordinates within a specific map.
        World coordinates track position on the larger overworld map.

        Returns:
            Tuple of (world_x, world_y) coordinates
        """
        return (self.mem8(self.RAM["world_x"]), self.mem8(self.RAM["world_y"]))

    def check_plot_flag(self, flag_name: str) -> bool:
        """
        Check if a story/plot flag has been set.

        Plot flags mark important story events like receiving Pokemon from Elm,
        clearing the Radio Tower, beating gym leaders, etc.

        This reads from wEventFlags (banked WRAM1) as a bitfield using event_flags.asm.

        Args:
            flag_name: Name of the event flag (e.g., "EVENT_GOT_A_POKEMON_FROM_ELM")

        Returns:
            True if flag is set (event completed), False otherwise
        """
        try:
            return bool(self.read_gold_event_map(flag_name))
        except KeyError:
            # Flag name not found in event_flags.asm
            return False

    def get_seen_species(self) -> Set[int]:
        """
        Get the set of Pokemon species that have been seen (Pokedex).

        Reads 32 bytes of bitfield data where each bit represents one species.
        Species IDs range from 1-256 (though Gen 2 only has 251 species).

        Returns:
            Set of species IDs that have been seen (e.g., {1, 4, 7, 152})
        """
        seen = set()
        pokedex_seen_addrs = self.RAM["pokedex_seen"]

        for byte_index, addr in enumerate(pokedex_seen_addrs):
            byte_value = self.mem8(addr)
            # Check each bit in the byte
            for bit_index in range(8):
                if byte_value & (1 << bit_index):
                    # Species ID = (byte_index * 8) + bit_index + 1
                    # +1 because species start at 1, not 0
                    species_id = (byte_index * 8) + bit_index + 1
                    seen.add(species_id)

        return seen

    def get_caught_species(self) -> Set[int]:
        """
        Get the set of Pokemon species that have been caught (Pokedex).

        Reads 32 bytes of bitfield data where each bit represents one species.
        Species IDs range from 1-256 (though Gen 2 only has 251 species).

        Returns:
            Set of species IDs that have been caught (e.g., {1, 4, 7, 152})
        """
        caught = set()
        pokedex_owned_addrs = self.RAM["pokedex_owned"]

        for byte_index, addr in enumerate(pokedex_owned_addrs):
            byte_value = self.mem8(addr)
            # Check each bit in the byte
            for bit_index in range(8):
                if byte_value & (1 << bit_index):
                    # Species ID = (byte_index * 8) + bit_index + 1
                    # +1 because species start at 1, not 0
                    species_id = (byte_index * 8) + bit_index + 1
                    caught.add(species_id)

        return caught

    def _log_plot_flag_state(self, flag_name: str, old_value: int, new_value: int, is_first_change: bool):
        """
        Log plot flag change with before/after values in multiple formats.

        This diagnostic method helps identify whether plot flags are bitfields
        (where individual bits toggle) or whole bytes (typically 0x00 → 0x01).

        Args:
            flag_name: Name of the plot flag (e.g., "player_rcv_pkdex")
            old_value: Previous byte value at the flag's address
            new_value: Current byte value at the flag's address
            is_first_change: True if this is the first change from initial state
        """
        if not self._plot_flags_log_file or self._plot_flags_log_file.closed:
            return

        addr = self.RAM.get(flag_name, "UNKNOWN")
        change_type = "FIRST CHANGE" if is_first_change else "SUBSEQUENT CHANGE"

        self._plot_flags_log_file.write(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] {change_type}\n"
            f"  Flag: {flag_name}\n"
            f"  Address: {addr:#06x} ({addr})\n"
            f"  BEFORE: {old_value:#04x} (dec: {old_value:3d}, bin: {old_value:08b})\n"
            f"  AFTER:  {new_value:#04x} (dec: {new_value:3d}, bin: {new_value:08b})\n"
            f"\n"
        )
        self._plot_flags_log_file.flush()

    def near_npc(self) -> bool:
        """
        Check if player is adjacent to an NPC (within interaction range).

        This reads the wWalkingIntoNPC RAM address which is set by the game
        when the player attempts to walk into a tile occupied by an NPC.

        Returns:
            True if player is next to an NPC and facing them, False otherwise
        """
        return self.mem8(self.RAM["walking_into_npc"]) > 0

    def in_dialog(self) -> bool:
        """
        Check if a dialog or script is currently active.

        This checks both wTextboxFlags (textbox visible) and wScriptRunning
        (script executing). Either condition means the player is in a dialog
        state and shouldn't be penalized for not moving.

        Returns:
            True if dialog/script is active, False otherwise
        """
        textbox_active = self.mem8(self.RAM["textbox_flags"]) > 0
        script_running = self.mem8(self.RAM["script_running"]) > 0
        return textbox_active or script_running

    def get_facing_direction(self) -> int:
        """
        Get the direction the player is currently facing.

        Game Boy Pokemon uses specific values for directions:
            0 = Down
            4 = Up
            8 = Left
            12 = Right

        Returns:
            Direction value (0, 4, 8, or 12)
        """
        return self.mem8(self.RAM["facing_direction"])

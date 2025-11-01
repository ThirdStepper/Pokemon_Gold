# event_flags.py
"""
Event Flags (wEventFlags) Helper Methods for Pokemon Gold Environment

This mixin provides utilities for reading Pokemon Gold event flags from WRAM.
Event flags track game state such as items received, NPCs met, events triggered, etc.
"""

import config


class EventFlagsMixin:
    """
    Mixin providing event flag reading utilities for Pokemon Gold.

    Event flags are stored in wEventFlags (banked WRAM1) as a sequential bitfield.
    Each event has a unique bit index defined in event_flags.asm.

    Required attributes from parent class:
    - self.RAM: RAM_MAP configuration dictionary
    - self.mem8(): Method to read a byte from memory
    """

    # Lazy-initialized on first use
    GOLD_EVENT_FLAG_MAP: dict[str, tuple[int, int]] | None = None  # name -> (addr, mask)

    def gold_parse_event_flags(self, path: str = "event_flags.asm") -> dict[str, tuple[int, int]]:
        """
        Parse event_flags.asm (RGBDS-style) to build name -> (addr, mask).
        Layout: wEventFlags at RAM_MAP["event_flags_base"]["addr"] (banked WRAM1).
        Each EVENT_* is a sequential bit index starting from 0, with jumps via const_next.
        """
        import re, io, os
        base = self.RAM["event_flags_base"]["addr"]
        const_re = re.compile(r'^\s*const\s+([A-Za-z0-9_]+)')
        const_def_re  = re.compile(r'^\s*const_def\b')
        const_next_re = re.compile(r'^\s*const_next\s+(\d+)')
        idx = 0
        mapping: dict[str, tuple[int, int]] = {}
        if not os.path.exists(path):
            return mapping
        with io.open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if const_def_re.search(line):
                    idx = 0
                    continue
                mnext = const_next_re.search(line)
                if mnext:
                    idx = int(mnext.group(1))
                    continue
                m = const_re.search(line)
                if m:
                    name = m.group(1)
                    byte_off = idx >> 3
                    bit = idx & 7
                    addr = base + byte_off
                    mask = 1 << bit
                    mapping[name] = (addr, mask)
                    idx += 1
        return mapping

    def gold_event_map(self) -> dict[str, tuple[int, int]]:
        if self.GOLD_EVENT_FLAG_MAP is None:
            self.GOLD_EVENT_FLAG_MAP = self.gold_parse_event_flags(config.EVENTFLAGS_PATH)
        return self.GOLD_EVENT_FLAG_MAP

    def gold_event_flag_addr_mask(self, name: str) -> tuple[int, int]:
        """
        Return (addr, mask) for an EVENT_* flag from events_flag.asm
        Raises KeyError if the name doesn't exist
        """
        return self.gold_event_map()[name]

    def read_gold_event_map(self, name: str) -> int:
        """
        Read the Event_* bit as 0/1 from wEventFlags (banked WRAM1).
        Uses RAM_MAP['event_flags_base] to locate the block
        """
        addr, mask = self.gold_event_flag_addr_mask(name)
        bank = self.RAM["event_flags_base"].get("bank", None)
        byte = self.mem8({"addr": addr, "bank": bank})
        return 1 if (byte & mask) else 0

    def read_gold_event_flag_by_index(self, idx: int) -> int:
        """
        Read a flag by sequential index (0-based) without needing a name.
        Useful for debugging. Index -> byte/bit inside wEventFlags.
        """

        base = self.RAM["event_flags_base"]["addr"]
        bank = self.RAM["event_flags_base"].get("bank", None)
        addr = base + (idx >> 3)
        mask = 1 << ( idx & 7)
        byte = self.mem8({"bank": bank, "addr": addr})
        return 1 if (byte & mask) else 0

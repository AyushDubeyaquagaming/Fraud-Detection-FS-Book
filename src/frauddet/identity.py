"""Phone-number normalization — the de-facto join key across collections.

CLAUDE.md rule: phone strings are compared as strings; normalize country-code
and leading-zero variants in ONE util, with tests. Phase 1 found three live
formats for the same Ugandan subscriber:

    players.username / contactNo : "0757575757"   (national, leading 0, 10 digits)
    gateway userId               : "751111111"    (bare 9-digit subscriber)
    withdrawal recipientId       : "256757897897" (E.164 country code 256)

Canonical form = the bare 9-digit subscriber number (drop 256, drop leading 0).
This is the key every flattened table will carry-join on.
"""
from __future__ import annotations

import re

# Uganda mobile subscriber numbers are 9 digits after the leading 0 / +256.
_SUBSCRIBER_LEN = 9


def normalize_phone(value) -> str | None:
    """Return the canonical 9-digit subscriber number, or None if implausible.

    Deterministic and string-based (no float coercion — leading zeros matter).
    Order matters: strip the 256 country code before the national leading 0,
    so "2560757..." collapses correctly.
    """
    if value is None:
        return None
    digits = re.sub(r"\D", "", str(value))
    if not digits:
        return None
    if digits.startswith("256") and len(digits) > _SUBSCRIBER_LEN:
        digits = digits[3:]
    if digits.startswith("0") and len(digits) > _SUBSCRIBER_LEN:
        digits = digits[1:]
    return digits or None

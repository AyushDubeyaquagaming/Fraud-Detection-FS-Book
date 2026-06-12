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


_OBJECTID_HEX = re.compile(r"^[0-9a-fA-F]{24}$")


def is_objectid_hex(value) -> bool:
    """True if value looks like a stringified Mongo ObjectId (24 hex chars).

    Needed because `walletaccounts.userId` is polymorphic: most rows hold the
    players._id hex string, a few hold a phone (verified Phase 1b). Real phone
    strings are <=12 digits, so the two forms never collide.
    """
    return bool(_OBJECTID_HEX.match(str(value))) if value is not None else False


def looks_like_test_number(phone: str | None) -> bool:
    """ADVISORY heuristic: does a normalized phone look like a made-up test number?

    Flags: a digit repeated 5+ times ("751111111"), a 2-digit block repeated 3+
    times ("751010101"), a 3-digit block repeated twice ("789789789"), or the
    whole number using <=2 distinct digits ("6666699999").

    NEVER use as an exclusion key: on dev data most REAL registered players also
    match (they are themselves test accounts). Classification evidence only.
    """
    if not phone:
        return False
    p = str(phone)
    return bool(
        re.search(r"(\d)\1{4}", p)
        or re.search(r"(\d{2})\1{2}", p)
        or re.search(r"(\d{3})\1", p)
        or len(set(p)) <= 2
    )


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

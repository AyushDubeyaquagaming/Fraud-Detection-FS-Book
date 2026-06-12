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

from dataclasses import dataclass
import re
from typing import Any, Iterable

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


@dataclass(frozen=True)
class IdentityResult:
    """Result of resolving a source identifier to the canonical player key."""

    player_key: str | None
    unjoined_class: str | None


class IdentityMapper:
    """One resolver for all flattening joins.

    The mapper resolves the forms verified in Phase 1/1b:
    normalized phone, `players._id` hex strings, and walletaccount-style
    polymorphic IDs. It also maps `players.playerId` for player referral fields;
    source event tables should still use phones/ObjectIds through this same API.
    """

    def __init__(
        self,
        *,
        phone_to_key: dict[str, str],
        objectid_to_key: dict[str, str],
        player_id_to_key: dict[str, str] | None = None,
        preregistration_phones: set[str] | None = None,
    ) -> None:
        self.phone_to_key = phone_to_key
        self.objectid_to_key = {k.lower(): v for k, v in objectid_to_key.items()}
        self.player_id_to_key = player_id_to_key or {}
        self.preregistration_phones = preregistration_phones or set()

    @classmethod
    def from_players(
        cls,
        players: Iterable[dict[str, Any]],
        preregistration_docs: Iterable[dict[str, Any]] | None = None,
    ) -> "IdentityMapper":
        """Build a mapper from raw player documents and optional OTP docs."""
        phone_to_key: dict[str, str] = {}
        objectid_to_key: dict[str, str] = {}
        player_id_to_key: dict[str, str] = {}

        for player in players:
            player_key = str(player.get("_id"))
            if not player_key:
                continue
            objectid_to_key[player_key.lower()] = player_key

            for phone_field in ("username", "contactNo"):
                phone = normalize_phone(player.get(phone_field))
                if phone:
                    phone_to_key[phone] = player_key

            player_id = player.get("playerId")
            if player_id is not None:
                player_id_to_key[str(player_id)] = player_key

        preregistration_phones: set[str] = set()
        if preregistration_docs is not None:
            for doc in preregistration_docs:
                phone = normalize_phone(doc.get("contactNo"))
                if phone and phone not in phone_to_key:
                    preregistration_phones.add(phone)

        return cls(
            phone_to_key=phone_to_key,
            objectid_to_key=objectid_to_key,
            player_id_to_key=player_id_to_key,
            preregistration_phones=preregistration_phones,
        )

    def resolve(self, value: Any) -> IdentityResult:
        """Resolve a source identifier to `(player_key, unjoined_class)`.

        `unjoined_class` is only populated when no player is resolved, and is
        one of `pre_registration`, `test_pattern`, or `unknown`.
        """
        text = "" if value is None else str(value).strip()
        if text:
            if is_objectid_hex(text):
                player_key = self.objectid_to_key.get(text.lower())
                if player_key:
                    return IdentityResult(player_key, None)

            if text in self.player_id_to_key:
                return IdentityResult(self.player_id_to_key[text], None)

        phone = normalize_phone(value)
        if phone:
            player_key = self.phone_to_key.get(phone)
            if player_key:
                return IdentityResult(player_key, None)
            if phone in self.preregistration_phones:
                return IdentityResult(None, "pre_registration")
            if looks_like_test_number(phone):
                return IdentityResult(None, "test_pattern")

        return IdentityResult(None, "unknown")

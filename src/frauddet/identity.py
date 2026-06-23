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


@dataclass(frozen=True)
class IdentityCollision:
    """A normalized phone attached to multiple player documents."""

    phone: str
    winning_key: str
    losing_keys: tuple[str, ...]


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
        referral_code_to_key: dict[str, str] | None = None,
        preregistration_phones: set[str] | None = None,
        phone_collisions: list[IdentityCollision] | None = None,
    ) -> None:
        self.phone_to_key = phone_to_key
        self.objectid_to_key = {k.lower(): v for k, v in objectid_to_key.items()}
        self.player_id_to_key = player_id_to_key or {}
        self.referral_code_to_key = referral_code_to_key or {}
        self.preregistration_phones = preregistration_phones or set()
        self.phone_collisions = phone_collisions or []

    @classmethod
    def from_players(
        cls,
        players: Iterable[dict[str, Any]],
        preregistration_docs: Iterable[dict[str, Any]] | None = None,
    ) -> "IdentityMapper":
        """Build a mapper from raw player documents and optional OTP docs.

        The source system can reuse a phone across player documents. We keep a
        deterministic winner for joins and also expose the collision so reports
        can flag the ambiguity.
        """
        player_docs = list(players)
        phone_to_key: dict[str, str] = {}
        objectid_to_key: dict[str, str] = {}
        player_id_to_key: dict[str, str] = {}
        referral_code_to_key: dict[str, str] = {}
        phone_candidates: dict[str, list[dict[str, Any]]] = {}

        for player in player_docs:
            raw_player_key = player.get("_id")
            if raw_player_key is None:
                continue
            player_key = str(raw_player_key)
            objectid_to_key[player_key.lower()] = player_key

            # Some player docs carry the same number in username/contactNo with
            # different formatting. Normalize both and treat them as one phone.
            player_phones = {
                phone
                for phone in (
                    normalize_phone(player.get("username")),
                    normalize_phone(player.get("contactNo")),
                )
                if phone
            }
            for phone in player_phones:
                phone_candidates.setdefault(phone, []).append(player)

            player_id = player.get("playerId")
            if player_id is not None:
                player_id_to_key[str(player_id)] = player_key

            referral_code = player.get("referralCode")
            if referral_code:
                referral_code_to_key[str(referral_code).strip()] = player_key

        phone_collisions: list[IdentityCollision] = []
        for phone, candidates in phone_candidates.items():
            winner = _choose_phone_owner(candidates)
            winning_key = str(winner.get("_id"))
            phone_to_key[phone] = winning_key
            if len(candidates) > 1:
                losing_keys = tuple(
                    str(player.get("_id"))
                    for player in sorted(
                        candidates,
                        key=lambda p: str(p.get("_id")),
                    )
                    if str(player.get("_id")) != winning_key
                )
                phone_collisions.append(
                    IdentityCollision(
                        phone=phone,
                        winning_key=winning_key,
                        losing_keys=losing_keys,
                    )
                )

        # OTP rows help distinguish "not a player yet" from truly unknown
        # source identifiers in unjoined reports.
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
            referral_code_to_key=referral_code_to_key,
            preregistration_phones=preregistration_phones,
            phone_collisions=phone_collisions,
        )

    def resolve(
        self,
        value: Any,
        *,
        allow_player_id: bool = False,
        allow_referral_code: bool = False,
    ) -> IdentityResult:
        """Resolve a source identifier to `(player_key, unjoined_class)`.

        Source event tables resolve only verified join forms by default:
        normalized phone and `players._id` hex. Player IDs and referral codes
        are player-profile fields, so callers must opt in explicitly.

        `unjoined_class` is only populated when no player is resolved, and is
        one of `pre_registration`, `test_pattern`, or `unknown`.
        """
        text = "" if value is None else str(value).strip()
        if text:
            if is_objectid_hex(text):
                player_key = self.objectid_to_key.get(text.lower())
                if player_key:
                    return IdentityResult(player_key, None)

            if allow_player_id and text in self.player_id_to_key:
                return IdentityResult(self.player_id_to_key[text], None)

            if allow_referral_code and text in self.referral_code_to_key:
                return IdentityResult(self.referral_code_to_key[text], None)

        phone = normalize_phone(value)
        if phone:
            player_key = self.phone_to_key.get(phone)
            if player_key:
                return IdentityResult(player_key, None)
            # Classify unresolved phones for reporting only. These classes do
            # not drop data; downstream features simply require player_key.
            if phone in self.preregistration_phones:
                return IdentityResult(None, "pre_registration")
            if looks_like_test_number(phone):
                return IdentityResult(None, "test_pattern")

        return IdentityResult(None, "unknown")


def _choose_phone_owner(players: list[dict[str, Any]]) -> dict[str, Any]:
    """Resolve reused phones: prefer non-deleted player, then latest createdAt."""
    return max(
        players,
        key=lambda player: (
            not bool(player.get("isDeleted", False)),
            _created_at_sort_value(player.get("createdAt")),
            str(player.get("_id")),
        ),
    )


def _created_at_sort_value(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)

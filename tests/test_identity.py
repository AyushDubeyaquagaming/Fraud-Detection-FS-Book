"""Tests for phone normalization — the cross-collection join key."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frauddet.identity import IdentityMapper, is_objectid_hex, looks_like_test_number, normalize_phone


def test_three_live_formats_collapse_to_same_subscriber():
    # The same subscriber appears in three forms across collections.
    national = normalize_phone("0757897897")     # players.username
    e164 = normalize_phone("256757897897")        # withdrawal recipientId
    assert national == e164 == "757897897"


def test_bare_nine_digit_kept():
    assert normalize_phone("751111111") == "751111111"


def test_whitespace_and_separators_stripped():
    assert normalize_phone("  0757 575 757 ") == "757575757"
    assert normalize_phone("+256-757-575-757") == "757575757"


def test_country_code_stripped_before_leading_zero():
    # Defensive: "2560757..." should collapse the same as the national form.
    assert normalize_phone("2560757575757") == normalize_phone("0757575757")


def test_empty_and_none():
    assert normalize_phone(None) is None
    assert normalize_phone("") is None
    assert normalize_phone("abc") is None


def test_string_based_no_float_coercion():
    # Leading zeros must survive (never treated as a number).
    assert normalize_phone("0700000001") == "700000001"


def test_is_objectid_hex():
    assert is_objectid_hex("6a0ea9ff174ad3c431d9e16d")          # real players._id
    assert not is_objectid_hex("0757575757")                     # phone
    assert not is_objectid_hex("6a0ea9ff174ad3c431d9e16")        # 23 chars
    assert not is_objectid_hex("6a0ea9ff174ad3c431d9e16z")       # non-hex char
    assert not is_objectid_hex(None)


def test_looks_like_test_number_flags_patterns():
    assert looks_like_test_number("751111111")    # digit x5
    assert looks_like_test_number("751010101")    # 2-digit block x3
    assert looks_like_test_number("789789789")    # 3-digit block x2
    assert looks_like_test_number("6666699999")   # <=2 distinct digits


def test_looks_like_test_number_passes_ordinary_numbers():
    assert not looks_like_test_number("759876543")
    assert not looks_like_test_number("702133888")  # '888' alone is not enough
    assert not looks_like_test_number(None)
    assert not looks_like_test_number("")


def test_identity_mapper_resolves_phone_forms_and_player_id():
    mapper = IdentityMapper.from_players(
        [
            {
                "_id": "6a0ea9ff174ad3c431d9e16d",
                "username": "0757575757",
                "contactNo": "0757575757",
                "playerId": 10003905,
            }
        ]
    )

    assert mapper.resolve("0757575757").player_key == "6a0ea9ff174ad3c431d9e16d"
    assert mapper.resolve("757575757").player_key == "6a0ea9ff174ad3c431d9e16d"
    assert mapper.resolve("+256 757 575 757").player_key == "6a0ea9ff174ad3c431d9e16d"
    assert mapper.resolve("10003905").player_key == "6a0ea9ff174ad3c431d9e16d"


def test_identity_mapper_resolves_players_objectid_hex():
    key = "6a0ea9ff174ad3c431d9e16d"
    mapper = IdentityMapper.from_players([{"_id": key, "username": "0757575757"}])

    assert mapper.resolve(key.upper()).player_key == key


def test_identity_mapper_classifies_junk_and_unresolvable_inputs():
    mapper = IdentityMapper.from_players(
        [{"_id": "6a0ea9ff174ad3c431d9e16d", "username": "0757575757"}],
        [{"contactNo": "0759999999"}],
    )

    assert mapper.resolve("not-a-player").unjoined_class == "unknown"
    assert mapper.resolve("0759999999").unjoined_class == "pre_registration"
    assert mapper.resolve("0751111111").unjoined_class == "test_pattern"


def test_identity_mapper_phone_collision_prefers_non_deleted_player():
    mapper = IdentityMapper.from_players(
        [
            {
                "_id": "deleted-player",
                "username": "0757575757",
                "contactNo": "0757575757",
                "isDeleted": True,
                "createdAt": "2026-06-02T00:00:00Z",
            },
            {
                "_id": "active-player",
                "username": "256757575757",
                "contactNo": "0757575757",
                "isDeleted": False,
                "createdAt": "2026-06-01T00:00:00Z",
            },
        ]
    )

    assert mapper.resolve("0757575757").player_key == "active-player"
    assert len(mapper.phone_collisions) == 1
    collision = mapper.phone_collisions[0]
    assert collision.phone == "757575757"
    assert collision.winning_key == "active-player"
    assert collision.losing_keys == ("deleted-player",)


def test_identity_mapper_phone_collision_tiebreaks_latest_created_at():
    mapper = IdentityMapper.from_players(
        [
            {
                "_id": "older-player",
                "username": "0757575757",
                "isDeleted": False,
                "createdAt": "2026-06-01T00:00:00Z",
            },
            {
                "_id": "newer-player",
                "username": "0757575757",
                "isDeleted": False,
                "createdAt": "2026-06-02T00:00:00Z",
            },
        ]
    )

    assert mapper.resolve("0757575757").player_key == "newer-player"


def test_identity_mapper_resolves_referral_code_exact_match():
    mapper = IdentityMapper.from_players(
        [{"_id": "referrer", "username": "0757575757", "referralCode": "ABC123"}]
    )

    assert mapper.resolve("ABC123").player_key == "referrer"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("identity: all tests passed")

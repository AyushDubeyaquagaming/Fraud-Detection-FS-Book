"""Tests for phone normalization — the cross-collection join key."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frauddet.identity import is_objectid_hex, looks_like_test_number, normalize_phone


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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("identity: all tests passed")

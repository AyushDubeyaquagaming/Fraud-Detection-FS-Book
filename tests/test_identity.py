"""Tests for phone normalization — the cross-collection join key."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frauddet.identity import normalize_phone


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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("identity: all tests passed")

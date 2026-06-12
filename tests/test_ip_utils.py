"""Tests for client-IP extraction from XFF chains."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frauddet import ip_utils

# A few real Cloudflare edges (in the configured ranges) and non-CF clients.
CF_EDGE = "162.159.122.165"      # in 162.158.0.0/15
CF_EDGE2 = "104.16.5.5"          # in 104.16.0.0/13
CLIENT = "80.227.165.206"        # a real client (Uganda ISP), not CF


def test_is_cloudflare_true_for_edges():
    assert ip_utils.is_cloudflare(CF_EDGE)
    assert ip_utils.is_cloudflare(CF_EDGE2)


def test_is_cloudflare_false_for_client_and_garbage():
    assert not ip_utils.is_cloudflare(CLIENT)
    assert not ip_utils.is_cloudflare("not-an-ip")
    assert not ip_utils.is_cloudflare("")


def test_split_chain_handles_string_and_list():
    assert ip_utils.split_chain(f"{CLIENT}, {CF_EDGE}") == [CLIENT, CF_EDGE]
    assert ip_utils.split_chain([CLIENT, CF_EDGE]) == [CLIENT, CF_EDGE]
    assert ip_utils.split_chain(None) == []
    assert ip_utils.split_chain("  ") == []


def test_extract_client_ip_picks_first_non_cf():
    # Real-world shape from useractivitylogs: client first, CF edge second.
    assert ip_utils.extract_client_ip(f"{CLIENT}, {CF_EDGE}") == CLIENT
    # Even if CF edges precede, the first non-CF wins.
    assert ip_utils.extract_client_ip(f"{CF_EDGE}, {CLIENT}") == CLIENT
    # All-Cloudflare chain -> no real client.
    assert ip_utils.extract_client_ip(f"{CF_EDGE}, {CF_EDGE2}") is None
    assert ip_utils.extract_client_ip(None) is None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("ip_utils: all tests passed")

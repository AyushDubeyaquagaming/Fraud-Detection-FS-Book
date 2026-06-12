"""Client-IP extraction from X-Forwarded-For chains.

CLAUDE.md hard rule: never build shared-IP features on raw chain contents —
Cloudflare edge IPs would fabricate links between unrelated players. The client
IP is the first non-Cloudflare element of the chain. This is the ONE place that
logic lives.

Phase 1 reality check: in `useractivitylogs`, `ip_address` is a COMMA-SEPARATED
STRING (e.g. "80.227.165.206, 162.159.122.165"), not an array. This module
accepts either form.
"""
from __future__ import annotations

import ipaddress
from functools import lru_cache

from . import config


@lru_cache(maxsize=4)
def _networks(cidrs: tuple[str, ...]) -> tuple[ipaddress.IPv4Network, ...]:
    return tuple(ipaddress.ip_network(c) for c in cidrs)


def cloudflare_networks() -> tuple[ipaddress.IPv4Network, ...]:
    """Cloudflare CIDRs from config.yaml, parsed and cached."""
    return _networks(tuple(config.load_config()["cloudflare_cidrs"]))


def is_cloudflare(ip: str, networks=None) -> bool:
    """True if ip falls in any Cloudflare CIDR. Unparseable strings -> False."""
    if networks is None:
        networks = cloudflare_networks()
    try:
        addr = ipaddress.ip_address(ip.strip())
    except (ValueError, AttributeError):
        return False
    return any(addr in net for net in networks)


def split_chain(raw) -> list[str]:
    """Normalize an XFF value (comma-separated string or list) to a clean list."""
    if raw is None:
        return []
    parts = raw if isinstance(raw, list) else str(raw).split(",")
    return [str(p).strip() for p in parts if str(p).strip()]


def extract_client_ip(raw, networks=None) -> str | None:
    """First non-Cloudflare IP in the chain (the real client), or None.

    The client is the left-most element; Cloudflare edges sit to the right.
    Returns None if the chain is empty or every element is a CF edge.
    """
    if networks is None:
        networks = cloudflare_networks()
    for ip in split_chain(raw):
        if not is_cloudflare(ip, networks):
            return ip
    return None

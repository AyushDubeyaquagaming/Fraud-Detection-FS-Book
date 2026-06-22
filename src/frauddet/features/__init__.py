"""Player-level feature builders operating on the frozen snapshot."""

from .multi_accounting import build_multi_accounting_features

__all__ = ["build_multi_accounting_features"]

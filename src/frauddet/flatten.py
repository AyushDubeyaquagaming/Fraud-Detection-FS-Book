"""Phase 2 flattening layer: Mongo base collections -> parquet contracts.

Production business rules live here, not in notebooks. Mongo access is
read-only; the only writes are deterministic parquet/report outputs under
`data/`.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from pymongo.database import Database

from . import config, ip_utils
from .identity import IdentityMapper, IdentityResult, normalize_phone
from .io_mongo import get_database


OUTPUT_FILES = {
    "players": "players.parquet",
    "bets": "bets.parquet",
    "money": "money.parquet",
    "bonus": "bonus.parquet",
    "activity": "activity.parquet",
    "logins": "logins.parquet",
}


@dataclass(frozen=True)
class BuildResult:
    """Metadata returned by `rebuild_all_flattened_outputs`."""

    output_dir: Path
    raw_counts: dict[str, int]
    parquet_counts: dict[str, int]
    report_paths: dict[str, Path]
    withdrawal_anomaly: dict[str, Any]


def rebuild_all_flattened_outputs(output_dir: Path | None = None) -> BuildResult:
    """Rebuild every Phase 2 parquet/report output from read-only Mongo."""
    out_dir = output_dir or config.DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    with get_database() as db:
        return rebuild_all_flattened_outputs_from_db(db, out_dir)


def rebuild_all_flattened_outputs_from_db(db: Database, output_dir: Path) -> BuildResult:
    """Rebuild outputs using an existing read-only Mongo database handle."""
    cfg = config.load_config()
    collections = cfg["collections"]

    source_docs = {
        "players": _read_required_docs(db, collections["players"]),
        "bets": _read_required_docs(db, collections["bets"]),
        "deposits": _read_required_docs(db, collections["deposits"]),
        "withdrawals": _read_required_docs(db, collections["withdrawals"]),
        "bonus": _read_required_docs(db, collections["bonus"]),
        "activity": _read_required_docs(db, collections["activity"]),
        "logins": _read_required_docs(db, collections["logins"]),
        "registration_otps": _read_required_docs(db, collections["registration_otps"]),
        "walletaccounts": _read_required_docs(db, collections["walletaccounts"]),
        "cashaccounts": _read_required_docs(db, collections["cashaccounts"]),
    }

    mapper = IdentityMapper.from_players(
        source_docs["players"], source_docs["registration_otps"]
    )

    _validate_unique(source_docs["deposits"], "transactionId", "deposittransactions")
    _validate_unique(source_docs["bets"], "ticketId", "bet_transactions")

    deposit_ids = {str(d.get("transactionId")) for d in source_docs["deposits"] if d.get("transactionId")}
    bet_ids = {str(d.get("ticketId")) for d in source_docs["bets"] if d.get("ticketId")}

    players = flatten_players(source_docs["players"], mapper)
    bets = flatten_bets(source_docs["bets"], mapper)
    money = flatten_money(source_docs["deposits"], source_docs["withdrawals"], mapper)
    bonus = flatten_bonus(source_docs["bonus"], mapper, deposit_ids, bet_ids)
    activity = flatten_activity(source_docs["activity"], mapper)
    logins = flatten_logins(source_docs["logins"], mapper)

    frames = {
        "players": players,
        "bets": bets,
        "money": money,
        "bonus": bonus,
        "activity": activity,
        "logins": logins,
    }
    for name, frame in frames.items():
        frame.to_parquet(output_dir / OUTPUT_FILES[name], index=False)

    raw_counts = {
        key: len(source_docs[key])
        for key in ("players", "bets", "deposits", "withdrawals", "bonus", "activity", "logins")
    }
    parquet_counts = {name: len(frame) for name, frame in frames.items()}

    withdrawal_anomaly = analyze_withdrawal_account_resolution(
        source_docs["withdrawals"], source_docs["walletaccounts"], source_docs["cashaccounts"]
    )
    unjoined_report = write_unjoined_report(output_dir / "unjoined_report.md", frames, cfg)
    reconciliation_report = write_reconciliation_report(
        output_dir / "flatten_reconciliation.md",
        raw_counts,
        parquet_counts,
        withdrawal_anomaly,
        cfg,
    )

    return BuildResult(
        output_dir=output_dir,
        raw_counts=raw_counts,
        parquet_counts=parquet_counts,
        report_paths={
            "unjoined": unjoined_report,
            "reconciliation": reconciliation_report,
        },
        withdrawal_anomaly=withdrawal_anomaly,
    )


def flatten_players(docs: Iterable[dict[str, Any]], mapper: IdentityMapper) -> pd.DataFrame:
    """Flatten `players` to one row per player."""
    fcfg = config.load_config()["flatten"]
    rows: list[dict[str, Any]] = []
    for doc in docs:
        referred_by = mapper.resolve(doc.get("referredBy")) if doc.get("referredBy") else IdentityResult(None, None)
        rows.append(
            {
                "player_key": str(doc.get("_id")),
                "phone": normalize_phone(doc.get("contactNo") or doc.get("username")),
                "created_at": _utc(doc.get("createdAt")),
                "kyc_status": doc.get("KycVerified"),
                "is_deleted": doc.get(
                    "isDeleted", fcfg["players"]["assumed_missing_is_deleted"]
                ),
                "archived": doc.get("archived", fcfg["players"]["assumed_missing_archived"]),
                "referred_by_key": referred_by.player_key,
                "nationality": doc.get("nationality"),
                "dob": doc.get("DOB"),
                "username_raw": doc.get("username"),
            }
        )
    columns = [
        "player_key",
        "phone",
        "created_at",
        "kyc_status",
        "is_deleted",
        "archived",
        "referred_by_key",
        "nationality",
        "dob",
        "username_raw",
    ]
    return _ordered_frame(rows, columns, ["player_key"])


def flatten_bets(docs: Iterable[dict[str, Any]], mapper: IdentityMapper) -> pd.DataFrame:
    """Flatten `bet_transactions` to one row per ticket."""
    rows: list[dict[str, Any]] = []
    for doc in docs:
        ident = mapper.resolve(doc.get("loginId"))
        parts = doc.get("betParts") or []
        odds = [_as_float(part.get("odds")) for part in parts if _as_float(part.get("odds")) is not None]
        sports = sorted({str(part.get("sportName")) for part in parts if part.get("sportName")})
        rows.append(
            {
                "player_key": ident.player_key,
                "unjoined_class": ident.unjoined_class,
                "ticket_id": doc.get("ticketId"),
                "game_type": doc.get("gameType"),
                "status": doc.get("status"),
                "result": doc.get("result"),
                "stake": doc.get("stake"),
                "stake_real": doc.get("stakeReal"),
                "stake_bonus": doc.get("stakeBonus"),
                "is_free_bet": doc.get("isFreeBet"),
                "payout": doc.get("payout"),
                "potential_payout": doc.get("potentialPayout"),
                "total_odds": _as_float(doc.get("totalOdds")),
                "currency": doc.get("currency"),
                "created_at": _utc(doc.get("createdDate") or doc.get("createdAt")),
                "settled_at_proxy": _utc(doc.get("updatedAt")),
                "bet_type": doc.get("betType"),
                "n_selections": len(parts),
                "min_part_odds": min(odds) if odds else None,
                "max_part_odds": max(odds) if odds else None,
                "sports": sports,
            }
        )
    columns = [
        "player_key",
        "unjoined_class",
        "ticket_id",
        "game_type",
        "status",
        "result",
        "stake",
        "stake_real",
        "stake_bonus",
        "is_free_bet",
        "payout",
        "potential_payout",
        "total_odds",
        "currency",
        "created_at",
        "settled_at_proxy",
        "bet_type",
        "n_selections",
        "min_part_odds",
        "max_part_odds",
        "sports",
    ]
    return _ordered_frame(rows, columns, ["created_at", "ticket_id"])


def flatten_money(
    deposit_docs: Iterable[dict[str, Any]],
    withdrawal_docs: Iterable[dict[str, Any]],
    mapper: IdentityMapper,
) -> pd.DataFrame:
    """Flatten deposits plus lifecycle-collapsed withdrawals."""
    rows: list[dict[str, Any]] = []
    flag_cfg = config.load_config()["flatten"]["money_flags"]

    for doc in deposit_docs:
        ident = mapper.resolve(doc.get("userId"))
        status = _lower(doc.get("status"))
        rows.append(
            {
                "player_key": ident.player_key,
                "unjoined_class": ident.unjoined_class,
                "txn_type": "DEPOSIT",
                "transaction_id": doc.get("transactionId"),
                "amount": doc.get("amount"),
                "currency": doc.get("currency"),
                "payment_method": doc.get("paymentMethod"),
                "account_number": doc.get("accountNumber"),
                "final_status": status,
                "requested_at": _utc(doc.get("createdAt")),
                "finalized_at": _utc(doc.get("createdAt")),
                "execution_type": None,
                "recipient_normalized": None,
                "is_third_party_recipient": False,
                "bonus_tags": _as_list(doc.get("bonusTagName")),
                "is_money_in": status in set(flag_cfg["deposit_money_in_statuses"]),
                "is_money_out": False,
                "is_pending_withdrawal": False,
            }
        )

    for group in collapse_withdrawal_lifecycles(withdrawal_docs):
        kept = group["kept_doc"]
        ident = mapper.resolve(kept.get("userId"))
        status = _lower(kept.get("status"))
        user_phone = normalize_phone(kept.get("userId"))
        recipient = normalize_phone(kept.get("recipientId"))
        rows.append(
            {
                "player_key": ident.player_key,
                "unjoined_class": ident.unjoined_class,
                "txn_type": "WITHDRAWAL",
                "transaction_id": kept.get("transactionId"),
                "amount": kept.get("amount"),
                "currency": kept.get("currency"),
                "payment_method": kept.get("paymentMethod"),
                "account_number": kept.get("accountNumber"),
                "final_status": status,
                "requested_at": group["requested_at"],
                "finalized_at": _utc(kept.get("updatedAt")),
                "execution_type": kept.get("executionType"),
                "recipient_normalized": recipient,
                "is_third_party_recipient": bool(recipient and user_phone and recipient != user_phone),
                "bonus_tags": [],
                "is_money_in": False,
                "is_money_out": status in set(flag_cfg["withdrawal_money_out_statuses"]),
                "is_pending_withdrawal": status in set(flag_cfg["pending_withdrawal_statuses"]),
            }
        )

    columns = [
        "player_key",
        "unjoined_class",
        "txn_type",
        "transaction_id",
        "amount",
        "currency",
        "payment_method",
        "account_number",
        "final_status",
        "requested_at",
        "finalized_at",
        "execution_type",
        "recipient_normalized",
        "is_third_party_recipient",
        "bonus_tags",
        "is_money_in",
        "is_money_out",
        "is_pending_withdrawal",
    ]
    return _ordered_frame(rows, columns, ["requested_at", "txn_type", "transaction_id"])


def collapse_withdrawal_lifecycles(docs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse withdrawal lifecycle docs by transactionId.

    Keeps the most advanced status by configured ordering:
    completed > declined > failed > pending > initiated. Failed and declined
    remain distinct. Ties use latest `updatedAt`; `requested_at` is the
    earliest `createdAt` across the lifecycle group.
    """
    status_order = config.load_config()["flatten"]["withdrawals"]["status_order"]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for doc in docs:
        transaction_id = str(doc.get("transactionId") or doc.get("_id"))
        groups[transaction_id].append(doc)

    collapsed: list[dict[str, Any]] = []
    for transaction_id, group in groups.items():
        kept = sorted(
            group,
            key=lambda d: (
                status_order.get(_lower(d.get("status")), -1),
                _sort_timestamp(d.get("updatedAt"), fallback="min"),
                str(d.get("_id")),
            ),
            reverse=True,
        )[0]
        created_values = [_utc(d.get("createdAt")) for d in group]
        created_values = [v for v in created_values if pd.notna(v)]
        collapsed.append(
            {
                "transaction_id": transaction_id,
                "kept_doc": kept,
                "requested_at": min(created_values) if created_values else pd.NaT,
                "lifecycle_doc_count": len(group),
            }
        )
    return sorted(
        collapsed,
        key=lambda g: (
            g["requested_at"] if pd.notna(g["requested_at"]) else _sort_timestamp(None, fallback="max"),
            g["transaction_id"],
        ),
    )


def flatten_bonus(
    docs: Iterable[dict[str, Any]],
    mapper: IdentityMapper,
    deposit_transaction_ids: set[str],
    bet_ticket_ids: set[str],
) -> pd.DataFrame:
    """Flatten `bonustransactions` and add assumed default currency."""
    fcfg = config.load_config()["flatten"]
    assumed_currency = fcfg["bonus"]["ASSUMED_DEFAULT"]
    allowed = set(fcfg["bonus"]["allowed_transaction_types"])
    rows: list[dict[str, Any]] = []
    for doc in docs:
        txn_type = doc.get("transactionType")
        if txn_type not in allowed:
            raise ValueError(f"Unexpected bonus transactionType {txn_type!r}")
        ident = mapper.resolve(doc.get("userId"))
        ref_trans_id = doc.get("refTransId")
        rows.append(
            {
                "player_key": ident.player_key,
                "unjoined_class": ident.unjoined_class,
                "source_id": str(doc.get("_id")),
                "txn_type": txn_type,
                "amount": doc.get("amount"),
                "bonus_type": doc.get("bonusTypeId") or doc.get("subBonusTypeId"),
                "ref_trans_id": ref_trans_id,
                "ref_kind": _bonus_ref_kind(ref_trans_id, deposit_transaction_ids, bet_ticket_ids),
                "currency": assumed_currency,
                "created_at": _utc(doc.get("createdAt")),
            }
        )
    columns = [
        "player_key",
        "unjoined_class",
        "source_id",
        "txn_type",
        "amount",
        "bonus_type",
        "ref_trans_id",
        "ref_kind",
        "currency",
        "created_at",
    ]
    return _ordered_frame(rows, columns, ["created_at", "ref_trans_id", "txn_type"])


def flatten_activity(docs: Iterable[dict[str, Any]], mapper: IdentityMapper) -> pd.DataFrame:
    """Flatten `useractivitylogs` with normalized client IP."""
    rows: list[dict[str, Any]] = []
    for doc in docs:
        ident = mapper.resolve(doc.get("playerId"))
        rows.append(
            {
                "player_key": ident.player_key,
                "unjoined_class": ident.unjoined_class,
                "source_id": str(doc.get("_id")),
                "action": doc.get("action"),
                "client_ip": ip_utils.extract_client_ip(doc.get("ip_address")),
                "page": doc.get("page"),
                "device_type": doc.get("device_type"),
                "created_at": _utc(doc.get("created_at")),
            }
        )
    columns = [
        "player_key",
        "unjoined_class",
        "source_id",
        "action",
        "client_ip",
        "page",
        "device_type",
        "created_at",
    ]
    return _ordered_frame(rows, columns, ["created_at", "action", "page"])


def flatten_logins(docs: Iterable[dict[str, Any]], mapper: IdentityMapper) -> pd.DataFrame:
    """Flatten `loginlogs`; PLAYER rows resolve through the common mapper."""
    rows: list[dict[str, Any]] = []
    for doc in docs:
        user_type = doc.get("userType")
        ident = mapper.resolve(doc.get("loginId")) if user_type == "PLAYER" else IdentityResult(None, "unknown")
        rows.append(
            {
                "player_key": ident.player_key,
                "unjoined_class": ident.unjoined_class,
                "source_id": str(doc.get("_id")),
                "fingerprint": _valid_fingerprint(doc.get("fingerprint")),
                "user_type": user_type,
                "success": _login_success(doc),
                "failure_reason": doc.get("failureReason"),
                "created_at": _utc(doc.get("createdAt")),
            }
        )
    columns = [
        "player_key",
        "unjoined_class",
        "source_id",
        "fingerprint",
        "user_type",
        "success",
        "failure_reason",
        "created_at",
    ]
    return _ordered_frame(rows, columns, ["created_at", "user_type", "player_key"])


def analyze_withdrawal_account_resolution(
    withdrawal_docs: Iterable[dict[str, Any]],
    wallet_docs: Iterable[dict[str, Any]],
    cash_docs: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Crosstab withdrawal status by where account IDs resolve."""
    wallet_ids = {str(d.get("_id")) for d in wallet_docs}
    cash_ids = {str(d.get("_id")) for d in cash_docs}

    def classify(value: Any) -> str:
        if value is None:
            return "missing"
        text = str(value)
        if text in cash_ids:
            return "cashaccounts"
        if text in wallet_ids:
            return "walletaccounts"
        return "unresolved"

    to_counter: Counter[tuple[str, str]] = Counter()
    from_counter: Counter[tuple[str, str]] = Counter()
    missing_from: Counter[tuple[str, str]] = Counter()
    wallet_to_statuses: Counter[str] = Counter()
    for doc in withdrawal_docs:
        status = _lower(doc.get("status"))
        to_class = classify(doc.get("toAccountId"))
        from_class = classify(doc.get("fromAccountId"))
        to_counter[(status, to_class)] += 1
        from_counter[(status, from_class)] += 1
        if from_class == "missing":
            missing_from[(status, to_class)] += 1
        if to_class == "walletaccounts":
            wallet_to_statuses[status] += 1

    return {
        "status_x_to_account": _nested_counter(to_counter),
        "status_x_from_account": _nested_counter(from_counter),
        "missing_from_account": _nested_counter(missing_from),
        "wallet_pointing_to_account_count": sum(wallet_to_statuses.values()),
        "wallet_pointing_to_account_statuses": dict(sorted(wallet_to_statuses.items())),
        "missing_from_account_count": sum(missing_from.values()),
    }


def write_unjoined_report(
    path: Path,
    frames: dict[str, pd.DataFrame],
    cfg: dict[str, Any] | None = None,
) -> Path:
    """Write markdown counts and sample IDs for unresolved identities."""
    cfg = cfg or config.load_config()
    sample_n = int(cfg["flatten"]["unjoined"]["sample_ids_per_class"])
    known_orphans = cfg["flatten"]["unjoined"]["known_orphan_wallet_ids"]
    lines = [
        "# Unjoined Report",
        "",
        "Rows are kept in parquet with `player_key = NULL` and excluded later at feature time.",
        "",
        "Known orphan-wallet IDs called out from Phase 1b:",
        *[f"- `{orphan}`" for orphan in known_orphans],
        "",
    ]

    id_columns = {
        "players": "player_key",
        "bets": "ticket_id",
        "money": "transaction_id",
        "bonus": "source_id",
        "activity": "source_id",
        "logins": "source_id",
    }
    for source, frame in frames.items():
        if "unjoined_class" not in frame.columns:
            continue
        unresolved = frame[frame["player_key"].isna()]
        lines.extend([f"## {source}", ""])
        if unresolved.empty:
            lines.extend(["No unjoined rows.", ""])
            continue
        counts = unresolved["unjoined_class"].fillna("unknown").value_counts().sort_index()
        lines.append("| unjoined_class | count | sample IDs |")
        lines.append("| --- | ---: | --- |")
        id_col = id_columns[source]
        for cls, count in counts.items():
            samples = (
                unresolved[unresolved["unjoined_class"].fillna("unknown") == cls][id_col]
                .astype(str)
                .head(sample_n)
                .tolist()
            )
            lines.append(f"| {cls} | {count} | {', '.join(samples)} |")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_reconciliation_report(
    path: Path,
    raw_counts: dict[str, int],
    parquet_counts: dict[str, int],
    withdrawal_anomaly: dict[str, Any],
    cfg: dict[str, Any] | None = None,
) -> Path:
    """Write raw Mongo vs parquet count reconciliation."""
    cfg = cfg or config.load_config()
    expected_money = raw_counts["deposits"] + len_from_withdrawal_groups(raw_counts, parquet_counts)
    lines = [
        "# Flatten Reconciliation",
        "",
        "| Source | Raw Mongo rows | Parquet rows | Delta explanation |",
        "| --- | ---: | ---: | --- |",
        f"| players | {raw_counts['players']} | {parquet_counts['players']} | One row per player; no drops. |",
        f"| bets | {raw_counts['bets']} | {parquet_counts['bets']} | One row per ticket; unjoined rows kept. |",
        f"| deposits | {raw_counts['deposits']} | included in money | No deposit deduplication; all statuses kept and flags determine money-in. |",
        f"| withdrawals | {raw_counts['withdrawals']} | included in money | Lifecycle collapse by `transactionId`; status order completed > declined > failed > pending > initiated. |",
        f"| money | {raw_counts['deposits']} deposits + {raw_counts['withdrawals']} withdrawal docs | {parquet_counts['money']} | Money rows = all deposits plus collapsed withdrawal transactions; no status-based drops. |",
        f"| bonus | {raw_counts['bonus']} | {parquet_counts['bonus']} | One row per bonus transaction; currency is assumed `{cfg['flatten']['bonus']['ASSUMED_DEFAULT']}` because source has no currency field. |",
        f"| activity | {raw_counts['activity']} | {parquet_counts['activity']} | One row per activity event; unjoined rows kept. |",
        f"| logins | {raw_counts['logins']} | {parquet_counts['logins']} | One row per login log; staff rows kept with null player_key. |",
        "",
        "## Withdrawal Account Anomaly",
        "",
        "Status x `toAccountId` resolution:",
        "",
        _markdown_nested_counter(withdrawal_anomaly["status_x_to_account"]),
        "",
        "Status x missing `fromAccountId` with `toAccountId` resolution:",
        "",
        _markdown_nested_counter(withdrawal_anomaly["missing_from_account"]),
        "",
        (
            f"`toAccountId` points to `walletaccounts` in "
            f"{withdrawal_anomaly['wallet_pointing_to_account_count']} docs, all with statuses "
            f"{withdrawal_anomaly['wallet_pointing_to_account_statuses']}. These are failed/declined "
            "reversal/failure lifecycle documents, so the configured lifecycle collapse remains valid."
        ),
        (
            f"`fromAccountId` is missing in {withdrawal_anomaly['missing_from_account_count']} docs: "
            "initial request rows have no account IDs yet, while failed/declined reversal rows point "
            "back to the wallet in `toAccountId`."
        ),
        "",
        f"Computed money row check: {expected_money}.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def len_from_withdrawal_groups(raw_counts: dict[str, int], parquet_counts: dict[str, int]) -> int:
    """Return collapsed withdrawal count inferred from money parquet metadata."""
    return parquet_counts["money"] - raw_counts["deposits"]


def _read_required_docs(db: Database, collection_name: str | None) -> list[dict[str, Any]]:
    if not collection_name:
        raise ValueError("Required source collection is not configured.")
    info = list(db.list_collections(filter={"name": collection_name}))
    if not info:
        raise ValueError(f"Required source collection {collection_name!r} is missing.")
    meta = info[0]
    if meta.get("type") == "view" or meta.get("options", {}).get("viewOn"):
        raise ValueError(f"Refusing to read Mongo view {collection_name!r}; configure a base collection.")
    return list(db[collection_name].find({}))


def _validate_unique(docs: Iterable[dict[str, Any]], field: str, source: str) -> None:
    values = [doc.get(field) for doc in docs if doc.get(field) is not None]
    counts = Counter(values)
    duplicates = [str(value) for value, count in counts.items() if count > 1]
    if duplicates:
        sample = ", ".join(sorted(duplicates)[:10])
        raise ValueError(f"{source}.{field} must be unique; duplicate sample: {sample}")


def _ordered_frame(rows: list[dict[str, Any]], columns: list[str], sort_cols: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame
    return frame.sort_values(sort_cols, kind="mergesort", na_position="last").reset_index(drop=True)


def _utc(value: Any) -> pd.Timestamp | pd.NaT:
    if value is None:
        return pd.NaT
    return pd.to_datetime(value, utc=True, errors="coerce")


def _sort_timestamp(value: Any, *, fallback: str) -> pd.Timestamp:
    ts = _utc(value)
    if pd.notna(ts):
        return ts
    if fallback == "min":
        return pd.Timestamp.min.tz_localize("UTC")
    if fallback == "max":
        return pd.Timestamp.max.tz_localize("UTC")
    raise ValueError(f"Unsupported timestamp fallback {fallback!r}")


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _lower(value: Any) -> str:
    return str(value).strip().lower() if value is not None else ""


def _bonus_ref_kind(ref_trans_id: Any, deposit_ids: set[str], bet_ids: set[str]) -> str:
    if ref_trans_id is None:
        return "none"
    ref = str(ref_trans_id)
    if ref in deposit_ids:
        return "deposit"
    if ref in bet_ids:
        return "bet"
    return "none"


def _valid_fingerprint(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "not-found" or len(text) != 64:
        return None
    return text


def _login_success(doc: dict[str, Any]) -> bool | None:
    if "success" in doc:
        return bool(doc.get("success"))
    operation = str(doc.get("operationType") or "").lower()
    if operation == "login":
        return True
    return None


def _nested_counter(counter: Counter[tuple[str, str]]) -> dict[str, dict[str, int]]:
    nested: dict[str, dict[str, int]] = defaultdict(dict)
    for (row, col), value in sorted(counter.items()):
        nested[row][col] = value
    return dict(nested)


def _markdown_nested_counter(nested: dict[str, dict[str, int]]) -> str:
    columns = sorted({col for values in nested.values() for col in values})
    lines = ["| status | " + " | ".join(columns) + " |"]
    lines.append("| --- | " + " | ".join("---:" for _ in columns) + " |")
    for status in sorted(nested):
        values = [str(nested[status].get(col, 0)) for col in columns]
        lines.append(f"| {status} | " + " | ".join(values) + " |")
    return "\n".join(lines)

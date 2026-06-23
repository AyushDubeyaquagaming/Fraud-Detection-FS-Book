# Phase 3 Feature Output Contract

Feature builders read only through `frauddet.snapshot.load_snapshot`.

`data/player_features.parquet` is wide with one row per canonical
`player_key`. Each feature has:

- `<feature_name>`: scalar value.
- `<feature_name>__null_reason`: null-contract reason or null.
- `<feature_name>__strength`: `strong`, `moderate`, `weak`, or `context_only`.
- `<feature_name>__scoring_role`: `scoring`, `supporting`, or `context_only`.

`data/player_features_evidence.parquet` is long with one row per
`player_key + feature_name` and a `feature_evidence` JSON string. Multi-account
evidence contains direct linked player keys and shared keys. Hashes remain
hashes; no raw NIN/email is stored. Recipient numbers may be reviewer-visible.

All linkage is one-hop. No connected-component or transitive expansion is
performed.

Device fingerprints whose player cardinality exceeds
`thresholds.ma_device_max_cardinality` are treated as shared/public/office
devices. They remain visible to `ma_device_count` as context, but cannot create
device-sharing links or corroborate referral and co-creation features.

`build_phase3_features` loads each frozen input once and writes the reviewed
`ma_`, `pay_`, and `bet_` groups together. The money-facing groups receive the same
`WithdrawalContext`, so completed-withdrawal filtering and recipient linkage
have one implementation. Turnover-dependent payment features are null with
`casino_activity_not_observable` when a player has no observable sportsbook
activity; v1 cannot infer zero play from absent casino telemetry.

Betting features are computed from `bets.parquet` only, with per-`game_type`
evidence so casino can slot in later. Dev betting anomalies are expected to be
mostly dormant because the statistical features are hard-gated on placeholder
minimum-volume thresholds pending production calibration.

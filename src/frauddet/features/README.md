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

# Feature Dictionary — Phase 3 (Player-Level Fraud Features)

**Status:** design spec for Phase 3 build. Every threshold here is a
`# PLACEHOLDER — recalibrate on production data`. Dev numbers are
statistically meaningless (n≈116 synthetic players); this phase builds the
**machinery**, not calibrated detection.

**Reads from:** the frozen snapshot only —
`data/snapshot_phase3_2026-06-19b/` (`config.yaml` `phase3.input_dir`).
Never live Mongo.

**Produces:**
- `player_features.parquet` — one row per player, all feature scalars + metadata.
- `feature_evidence` (companion store, e.g. `player_features_evidence.parquet`
  or a JSON sidecar keyed by `player_key` + `feature_name`) — the
  reviewer-facing transaction-level evidence behind each feature.

Only players with a non-null `player_key` enter the feature table. Unjoined
rows (`pre_registration` / `test_pattern` / `unknown` / `staff`) are excluded,
per the Phase 2 identity policy.

---

## 1. Design principles (read before implementing)

1. **Scalar + evidence.** Every feature emits a scalar the scorer can threshold
   AND reviewer-facing evidence (transaction IDs, pairs, recipient refs, or
   reason-text for aggregates). A feature that produces only a number a reviewer
   can't act on is a bug.

2. **The null contract (most important rule).**
   - `null` means **"cannot safely measure"** — the scorer must NOT treat it as
     `0`. It carries a `feature_null_reason`.
   - `0` means **"measured, observed none."**
   - These score differently. Conflating them produces false accusations
     (treating "we couldn't see their play" as "they didn't play").

3. **Three sub-scores → one overall.** Features roll up into three independent
   sub-scores; the sub-scores sum to `overall_risk`. Reviewers see *why* per
   category, not just one opaque number.
   - `multi_accounting_score` ← `ma_` features
   - `payment_fraud_score` ← `pay_` features
   - `betting_anomaly_score` ← `bet_` features

4. **Combination over isolation.** The dominant false positive across the whole
   system is the **legitimate lucky/skilled/high-volume player**. Most strong
   features are designed to fire only in *combination* (e.g. high
   withdrawal/deposit ratio is innocent alone; ratio + low turnover + fast exit
   is not). The scoring layer (Phase 4) owns combination logic; this dictionary
   marks which features are combination-gated.

5. **Game-context-aware betting.** Betting features are computed per
   `game_type`. Today that is ~100% Sports-book; the structure exists so casino
   betting slots into the same logic when per-round logging arrives, without
   reworking sportsbook.

6. **Built ≠ active on dev.** Many strong features (shared-NIN, win-rate,
   deposit-then-exit for casino-touching players) will be null/zero on dev
   because the signal needs production volume or casino logging. That is
   correct, not broken. Each such feature is flagged "dormant on dev."

---

## 2. Standard feature output schema

Every feature emits exactly these fields (scalar row in `player_features`,
evidence in the companion store):

| field | type | meaning |
|---|---|---|
| `feature_value` | float / int / bool / null | the scalar the scorer reads |
| `feature_evidence` | list / struct / reason-text | reviewer-facing support |
| `feature_null_reason` | enum / null | why value is null (see below); null when value is present |
| `feature_strength` | enum | `strong` / `moderate` / `weak` / `context_only` |
| `feature_scoring_role` | enum | `scoring` / `supporting` / `context_only` |

**`feature_null_reason` vocabulary** (extensible; defined per feature):
`no_completed_deposits`, `no_completed_withdrawals`,
`no_sportsbook_bets_observed`, `casino_activity_not_observable`,
`insufficient_deposit_denominator`, `insufficient_settled_bets`,
`insufficient_bets_for_timing`, `insufficient_bets_for_volatility`,
`no_withdrawals`, `no_bets`.

`scoring` = drives flags directly. `supporting` = contributes only in
combination with a scoring feature. `context_only` = surfaced in the case file
for reviewer context, **zero scoring weight in v1**.

---

## 3. Multi-accounting group (`ma_`)

**Sub-score:** `multi_accounting_score`. **Top business priority. Strongest
group on dev** (structural signals survive low volume). Relational by nature:
build linkage maps across the population first, then roll up to per-player
counts. **One-hop (direct) sharing only — no transitive graph components in v1**
(a single shared office/library machine would merge hundreds of unrelated
people into one fake ring).

Evidence convention for `ma_`: the list of **other `player_key`s** sharing the
attribute, plus a reference to the shared key (hashed identifiers shown as hash,
never raw).

### Tier 1 — Strong (built now; some dormant on dev)

#### `ma_nin_shared_account_count` — strong — **dormant on dev**
- **Definition:** count of other distinct players sharing this player's
  `nin_hash` (from `players.parquet`).
- **Value:** int (0 when no share). **Evidence:** linked `player_key`s + the
  shared `nin_hash`.
- **Hypothesis:** same national ID across accounts = same person. Near-proof of
  multi-accounting — the strongest identity signal that exists.
- **FP traps:** almost none; possible data-entry duplication, or fraudulent
  family-ID reuse.
- **Coverage:** only players with a non-null `nin_hash` (~70% on dev).
  **Null:** none needed — absent NIN → no share → `0` is honest (a player with
  no NIN simply has no NIN linkage).
- **Dev note:** 0 shares in snapshot (synthetic accounts each have a unique
  fake NIN). Activates in production. Mark `# dev signal expected zero`.
- **Scoring role:** scoring.

#### `ma_email_shared_account_count` — strong — **live on dev**
- **Definition:** count of other players sharing this player's `email_hash`.
- **Value:** int. **Evidence:** linked `player_key`s + shared `email_hash`.
- **Hypothesis:** shared email = same operator; common in bonus-farming rings.
- **FP traps:** low; a shared family email.
- **Coverage:** ~25% on dev (email_hash non-null). **Null:** none; absent → 0.
- **Dev note:** snapshot has **one** email_hash shared by 2 players — the only
  live identity-linkage signal in dev. Proves the path works end to end.
- **Scoring role:** scoring.

#### `ma_device_shared_account_count` — strong
- **Definition:** count of distinct other players who share ≥1 valid fingerprint
  with this player (`logins.parquet`, `user_type=PLAYER`, non-null 64-hex
  fingerprint).
- **Value:** int. **Evidence:** linked `player_key`s + the shared
  fingerprint(s).
- **Hypothesis:** one physical device operating multiple accounts.
- **FP traps:** shared family device; public/library/office machines (the dev
  office-machine artifact — one hash across ~28 players is noise).
- **Coverage:** players with ≥1 valid-fingerprint login (~most; ~60% have a
  single stable fingerprint). **Null:** `null` + `no_valid_fingerprint_logins`
  if the player never logged in with a valid fingerprint.
- **Dev note:** sharing counts are office artifacts → calibration is production.
- **Scoring role:** scoring.

#### `ma_withdrawal_recipient_shared_count` — strong — *shared with `pay_`*
- **Definition:** for this player's withdrawal `recipient_normalized`
  number(s), count of other distinct players who withdraw to the same number
  (`money.parquet`).
- **Value:** int. **Evidence:** the shared recipient number(s) + linked
  `player_key`s + the `withdrawal_id`s.
- **Hypothesis:** multiple accounts cashing out to one phone = mule funnel /
  collusion paying one beneficiary.
- **FP traps:** withdrawing to a family member's number — rarer than the signal.
- **Coverage:** players with ≥1 completed withdrawal. **Null:** `null` +
  `no_completed_withdrawals` otherwise.
- **Note:** this is the **same linkage** as `pay_third_party_withdrawal_*` —
  computed once, surfaced in BOTH `multi_accounting_score` and
  `payment_fraud_score` by design (a shared recipient is simultaneously a
  multi-account link and a payment-fraud signal).
- **Scoring role:** scoring.

#### `ma_identity_phone_collision_count` — strong (when it fires) — **dormant on dev**
- **Definition:** number of *other* player records whose normalized phone
  resolves to the same value as this player's (from the identity mapper's
  collision detection).
- **Value:** int. **Evidence:** the colliding `player_key`s + the shared phone.
- **Hypothesis:** the identity system itself detected that one phone maps to
  multiple player records — strong multi-account evidence when it occurs.
- **FP traps:** legitimate re-registration (open backend question: does
  re-registration soft-delete the prior player, or can two live records share a
  phone?).
- **Coverage:** all players. **Null:** none; `0` when no collision.
- **Dev note:** snapshot has **zero** collisions (the historical `751452653`→3
  case is not in this frozen snapshot). Activates in production.
- **Scoring role:** scoring.

### Tier 2 — Moderate / supporting

#### `ma_referred_by_linked_account` — strong-when-fires, **supporting** — **dormant on dev**
- **Definition:** boolean — was this player referred by an account they ALSO
  share a device, NIN, email, or withdrawal recipient with (cross of
  `referred_by_key` with the Tier-1 linkages)?
- **Value:** bool. **Evidence:** the referrer `player_key` + which linkage
  corroborated it.
- **Hypothesis:** the self-referral smoking gun — referring your own second
  account to farm the referral bonus.
- **FP traps:** a relative referring you while you share a home device.
- **Coverage:** needs a referral AND a corroborating link. **Null:** `0`/False
  when no referral; the value is only meaningful with corroboration.
- **Dev note:** ~2 non-null referrals in snapshot → effectively dormant.
- **Scoring role:** supporting (the *corroboration* is what makes it score;
  referral count alone is not suspicious — the bonus program rewards referrals).

#### `ma_referral_fanout_count` — weak/supporting
- **Definition:** count of accounts directly referred by this player (one hop).
- **Value:** int. **Evidence:** referred `player_key`s.
- **Hypothesis:** large fan-out can indicate self-referral bonus farming.
- **FP traps:** **legitimately successful referrers — the program rewards
  this.** Fan-out alone is not suspicious; only fan-out + shared identity/device
  among the referred accounts is. Pairs with `ma_referred_by_linked_account`.
- **Coverage:** all players (`0` typical). **Dev note:** dormant.
- **Scoring role:** supporting.

#### `ma_cocreated_linked_count` — moderate
- **Definition:** count of other players created within a placeholder window
  (e.g. 15 min) of this player's `created_at` AND sharing a device / recipient /
  NIN / email.
- **Value:** int. **Evidence:** the co-created linked `player_key`s + shared
  attribute + creation timestamps.
- **Hypothesis:** scripted batch account creation.
- **FP traps:** marketing-campaign signup spikes produce many unrelated
  registrations at once — which is exactly why a **shared signal is required**,
  not time alone.
- **Coverage:** all players. **Null:** `0` when none.
- **Scoring role:** supporting.

### Tier 3 — Weak / context-only

#### `ma_device_count` — weak / context_only
- Distinct valid fingerprints the player logs in from. Hypothesis: device
  cycling to evade. FP: ordinary multi-device users (phone+laptop+tablet).
  Evidence: the fingerprints. Context only.

### Tier 4 — Deferred (do not build in v1)

- **IP-sharing features** (`ma_ip_shared_account_count`,
  `ma_ip_max_cluster_size`) — **deferred.** Dev IP data is entirely localhost +
  one office IP: no genuine signal to validate against, only known noise. By the
  "sound mechanism + valuable-when-fired" test, IP sharing is sound-mechanism
  but low-value-and-noisy — carrier NAT alone fronts thousands of unrelated
  users behind one IP. Build in production with an infra-exclusion list and a
  high-cardinality cap, not now.
- **`ma_bank_account_shared_count`** — deferred; `bankDetails` empty in data.
- **`ma_passport_shared_count`** — deferred; `passportNumber` default-polluted
  (values held by >5% of players are stubs, not identities).

---

## 4. Payment-fraud group (`pay_`)

**Sub-score:** `payment_fraud_score`. **Strong but partly casino-blocked.** The
group's spine is the pattern: **deposit in → quick withdrawal out → little/no
measured play → suspicious recipient/manual-payment context.** Reads
`money.parquet` (+ `bets.parquet` for turnover). Evidence = `transaction_id`s,
deposit-withdrawal pairs, recipient numbers.

**bets↔money boundary:** turnover features cross bets↔money; valid only because
Phase 2 relabeled bets to UGX. Each is flagged ⚠️.

### Tier 1 — Strong V1 scoring features

#### `pay_deposit_then_exit_flag` — strong — **the flagship**
- **Definition:** boolean — exists a completed deposit D followed within window
  Y of a completed withdrawal W of ≥ X% of D, with stake between
  `D.finalized_at` and `W.requested_at` < Z% of D? (Y, X, Z placeholders.)
- **Value:** bool / null. **Evidence:** the `(deposit_id, withdrawal_id)` pair,
  % withdrawn, and intervening `bet_id`s (empty list if none) — the reviewer's
  whole case in one row.
- **Hypothesis:** money in → out → ~no play: the defining laundering /
  passthrough signature; the strongest single explainable payment-fraud pattern.
- **FP traps:** a fast legitimate cash-out — but the betting condition guards
  it; someone who played won't satisfy "stake < Z%". Residual: deposit → decide
  not to play → withdraw (innocent, rare, worth a glance).
- **Coverage:** ≥1 completed deposit AND ≥1 subsequent completed withdrawal.
- **Null (three-way turnover logic — critical):**
  - sportsbook bets exist in window → evaluate the betting condition normally.
  - **no bets of any kind AND no casino footprint observable** → conceptually
    "zero play confirmed" → flag may fire. **In v1 this branch is UNREACHABLE**
    (casino telemetry doesn't tie to players), so it collapses to null. Code is
    structured so casino logging later unlocks it.
  - no sportsbook bets + casino activity exists but unvaluable → `null` +
    `casino_activity_not_observable`.
  - **Net v1 behavior:** casino-touching players → `null` +
    `casino_activity_not_observable`, NOT `False`. We never emit a confident
    `False` when we couldn't see the play.
- **Dev note:** largely dormant for casino-touching players until casino logging
  lands or sportsbook-only fraud appears.
- **Scoring role:** scoring. Highest weight in the group.

#### `pay_intervening_turnover_ratio` — strong — the "did they actually play" measure
- **Definition:** ⚠️ for the triggering deposit→withdrawal pair, stake between
  D and W ÷ D amount.
- **Value:** float ≥ 0 / null. **Evidence:** the `(deposit_id, withdrawal_id)`
  pair + intervening `bet_id`s + the stake sum.
- **Hypothesis:** low ratio = deposited but barely staked before cashing out;
  this is what converts a high withdrawal/deposit ratio from "lucky winner" into
  "passthrough."
- **FP traps:** none standalone — it's a *disambiguator* for other features.
- **Coverage / null:** the three-way logic above (same as the flagship). v1: no
  sportsbook bets → `null` + `casino_activity_not_observable`. A genuine
  `0.0` (sportsbook-active but staked nothing in the window) is a real,
  scoreable value.
- **Scoring role:** scoring, primarily in combination (it gates the flagship and
  reshapes the ratio's meaning).

#### `pay_min_minutes_deposit_to_withdrawal` — strong
- **Definition:** smallest gap between any completed deposit's `finalized_at`
  and a subsequent completed withdrawal's `requested_at`.
- **Value:** float minutes / null. **Evidence:** the `(deposit_id,
  withdrawal_id)` pair producing the minimum.
- **Hypothesis:** minutes-level in→out doesn't fit normal play.
- **FP traps:** fast legitimate winning cash-out → pairs with turnover; fast
  alone is suspicious, fast + no-play is damning. Never the sole flag.
- **Coverage / null:** ≥1 completed deposit and ≥1 subsequent completed
  withdrawal; else `null` + `no_completed_deposits` / `no_completed_withdrawals`.
- **Scoring role:** scoring, in combination.

#### `pay_third_party_withdrawal_flag` / `pay_third_party_withdrawal_count` — strong — *shared with `ma_`*
- **Definition:** did the player ever complete a withdrawal to
  `recipient_normalized` ≠ their own number; how many. Rolls up the
  `is_third_party_recipient` flag in `money.parquet`.
- **Value:** bool / int. **Evidence:** list of `(withdrawal_id,
  recipient_number)` — recipient shown so the reviewer sees where money went.
- **Hypothesis:** cashing out to someone else's mobile-money number = muling,
  account-selling, collusion payout.
- **FP traps:** paying a family member or own second number — real but rarer.
- **Coverage / null:** ≥1 completed withdrawal; else `null` +
  `no_completed_withdrawals`.
- **Note:** same linkage as `ma_withdrawal_recipient_shared_count` — one
  computation, feeds both sub-scores.
- **Scoring role:** scoring.

### Tier 2 — Moderate supporting features

#### `pay_withdrawal_to_deposit_ratio` — moderate, context-leaning — **never fires alone**
- **Definition:** sum(completed withdrawal amount) ÷ sum(completed deposit
  amount), all UGX (uses `is_money_in` / `is_money_out`).
- **Value:** float / null. **Evidence:** reason-text stating both sums
  (aggregate, not transaction-specific).
- **Hypothesis:** withdrawing materially more than deposited can indicate a
  cash-out endpoint.
- **FP traps:** **the lucky legitimate winner — the system's single biggest
  FP.** Therefore `context_only` UNLESS combined with low intervening turnover,
  fast exit, third-party recipient, or manual-recon exposure. The scorer MUST
  require a co-signal.
- **Coverage / null:** needs ≥1 completed deposit as denominator. **Denominator
  gate:** if total completed deposits < placeholder floor → `null` +
  `insufficient_deposit_denominator` (prevents deposit-2k-withdraw-50k and
  divide-by-near-zero distortions — independent of the never-alone rule).
- **Scoring role:** supporting (scores only in combination; zero weight alone).

#### `pay_fast_withdrawal_count` — moderate
- **Definition:** count of completed withdrawals within window Y of a preceding
  completed deposit.
- **Value:** int. **Evidence:** list of `(withdrawal_id, matched_deposit_id)`.
- **Hypothesis:** repeated fast cycling, not a one-off.
- **FP traps:** as min-minutes; gate on count ≥ N.
- **Coverage / null:** needs both sides; `null` + reason otherwise.
- **Scoring role:** supporting.

#### `pay_manual_reconciliation_count` / `pay_manual_reconciliation_ratio` — moderate, platform-specific
- **Definition:** count and share of the player's money-in deposits with
  `final_status == manual_reconciliation`.
- **Value:** int / float. **Evidence:** list of those `deposit_id`s.
- **Hypothesis:** manual reconciliation bypasses the gateway (admin hand-credits
  the wallet). An elevated share may indicate process exploitation or insider
  collusion — a signal available *because of how this platform works*.
- **FP traps:** genuine gateway failures hit honest players; only an elevated
  *share with volume* is meaningful. Small counts are noise.
- **Coverage / null:** any player with deposits; most `0` (real zero).
- **Scoring role:** supporting.

#### `pay_declined_withdrawal_count` — moderate
- **Definition:** count of withdrawals with `final_status == declined`
  (human/admin refusal — kept distinct from `failed` in the flatten layer).
- **Value:** int. **Evidence:** list of `withdrawal_id`s.
- **Hypothesis:** an admin actively refusing payouts is stronger than a
  technical failure — possibly funds already flagged.
- **FP traps:** legitimate limit/verification declines.
- **Coverage / null:** ≥1 withdrawal; else `null` + `no_withdrawals`.
- **Scoring role:** supporting.

### Tier 3 — Weak / context-only (zero scoring weight in v1)

- **`pay_failed_withdrawal_count`** — weak — technical `failed` withdrawals.
  Evidence: ids. Noisier than `declined`. Context.
- **`pay_net_money_flow`** — weak — sum(money_in) − sum(money_out), UGX.
  Lucky winners dominate it. Context for the ratio.
- **`pay_distinct_payment_methods`** / **`pay_distinct_payment_accounts`** —
  weak — instrument diversity. Plausibly stolen-instrument testing, but honest
  users have several. Evidence: the methods/accounts. Context.

### Tier 4 — Deferred

- **`pay_deposit_amount_structuring`** — deferred. Sub-threshold clustering is
  the clearest IP-equivalent in this group: plausible mechanism, very noisy,
  needs real volume + real thresholds. Build/calibrate in production.
- **Bank-account payment linkage** (`pay_bank_*`) — deferred until `bankDetails`
  populated.

---

## 5. Betting-anomaly group (`bet_`)

**Sub-score:** `betting_anomaly_score`. **Weakest group on dev, by design** —
betting-anomaly features are *statistical* (they need population distributions
to define "unusual"), and n≈116 / ~440 bets has no meaningful distribution. We
build the machinery, **hard-gate every feature on minimum volume**, and expect
most to lie dormant until production. **Sportsbook-only** by data availability;
all features computed per `game_type` so casino slots in later.
Reads `bets.parquet`. Evidence = `ticket_id`s.

### Tier 1 — Strong-when-measurable (volume-gated; dormant on dev)

#### `bet_win_rate_vs_volume` — strong (prod), **dormant (dev)**
- **Definition:** settled-bet win rate (`result==WIN` ÷ settled bets),
  **only when settled-bet count ≥ placeholder floor** (e.g. 30); below → null.
- **Value:** float [0,1] / null. **Evidence:** wins/settled counts + winning
  `ticket_id`s + reason-text with the rate.
- **Hypothesis:** a win rate implausibly high *over real volume* suggests
  insider info, account-selling of "winning" accounts, or exploitation. The name
  encodes the gate — rate WITH volume is what matters.
- **FP traps:** **the lucky / small-sample / skilled winner — dominant FP.**
  Never standalone; "high rate + high volume + maybe fast cash-out." The volume
  gate is the primary defense.
- **Coverage / null:** `null` + `insufficient_settled_bets` below floor (most
  dev players). Low rate is a real measurement.
- **Scoring role:** scoring, in combination, production-calibrated.

#### `bet_timing_regularity` — strong-when-measurable — the bot signal
- **Definition:** regularity of inter-bet gaps (e.g. coefficient of variation of
  seconds between consecutive bets; low CV = robotically even). Only when bet
  count ≥ floor (e.g. 20).
- **Value:** float (low = suspicious) / null. **Evidence:** the ordered
  `ticket_id`s + timestamps so the reviewer sees the cadence.
- **Hypothesis:** humans bet irregularly; near-constant intervals suggest
  automation.
- **FP traps:** a disciplined human; live-event betting at regular breaks. A
  hint, not proof.
- **Coverage / null:** `null` + `insufficient_bets_for_timing` below floor.
- **Scoring role:** scoring, in combination.

### Tier 2 — Moderate supporting

#### `bet_stake_volatility` — moderate
- **Definition:** stake dispersion (std/mean of stake), gated on a small minimum
  bet count.
- **Value:** float / null. **Evidence:** stake summary + extreme `ticket_id`s.
- **Hypothesis:** wildly erratic stakes → bonus-hunting / limit-testing /
  chasing; very uniform stakes → scripted betting.
- **FP traps:** normal bankroll variation; noisy both directions.
- **Coverage / null:** `null` + `insufficient_bets_for_volatility` below floor.
- **Scoring role:** supporting.

#### `bet_bonus_funded_stake_share` — moderate — *cross-links to `pay_`*
- **Definition:** share of stake that is bonus-funded (`stake_bonus` ÷ total
  stake) + free-bet count (fields already in `bets.parquet`).
- **Value:** float [0,1]. **Evidence:** the bonus-staked `ticket_id`s.
- **Hypothesis:** play almost entirely bonus-funded, especially with a fast
  cash-out, is the bonus-abuse pattern — links betting to the payment sub-score
  (bonus-funded play + deposit-then-exit).
- **FP traps:** legitimate welcome-bonus use. Meaningful mainly in combination.
- **Coverage / null:** any player with bets; `0` is real (no bonus play).
- **Scoring role:** supporting.

#### `bet_game_type_concentration` — moderate, structural — **the casino hook**
- **Definition:** per-`game_type` stake share (today ~100% Sports-book),
  computed as a distribution.
- **Value:** share in the dominant game_type (+ per-type vector in evidence).
  **Evidence:** the per-game_type breakdown.
- **Hypothesis:** *today* a structural/coverage marker — it identifies who is
  sportsbook-active vs not, which **directly feeds the `pay_` casino-null
  logic** (whether turnover is measurable). *Later*, concentration in a specific
  exploitable game becomes a real exploitation signal.
- **FP traps:** none today (descriptive).
- **Coverage / null:** any player with bets.
- **Scoring role:** `context_only` now; **promotes to supporting when casino
  data exists.** This single feature is what makes casino integration a switch,
  not a rebuild.

### Tier 3 — Weak / context-only

- **`bet_avg_odds` / `bet_odds_profile`** — weak — mean/spread of `total_odds`.
  Longshot-only or always-min-odds patterns can hint at strategies/arbitrage;
  noisy. Context.
- **`bet_count` / `bet_active_days`** — weak/context — raw volume and span; not
  signals themselves, they are the denominators the gates use, surfaced for
  context.
- **`bet_void_rate`** — weak — share of VOID bets. Usually legitimate (cancelled
  events). Context.

### Tier 4 — Deferred until casino logging

- **All casino game-exploitation features** — deferred by the product scope
  decision (no per-round logging). The per-`game_type` structure +
  `bet_game_type_concentration` are the only things built now so casino betting
  features drop into this group later without reworking sportsbook.

---

## 6. Lifecycle / gating features (shared, computed once)

Not a sub-score; these support FP-control and combination logic across groups.

- `account_age_days` — `now(snapshot_date) − players.created_at`. Used to gate
  new-player instability (a 90%-win-rate over 3 bets on a 1-day-old account is
  noise).
- `first_deposit_at`, `first_bet_at`, `first_withdrawal_at`, and the minutes
  between account-creation → first-deposit → first-bet → first-withdrawal.
  Lifecycle velocity context for the payment flagship.
- `kyc_status` (from `players.parquet`).
- `n_completed_deposits`, `n_completed_withdrawals`, `n_settled_bets`,
  `n_bets`, `n_active_days` — the denominators the gates reference (also
  surfaced as `bet_count` etc. for reviewers).

---

## 7. Sub-score → overall-risk rollup (Phase 4 owns the math; placeholders here)

```
multi_accounting_score = weighted sum of ma_ scoring/supporting features
payment_fraud_score    = weighted sum of pay_ scoring/supporting features
betting_anomaly_score  = weighted sum of bet_ scoring/supporting features
overall_risk           = multi_accounting_score + payment_fraud_score
                         + betting_anomaly_score
```

- Each feature contributes points **only when its rule fires**; `supporting`
  features contribute **only when a `scoring` feature in the same group also
  fires** (combination logic).
- `context_only` features contribute **0** to any score — present in the case
  file only.
- `null` features contribute **0** and are shown to the reviewer as
  "not measurable (reason)", never as "clean".
- Every point/weight/band cut is a `# PLACEHOLDER` in `config.yaml` →
  `thresholds:` (currently empty). **Nothing is tuned on dev.**

---

## 8. config.yaml placeholder inventory (all PLACEHOLDER)

Add under `thresholds:` (or a `features:` block). None used until Phase 4.

**Payment:**
`pay_fast_window_minutes` (Y), `pay_pct_withdrawn_threshold` (X),
`pay_intervening_turnover_pct` (Z), `pay_min_deposit_denominator`,
`pay_fast_withdrawal_count_gate` (N), `pay_manual_recon_ratio_threshold`,
`pay_manual_recon_min_count`.

**Betting:**
`bet_min_settled_bets_for_winrate`, `bet_min_bets_for_timing`,
`bet_min_bets_for_volatility`, `bet_timing_cv_threshold`,
`bet_win_rate_threshold`.

**Multi-accounting:**
`ma_cocreation_window_minutes`, `ma_referral_fanout_threshold`, and (deferred,
for prod) `ma_ip_infra_exclusion_list`, `ma_ip_max_cardinality_cap`.

**Gating:** `min_account_age_days_for_betting_flags`,
plus per-feature minimum-activity gates referenced above.

**Scoring weights / bands:** `weights.<feature_name>` per feature;
`bands.low/medium/high` cut points. All placeholder.

---

## 9. What is deliberately NOT built in v1 (and why)

| Deferred | Reason |
|---|---|
| IP-sharing (`ma_ip_*`) | Dev IP = localhost + one office IP; no signal, only noise; carrier NAT makes it low-value in prod too. Build with infra-exclusion + cardinality cap in prod. |
| Bank linkage (`ma_bank_*`, `pay_bank_*`) | `bankDetails` empty in data. |
| Passport linkage (`ma_passport_*`) | `passportNumber` default-polluted (stub values held by >5% of players). |
| Deposit structuring (`pay_deposit_amount_structuring`) | Very noisy; needs prod volume + real thresholds. |
| Casino game-exploitation (all) | No per-round logging (product scope decision). Hook (`bet_game_type_concentration` + per-game_type structure) built so it's a future switch. |
| Transitive identity clustering | One shared machine merges unrelated people into a fake ring. One-hop only in v1. |

---

## 10. Honest summary

- **Multi-accounting** is the strongest group on dev (structural signals);
  email linkage actually fires; NIN and phone-collision are built but dormant
  until production volume.
- **Payment fraud** is strong but its flagship (`deposit_then_exit`) is largely
  dormant for casino-touching players until casino logging lands — by design, it
  returns `null` rather than a false `False`.
- **Betting anomaly** is mostly dormant on dev (statistical features need real
  distributions); built, hard-gated, and game-context-aware for casino.
- The recurring dominant false positive is the **legitimate lucky / skilled /
  high-volume player**, which is why nearly every strong feature is
  combination-gated and volume-gated.
- The system is built to **light up progressively** as production data and
  casino logging arrive — not to pretend it detects what it cannot yet see.
# GTA 6 Trophy Rarity Forecast — V2 (refresh-aware)

[![bounty](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?ref=5548afh3)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Dilslino/gta6-trophy-rarity-forecast-01-entry-v2/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?ref=5548afh3)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Dilslino/gta6-trophy-rarity-forecast-01-entry-v2/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?ref=5548afh3)

Forecast each GTA VI (PS5) trophy's earn rate on Exophase at the case target.
Bounty: <https://make.fun/bounty/gta6-trophy-rarity-forecast-01?ref=5548afh3>

## The model (V2: refresh-aware)

V1 lost because it treated the current earned/tracked ratio as static.
Exophase does not update continuously: on launch night the page sat
unchanged at 638 players tracked for hours, then refreshed to 3,868 tracked
with every rate but the 100% one lower. The tracked base explodes at a
refresh while earned counts grow slowly, so the ratio **falls**. V2 forecasts
the two counts separately and divides them:

```
rate_i(target) = E_i(target) / N(target)
```

- **Separate counts.** `E_i(t) = rate_i(t) / 100 * N(t)` is recovered from
  every history read (the recovered counts sum to the page's
  `trophies_earned`, so this is exact up to 2-decimal rounding).
- **Log-linear growth.** Least-squares slopes of `log N` and `log E_i`
  against time; each trophy's earned slope is shrunk 65% toward its
  tier-group slope (launch-night earners number 4–13 per trophy, so a
  trophy's own slope is mostly noise).
- **Saturating horizons.** Effective horizon `tau * (1 - exp(-dt / tau))`
  with `tau_N = 36 h`, `tau_E = 72 h`: a 90-minute horizon uses the full
  slope, a week-long one does not extrapolate the launch-night explosion
  forever.
- **Flat history = a refresh is coming.** When every read is identical,
  the observed slopes are 0 — but the next refresh multiplies the tracked
  base by ~6.06 (3868/638, the first refresh the house saw) while earned
  counts grow only ~20%. V2 applies that jump (scaled in log space by the
  horizon). When the history already contains refreshes, the observed
  slopes carry them and no jump is added.
- **Group ceiling cap.** Median capped by tier (platinum 20%, gold 40%,
  silver 60%, bronze 90%): a logit-linear fit with no ceiling can
  overshoot the median story.
- **Anomaly.** "Just Like Paradise" (id 2) prints 100% earned on the
  launch-night page; it cannot survive a refresh. Its median is pulled to
  15% with a wide spread (sigma 1.9 logit).
- **Spread.** `sigma = 0.25 + 0.45*sqrt(horizon_days) + refresh
  uncertainty`, clamped to [0.4, 2.0] logit; quantiles are the median
  shifted by the normal z-scores in logit space.

## Validation (local, run before submission)

- Practice p00/p01: valid, deterministic, ~0.1 s/case, ~2.8 KB per output.
- Practice loss under refresh truth (refreshed page: N 638 → 3868,
  earned ×1.0–1.5): **V2 0.25–0.29** vs starter 1.47–1.88 (V1 static
  scores the same as the starter here). Typical miss 0.06–0.22 pp,
  coverage_90 1.00.
- Monte Carlo, 40 draws per regime: **fast tracked-explosion (×6 in
  90 min): V2 0.26 vs V1 1.75, V2 wins 40/40**; final-like week:
  V2 0.44 vs V1 0.44 (21/40). Slow-refresh / no-refresh regimes favour
  the static model by construction — V2 bets the refresh happens, which
  the rules state it does for the scored practice cases; the final
  case's history is non-flat, so the jump does not fire there.

## Run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Stdlib only, no network, deterministic (same input → byte-identical
output). Reads `$MAKEFUN_INPUT_DIR` (else `/input`), writes
`$MAKEFUN_OUTPUT_DIR` (else `/output`), one `<id>.out` per `<id>.in`.

## Gates (verified locally)

- Runs twice byte-identical as uid 65534 with `env -i`.
- Never deletes the output directory (or anything in it): a sentinel
  file placed in the output dir survives a run; `main.py` contains no
  delete call.
- p00/p01 valid (55 trophies each, five non-decreasing quantiles in
  0–100, exactly one `trophies` key), ~0.08 s/case, 2,791 bytes each.

## License

MIT — see `LICENSE`.

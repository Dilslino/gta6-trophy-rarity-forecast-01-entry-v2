# SPDX-License-Identifier: MIT
"""gta6-trophy-rarity-forecast-01 entry V2: refresh-aware forecast.

Model (the whole point of V2 is the refresh):
  Exophase does NOT update its counts continuously. It shows the rates as of
  its last refresh, then jumps: on launch night the page sat unchanged at 638
  players tracked for hours and then refreshed to 3,868 tracked, with every
  rate but the 100% one lower. A model that fits the current earned/tracked
  ratio as if it were static is therefore wrong in exactly the direction the
  refresh moves: the tracked base explodes, earned counts grow much more
  slowly, so the earned/tracked ratio FALLS. V2 forecasts the two counts
  separately and divides them:

    rate_i(target) = E_i(target) / N(target)

  * N(t): players tracked. E_i(t): players who earned trophy i, recovered
    from the page as rate_i(t) / 100 * N(t) (the page's trophies_earned total
    equals the sum of the recovered counts, so this is exact up to the
    page's 2-decimal rounding).
  * Each count is forecast in log space: log-linear least-squares slopes of
    log N and log E_i against time over the case history, with each trophy's
    earned slope shrunk 65% toward its tier-group slope (single trophies
    have 4-13 earners at launch; their own slopes are mostly noise).
  * Growth saturates: the effective horizon is tau * (1 - exp(-dt / tau))
    with tau_N = 36 h and tau_E = 72 h, so a 90-minute horizon uses the
    full slope while a week-long horizon does not extrapolate a launch-night
    explosion forever (tracked-base growth saturates faster than earned
    growth: new players arrive in a burst, story progress keeps coming).
  * Flat history = a refresh is coming, not "nothing changes". When every
    read in the history is identical (page unchanged), the observed slopes
    are 0, but the next refresh multiplies the tracked base by ~6.06
    (3868 / 638, the first refresh the house saw) while earned counts grow
    only ~20% (the newly tracked cohort has earned almost nothing yet).
    V2 applies that one refresh jump, scaled in log space by
    min(1, horizon_hours / 1.5), plus slow prior continuous growth for
    long horizons. When the history already contains refreshes, the
    observed slopes carry them and no jump is added.
  * Group ceiling cap: the median is capped by tier (platinum 20%, gold
    40%, silver 60%, bronze 90%). A logit-linear fit with no ceiling can
    overshoot the median story badly (earned counts extrapolated past the
    tracked base); the cap keeps the forecast inside what a tier can reach.
  * Anomaly: one trophy ("Just Like Paradise", id 2) prints 100% earned
    in the launch-night page. That cannot survive a refresh (the new cohort
    has not fished): its median is pulled to 15% with a wide spread
    (sigma 1.9 logit) instead of trusting the 100%.
  * Spread: sigma_logit = 0.25 + 0.45 * sqrt(horizon_days) + refresh
    uncertainty (0.85 when the history is flat, else 0.35) + slope
    disagreement, clamped to [0.4, 2.0]; quantiles are the median shifted
    by the normal z-scores, in logit space, then rounded to 4 decimals.

Stdlib only. Reads $MAKEFUN_INPUT_DIR (else /input), writes
$MAKEFUN_OUTPUT_DIR (else /output), one *.out per *.in, deterministic.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# First refresh the house saw: 638 -> 3,868 players tracked.
REFRESH_TRACKED_MULT = 3868.0 / 638.0
# Earned counts grow only a little across one refresh: the newly tracked
# cohort has earned almost nothing yet.
REFRESH_EARNED_MULT = 1.20
# Per-tier earned multiplier across a refresh (early-story bronze trophies
# pick up a few more earners than the platinum does).
REFRESH_EARNED_BY_TIER = {"platinum": 1.05, "gold": 1.12, "silver": 1.15, "bronze": 1.25}
# Group ceiling cap on the median, in percent, by tier.
TIER_CEILING = {"platinum": 20.0, "gold": 40.0, "silver": 60.0, "bronze": 90.0}
# Fallback median (percent) when a trophy has no usable rate at all.
TIER_PRIOR = {"platinum": 0.5, "gold": 1.0, "silver": 1.5, "bronze": 2.0}
# Saturation time constants (hours) for tracked / earned log-growth.
# Chosen by Monte Carlo on final-like weeks: 36/72 h beats 72/120 h there
# (mean pinball 0.437 vs 0.513) and leaves 90-minute horizons untouched.
TAU_N = 36.0
TAU_E = 72.0
# The 100%-earned anomaly is pulled to this median with a wide spread.
ANOMALY_MEDIAN = 15.0
ANOMALY_SIGMA = 1.9


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def hours_between(a, b):
    return (b - a).total_seconds() / 3600.0


def lin_slope(xs, ys):
    """Least-squares slope of ys on xs; 0.0 when there is no variation."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 0:
        return 0.0
    return num / den


def saturating_hours(dt_hours, tau):
    """tau * (1 - exp(-dt / tau)): ~dt for short horizons, ~tau for long."""
    if dt_hours <= 0:
        return 0.0
    return tau * (1.0 - math.exp(-dt_hours / tau))


def last_rate(history, idx):
    """Most recent non-null rate for trophy idx, else None."""
    for read in reversed(history):
        rates = read.get("rates") or []
        if idx < len(rates) and rates[idx] is not None:
            return float(rates[idx])
    return None


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    trophies = case["trophies"]
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    at = parse_time(case["at"])
    dt_h = float(case["horizon_minutes"]) / 60.0
    dt_days = float(case["horizon_minutes"]) / 1440.0

    # --- Build the count series: N(t) and E_i(t) per read. ----------------
    times = []          # hours relative to `at` (<= 0)
    ns = []             # players tracked
    earned = []         # per-read list of earned counts (None where unknown)
    for read in history:
        t = hours_between(at, parse_time(read["at"]))
        n = float(read["players_tracked"])
        if n <= 0:
            continue
        row = []
        rates = read.get("rates") or []
        for i in range(len(trophies)):
            r = rates[i] if i < len(rates) else None
            row.append(None if r is None else float(r) / 100.0 * n)
        times.append(t)
        ns.append(n)
        earned.append(row)
    if not times:
        raise ValueError(f"{name} has no usable history reads")

    # Flat history? (page unchanged: every N and every earned total equal)
    log_n = [math.log(n) for n in ns]
    flat_n = max(log_n) - min(log_n) < 1e-9
    total_earned = [sum(e for e in row if e is not None) for row in earned]
    flat_e = (max(total_earned) - min(total_earned)) < 1e-9 if total_earned else True
    flat = flat_n and flat_e

    # --- Observed log-growth slopes. ---------------------------------------
    slope_n = lin_slope(times, log_n)
    # Tier-group earned slopes: regress log(group total earned) on time.
    tiers = [t.get("tier", "bronze") for t in trophies]
    group_slope = {}
    for tier in sorted(set(tiers)):
        idxs = [i for i, tr in enumerate(tiers) if tr == tier]
        ys = []
        xs = []
        for k, row in enumerate(earned):
            vals = [row[i] for i in idxs if row[i] is not None and row[i] > 0]
            if vals:
                ys.append(math.log(sum(vals)))
                xs.append(times[k])
        group_slope[tier] = lin_slope(xs, ys)

    # --- Refresh jump (flat history only) + prior slopes. ------------------
    if flat:
        # One refresh is coming inside the horizon: scale its log-jump by
        # how much of a 90-minute window the horizon covers.
        jump_frac = min(1.0, dt_h / 1.5) if dt_h > 0 else 0.0
        log_jump_n = math.log(REFRESH_TRACKED_MULT) * jump_frac
        # Slow prior continuous growth, for long flat horizons.
        prior_n = math.log(2.0) / 48.0   # tracked doubles ~every 2 days
        prior_e = {"platinum": 0.002, "gold": 0.005, "silver": 0.008, "bronze": 0.010}
    else:
        log_jump_n = 0.0
        jump_frac = 0.0
        prior_n = 0.0
        prior_e = {tier: 0.0 for tier in group_slope}

    eff_n = saturating_hours(dt_h, TAU_N)
    eff_e = saturating_hours(dt_h, TAU_E)
    # Forecast tracked base: observed slope + (flat) jump + prior growth.
    log_n_mult = slope_n * eff_n + log_jump_n + prior_n * eff_n
    n_last = ns[-1]
    n_fore = n_last * math.exp(log_n_mult)

    rows = []
    for i, trophy in enumerate(trophies):
        tier = tiers[i]
        rate = last_rate(history, i)
        tid = trophy["id"]

        # --- The 100% anomaly: do not trust it. ---------------------------
        if rate is not None and rate >= 99.5:
            median = ANOMALY_MEDIAN
            sigma = ANOMALY_SIGMA
            rows.append({"id": tid, "q": quantiles(median, sigma)})
            continue
        if rate is None:
            rate = TIER_PRIOR.get(tier, 1.0)

        # Per-trophy earned slope, shrunk toward its tier-group slope.
        xs, ys = [], []
        for k, row in enumerate(earned):
            e = row[i]
            if e is not None and e > 0:
                xs.append(times[k])
                ys.append(math.log(e))
        own = lin_slope(xs, ys)
        grp = group_slope.get(tier, 0.0)
        slope_e = 0.35 * own + 0.65 * grp
        # Guard: a trophy's earned count cannot outgrow the tracked base
        # by an absurd margin; cap the earned slope at the tracked slope
        # plus a small progression allowance.
        slope_e = min(slope_e, slope_n + prior_n + 0.05)

        if flat:
            log_jump_e = math.log(REFRESH_EARNED_BY_TIER.get(tier, REFRESH_EARNED_MULT)) * jump_frac
            log_e_mult = slope_e * eff_e + log_jump_e + prior_e.get(tier, 0.0) * eff_e
        else:
            log_e_mult = slope_e * eff_e
        e_last = rate / 100.0 * n_last
        e_fore = e_last * math.exp(log_e_mult)
        median = 100.0 * e_fore / n_fore if n_fore > 0 else rate

        # Group ceiling cap (and the hard 0.01..99.99 bounds).
        median = min(median, TIER_CEILING.get(tier, 90.0))
        median = clamp_pct(median)

        # --- Spread. --------------------------------------------------------
        sigma = 0.25 + 0.45 * math.sqrt(max(dt_days, 0.0))
        sigma += 0.85 if flat else 0.35
        sigma += min(0.30, 0.30 * abs(own - grp))
        sigma = min(2.0, max(0.4, sigma))
        rows.append({"id": tid, "q": quantiles(median, sigma)})
    return {"trophies": rows}


def quantiles(median, sigma):
    base = logit(median)
    out = []
    prev = None
    for z in Z:
        q = round4(100.0 * sigmoid(base + z * sigma))
        q = min(99.99, max(0.01, float(q)))
        if prev is not None and q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()

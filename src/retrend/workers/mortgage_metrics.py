"""Deterministic mortgage-rate metrics. Plain code: no LLM, no I/O.

The model only judges these numbers; it never computes or supplies them.
"""

from __future__ import annotations

from datetime import date, timedelta

from retrend.fred import Observation

# Placeholder rule for the rule-based baseline. These numbers are NOT tuned; the
# evaluation harness (Phase 4) should tune or replace them.
BASELINE_WINDOW_DAYS = 91  # 13 weeks
BASELINE_THRESHOLD_BP = 25.0
# A look-back reading must fall within this many days of its target date.
MATCH_TOLERANCE_DAYS = 14

# Metrics shown to the model. Excludes the baseline (to avoid anchoring the model on the
# rule) and calendar dates (see the README note on backtest contamination).
PROMPT_METRIC_KEYS = (
    "latest_rate_pct",
    "days_since_latest_obs",
    "change_4w_bp",
    "change_13w_bp",
    "change_52w_bp",
    "range_52w_low_pct",
    "range_52w_high_pct",
    "percentile_52w",
    "percentile_window",
    "window_years",
    "payment_per_100k",
    "payment_change_52w_pct",
)


def monthly_payment_per_100k(rate_pct: float, years: int = 30) -> float:
    """Monthly principal-and-interest payment on a $100,000 fixed-rate loan."""
    n = years * 12
    r = rate_pct / 100.0 / 12.0
    principal = 100_000.0
    if r == 0:
        return principal / n
    return principal * r / (1.0 - (1.0 + r) ** -n)


def _at_or_before(obs: list[Observation], target: date) -> Observation | None:
    """Newest reading dated on or before ``target`` and within the tolerance window."""
    best = None
    for o in obs:  # sorted ascending
        if o.day <= target:
            best = o
        else:
            break
    if best is None or (target - best.day).days > MATCH_TOLERANCE_DAYS:
        return None
    return best


def baseline_direction(change_13w_bp: float | None) -> tuple[str, str]:
    """Rising rates -> downward pressure on prices; falling rates -> upward pressure."""
    if change_13w_bp is None:
        return "flat", "insufficient history for the 13-week change"
    t = BASELINE_THRESHOLD_BP
    if change_13w_bp >= t:
        return "down", f"13-week change {change_13w_bp:+.1f} bp >= +{t:.0f} bp"
    if change_13w_bp <= -t:
        return "up", f"13-week change {change_13w_bp:+.1f} bp <= -{t:.0f} bp"
    return "flat", f"13-week change {change_13w_bp:+.1f} bp within +/-{t:.0f} bp"


# Alternative rule: judge the LEVEL of borrowing cost, not the recent change. The first backtest
# found the 13-week-change rule had the wrong sign (price momentum tends to lag rate moves), so
# this one asks a different question: is the monthly payment on a fixed loan high or low versus
# the last few years, and has it moved a lot over the year? Thresholds are untuned placeholders;
# compare both rules with `retrend backtest` (variant only_mortgage) before trusting either.
AFFORD_PERCENTILE_HIGH = 0.75
AFFORD_PERCENTILE_LOW = 0.25
AFFORD_PAYMENT_CHANGE_PCT = 5.0


def affordability_direction(percentile_window: float | None,
                            payment_change_52w_pct: float | None) -> tuple[str, str]:
    if percentile_window is None or payment_change_52w_pct is None:
        return "flat", "insufficient history for the affordability rule"
    p, c = percentile_window, payment_change_52w_pct
    if p >= AFFORD_PERCENTILE_HIGH and c >= AFFORD_PAYMENT_CHANGE_PCT:
        return "down", (f"rate at the {p:.0%} percentile of its window and payment up {c:.1f}% over 52 weeks")
    if p <= AFFORD_PERCENTILE_LOW and c <= -AFFORD_PAYMENT_CHANGE_PCT:
        return "up", (f"rate at the {p:.0%} percentile of its window and payment down {abs(c):.1f}% over 52 weeks")
    return "flat", f"rate at the {p:.0%} percentile and payment change {c:+.1f}% over 52 weeks: no clear squeeze or relief"


def compute_metrics(obs: list[Observation], ref_date: date) -> dict[str, float | int | str | None]:
    """Metrics from readings sorted oldest to newest. ``ref_date`` is the run date
    (today, or the as-of date in a backtest)."""
    if not obs:
        raise ValueError("no observations")
    latest = obs[-1]

    def change_bp(days: int) -> float | None:
        prev = _at_or_before(obs, latest.day - timedelta(days=days))
        return None if prev is None else round((latest.value - prev.value) * 100.0, 1)

    c4, c13, c52 = change_bp(28), change_bp(BASELINE_WINDOW_DAYS), change_bp(364)

    prev52 = _at_or_before(obs, latest.day - timedelta(days=364))
    low = high = pct52 = None
    if prev52 is not None:  # only trust 52-week stats when the history really spans a year
        window = [o.value for o in obs if o.day >= latest.day - timedelta(days=364)]
        low, high = min(window), max(window)
        pct52 = round(sum(v <= latest.value for v in window) / len(window), 3)

    payment = monthly_payment_per_100k(latest.value)
    payment_change = None
    if prev52 is not None:
        payment_change = round((payment / monthly_payment_per_100k(prev52.value) - 1.0) * 100.0, 1)

    direction, rule = baseline_direction(c13)
    pct_window = round(sum(o.value <= latest.value for o in obs) / len(obs), 3)
    a_dir, a_rule = affordability_direction(pct_window, payment_change)
    return {
        "latest_rate_pct": latest.value,
        "latest_obs_date": latest.day.isoformat(),
        "days_since_latest_obs": (ref_date - latest.day).days,
        "change_4w_bp": c4,
        "change_13w_bp": c13,
        "change_52w_bp": c52,
        "range_52w_low_pct": low,
        "range_52w_high_pct": high,
        "percentile_52w": pct52,
        "percentile_window": round(sum(o.value <= latest.value for o in obs) / len(obs), 3),
        "window_years": round((latest.day - obs[0].day).days / 365.25, 1),
        "payment_per_100k": round(payment, 2),
        "payment_change_52w_pct": payment_change,
        "baseline_direction": direction,
        "baseline_rule": rule,
        "affordability_direction": a_dir,
        "affordability_rule": a_rule,
    }

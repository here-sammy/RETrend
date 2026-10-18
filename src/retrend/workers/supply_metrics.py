"""Supply / demand metrics from Redfin monthly rows. Plain code."""

from __future__ import annotations

from statistics import mean

PROMPT_METRIC_KEYS = (
    "months_of_supply", "months_of_supply_change_yoy", "inventory_yoy_pct",
    "new_listings_yoy_pct", "homes_sold_yoy_pct", "pending_sales_yoy_pct",
    "median_dom", "median_dom_change_yoy", "price_drops_pct", "price_drops_change_yoy_pp",
    "avg_sale_to_list", "sold_above_list_pct", "median_sale_price_yoy_pct",
)


def _frac_to_pct(v):
    """Redfin ratios (price_drops, sold_above_list) are fractions; show percentages."""
    return None if v is None else round(v * 100.0, 1)


def _avg(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return mean(vals) if vals else None


def _pct(new, old):
    return None if new is None or not old else round((new / old - 1) * 100.0, 1)


def _d(new, old, digits=1):
    return None if new is None or old is None else round(new - old, digits)


def _recent_avg(rows, key):
    return _avg(rows[-3:], key)


def compute_metrics(rows: list[dict]) -> dict:
    """``rows`` are monthly, sorted by period_end. Each level is a 3-month average.

    Year-over-year changes use the file's own YoY columns when it has them (so a 9-month
    file still gets them); otherwise they compare with the same 3 months a year earlier.
    Months of supply is taken from the file if present, else derived as active listings
    divided by homes sold in the month (labelled in ``months_of_supply_source``)."""
    if len(rows) < 3:
        raise ValueError("need at least 3 monthly rows")
    cur = rows[-3:]
    prev = rows[-15:-12] if len(rows) >= 15 else []
    r = lambda v, n=2: None if v is None else round(v, n)  # noqa: E731

    def level(key):
        return _avg(cur, key)

    def yoy_pct(key, yoy_key):
        given = _avg(cur, yoy_key)
        if given is not None:
            return round(given, 1)
        return _pct(level(key), _avg(prev, key) if prev else None)

    inv, sold = level("inventory"), level("homes_sold")
    mos, mos0 = level("months_of_supply"), (_avg(prev, "months_of_supply") if prev else None)
    source = "reported"
    if mos is None and inv is not None and sold:
        source = "derived_active_listings_over_homes_sold"
        mos = inv / sold
        inv_yoy, sold_yoy = yoy_pct("inventory", "inventory_yoy"), yoy_pct("homes_sold", "homes_sold_yoy")
        if inv_yoy is not None and sold_yoy is not None and (1 + sold_yoy / 100) > 0:
            mos0 = (inv / (1 + inv_yoy / 100)) / (sold / (1 + sold_yoy / 100))
    elif mos is None:
        source = None

    pd, pd0 = level("price_drops"), (_avg(prev, "price_drops") if prev else None)
    dom, dom0 = level("median_dom"), (_avg(prev, "median_dom") if prev else None)
    dom_change = _avg(cur, "median_dom_yoy")
    if dom_change is None:
        dom_change = _d(dom, dom0, 0)
    m = {
        "months_of_supply": r(mos),
        "months_of_supply_source": source,
        "months_of_supply_change_yoy": _d(mos, mos0, 2),
        "inventory_yoy_pct": yoy_pct("inventory", "inventory_yoy"),
        "new_listings_yoy_pct": yoy_pct("new_listings", "new_listings_yoy"),
        "homes_sold_yoy_pct": yoy_pct("homes_sold", "homes_sold_yoy"),
        "pending_sales_yoy_pct": yoy_pct("pending_sales", "pending_sales_yoy"),
        "median_dom": r(dom, 0),
        "median_dom_change_yoy": r(dom_change, 0),
        "price_drops_pct": _frac_to_pct(pd),
        "price_drops_change_yoy_pp": None if pd is None or pd0 is None else round((pd - pd0) * 100, 1),
        "avg_sale_to_list": r(level("avg_sale_to_list"), 3),
        "sold_above_list_pct": _frac_to_pct(level("sold_above_list")),
        "median_sale_price_yoy_pct": yoy_pct("median_sale_price", "median_sale_price_yoy"),
        "months_of_history": len(rows),
    }
    m["baseline_direction"], m["baseline_rule"] = baseline_direction(m)
    return m


def baseline_direction(m: dict) -> tuple[str, str]:
    """Months of supply: >6 is a buyer's market, <4 a seller's; a year-over-year move in
    the same direction strengthens the call."""
    mos, chg = m.get("months_of_supply"), m.get("months_of_supply_change_yoy")
    if mos is None:
        return "flat", "months of supply unavailable"
    if mos >= 6 and (chg is None or chg >= 0):
        return "down", f"months of supply {mos} is at buyer's-market levels and not falling"
    if mos <= 4 and (chg is None or chg <= 0):
        return "up", f"months of supply {mos} is at seller's-market levels and not rising"
    return "flat", f"months of supply {mos} (change vs year ago: {chg}) is in the balanced zone"

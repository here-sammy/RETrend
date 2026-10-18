"""Chart-ready Zillow series for one city: values, rents and a few derived measures."""

from __future__ import annotations

from datetime import date


def _yoy(points: list[tuple[str, float]]) -> list[tuple[str, float]]:
    by = {d[:7]: v for d, v in points}
    out = []
    for d, v in points:
        y, m = int(d[:4]), int(d[5:7])
        prev = by.get(f"{y - 1}-{m:02d}")
        if prev:
            out.append((d, round((v / prev - 1) * 100, 2)))
    return out


def _drawdown(points: list[tuple[str, float]]) -> list[tuple[str, float]]:
    peak, out = 0.0, []
    for d, v in points:
        peak = max(peak, v)
        out.append((d, round((v / peak - 1) * 100, 2)))
    return out


def _change(points: list[tuple[str, float]], months: int) -> float | None:
    if len(points) <= months:
        return None
    new, old = points[-1][1], points[-1 - months][1]
    return round((new / old - 1) * 100, 1) if old else None


def _by_month(points, month_date, back):
    y, m = int(month_date[:4]), int(month_date[5:7])
    key = f"{y - back // 12}-{m:02d}"
    for d, v in points:
        if d[:7] == key:
            return v
    return None


def price_view(loc, bundle: dict, dom: dict | None = None) -> dict:
    zhvi, zori = bundle["zhvi"], bundle["zori"]
    series = {
        "zhvi": {"label": "Home value (ZHVI)", "unit": "$", "points": zhvi},
        "zhvi_yoy": {"label": "Home value, change vs a year earlier", "unit": "%", "points": _yoy(zhvi)},
        "drawdown": {"label": "Home value, distance from its prior peak", "unit": "%", "points": _drawdown(zhvi)},
    }
    stats = {"home_value": zhvi[-1][1], "latest_month": zhvi[-1][0],
             "change_1y_pct": _change(zhvi, 12), "change_3y_pct": _change(zhvi, 36),
             "change_5y_pct": _change(zhvi, 60), "from_peak_pct": series["drawdown"]["points"][-1][1],
             "first_month": zhvi[0][0], "months": len(zhvi)}
    if zori:
        series["zori"] = {"label": "Rent (ZORI)", "unit": "$/mo", "points": zori}
        series["zori_yoy"] = {"label": "Rent, change vs a year earlier", "unit": "%", "points": _yoy(zori)}
        rent = {d: v for d, v in zori}
        ratio = [(d, round(v / (rent[d] * 12), 1)) for d, v in zhvi if d in rent and rent[d]]
        if ratio:
            series["price_to_rent"] = {"label": "Price-to-rent (years of rent to equal the home value)",
                                       "unit": "years", "points": ratio}
        stats.update(rent=zori[-1][1], rent_change_1y_pct=_change(zori, 12))
    notes = []
    dtp = bundle.get("dtp")
    if dtp and len(dtp["points"]) >= 2:
        pts = dtp["points"]
        series["dtp"] = {"label": "Days to pending (Zillow, metro)", "unit": "days", "points": pts}
        prev = _by_month(pts, pts[-1][0], 12)
        stats.update(dtp=round(pts[-1][1]), dtp_month=pts[-1][0], dtp_region=dtp["region"],
                     dtp_change_1y_days=None if prev is None else round(pts[-1][1] - prev))
    if dom and len(dom["points"]) >= 2:
        pts = dom["points"]
        series["median_dom"] = {"label": "Median days on market (Redfin)", "unit": "days", "points": pts}
        prev = _by_month(pts, pts[-1][0], 12)
        stats.update(median_dom=round(pts[-1][1]), median_dom_month=pts[-1][0],
                     median_dom_change_1y_days=None if prev is None else round(pts[-1][1] - prev),
                     dom_region=dom["region"], dom_geo=dom["geo"])
    else:
        notes.append("Median days on market is not shown: there is no Redfin data for this place.")
    return {"city": loc.city, "state": loc.state, "region": bundle["region"], "geo": bundle["geo"],
            "quality": bundle["quality"], "series": series, "stats": stats, "notes": notes,
            "source": "Zillow Research (ZHVI: all homes, middle tier, smoothed, seasonally adjusted; "
                      "ZORI: all homes plus multifamily, smoothed; days to pending: Zillow; days on market: Redfin)"}

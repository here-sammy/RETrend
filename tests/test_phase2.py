"""Phase 2: Zillow + Redfin loaders, metrics, and the housing/supply workers on synthetic
fixtures for three cities. No network; fixture files mimic the layouts the loaders expect."""

import gzip
import io
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from helpers import T0, Clock
from retrend.llm import JsonJudge, StubChat
from retrend.location import Location
from retrend.schema import WorkerReport
from retrend.sources.files import SourceError, download_cached
from retrend.sources.matching import metro_matches, norm
from retrend.sources.redfin import NoRegionMatch as RedfinNoMatch, RedfinLoader
from retrend.sources.redfin import extract_region as redfin_extract
from retrend.sources.zillow import NoRegionMatch as ZillowNoMatch, ZillowLoader
from retrend.sources.zillow import extract_region as zillow_extract
from retrend.store import RawStore
from retrend.workers import housing_metrics, supply_metrics
from retrend.workers.housing import HousingPriceWorker
from retrend.workers.supply import SupplyWorker
from retrend.workers.timeseries import Point

CITIES = [Location(city="Seattle", state="WA"), Location(city="Austin", state="TX"),
          Location(city="Phoenix", state="AZ")]
METROS = {"Seattle": "Seattle, WA", "Austin": "Austin, TX", "Phoenix": "Phoenix, AZ"}


def month_ends(n, last=date(2026, 8, 31)):
    out, y, m = [], last.year, last.month
    for _ in range(n):
        nxt = date(y + (m == 12), m % 12 + 1, 1)
        out.append(nxt - timedelta(days=1))
        y, m = (y, m - 1) if m > 1 else (y - 1, 12)
    return sorted(out)


def zillow_csv(geo, growth=0.005):
    days = month_ends(40)
    head = ["RegionID", "SizeRank", "RegionName", "RegionType", "StateName"] + \
        [d.isoformat() for d in days]
    rows = [head]
    names = [(c.city, c.state) for c in CITIES] if geo == "city" else \
        [(METROS[c.city], c.state) for c in CITIES]
    for i, (name, st) in enumerate(names):
        vals = [str(round(400000 * (1 + growth) ** k)) for k in range(len(days))]
        rows.append([str(i), str(i + 1), f'"{name}"', geo, st] + vals)
    # decoys: same city name in another state; blank early months
    rows.append(["99", "50", "Phoenix" if geo == "city" else "Phoenix, MD", geo, "MD"] + ["1"] * len(days))
    rows[1][5:8] = ["", "", ""]
    return "\n".join(",".join(r) for r in rows) + "\n"


def redfin_tsv(geo, mos=3.0):
    cols = ["period_begin", "period_end", "period_duration", "region_type", "region",
            "state_code", "property_type", "is_seasonally_adjusted", "median_sale_price",
            "homes_sold", "pending_sales", "new_listings", "inventory", "months_of_supply",
            "median_dom", "avg_sale_to_list", "sold_above_list", "price_drops"]
    lines = ["\t".join(cols)]
    rtype = "metro" if geo == "metro" else "place"
    for c in CITIES:
        region = f"{c.city}, {c.state} metro area" if geo == "metro" else f"{c.city}, {c.state}"
        for k, end in enumerate(month_ends(20)):
            start = end.replace(day=1)
            for ptype in ("All Residential", "Condo/Co-op"):
                lines.append("\t".join(str(x) for x in [
                    start, end, 30, rtype, region, c.state, ptype, "f", 500000 + k * 1000,
                    100, 90, 120, 300 + k * 10, mos + k * 0.1, 30 - k * 0.5, 0.99, 0.2,
                    "NA" if k == 0 else 0.1]))
    return "\n".join(lines) + "\n"


class FakeResponse(io.BytesIO):
    pass


def opener(files):
    calls = []

    def open_url(url):
        calls.append(url)
        for key, data in files.items():
            if key in url:
                return FakeResponse(data)
        raise OSError("404")
    open_url.calls = calls
    return open_url


def all_files():
    return {
        "Metro_zhvi": zillow_csv("metro").encode(), "City_zhvi": zillow_csv("city").encode(),
        "Metro_zori": zillow_csv("metro", 0.003).encode(), "City_zori": zillow_csv("city", 0.003).encode(),
        "redfin_metro": gzip.compress(redfin_tsv("metro").encode()),
        "city_market": gzip.compress(redfin_tsv("city", 7.0).encode()),
    }


class MatchingTests(unittest.TestCase):
    def test_metro_names(self):
        self.assertTrue(metro_matches("Minneapolis-St. Paul, MN-WI", "St. Paul", "WI"))
        self.assertFalse(metro_matches("Phoenix, MD", "Phoenix", "AZ"))
        self.assertEqual(norm("St.  Louis"), norm("st louis"))


class DownloadTests(unittest.TestCase):
    def test_cached_then_refreshed_and_failure_leaves_no_partial(self):
        with tempfile.TemporaryDirectory() as d:
            op = opener({"f.csv": b"abc"})
            p = download_cached("https://x/f.csv", d, max_age=timedelta(days=1), open_url=op)
            download_cached("https://x/f.csv", d, max_age=timedelta(days=1), open_url=op)
            self.assertEqual(len(op.calls), 1)
            download_cached("https://x/f.csv", d, max_age=timedelta(seconds=-1), open_url=op)
            self.assertEqual(len(op.calls), 2)
            with self.assertRaises(SourceError) as cm:
                download_cached("https://x/g.csv", d, max_age=timedelta(days=1), open_url=op,
                                hint="SET ENV")
            self.assertIn("SET ENV", str(cm.exception))
            self.assertEqual(sorted(x.name for x in Path(d).iterdir()), [p.name])


class ExtractTests(unittest.TestCase):
    def test_zillow_city_exact_and_state_disambiguated(self):
        for c in CITIES:
            p = zillow_extract(zillow_csv("city").splitlines(), c, "zhvi", "city")
            self.assertEqual(p["region"]["name"], c.city)
            self.assertEqual(p["quality"], "exact")
        p = zillow_extract(zillow_csv("city").splitlines(), CITIES[0], "zhvi", "city")
        self.assertEqual(len(p["observations"]), 37)  # blank months dropped

    def test_zillow_metro_is_approximate_and_missing_city_suggests(self):
        p = zillow_extract(zillow_csv("metro").splitlines(), CITIES[1], "zhvi", "metro")
        self.assertEqual(p["quality"], "approximate")
        with self.assertRaises(ZillowNoMatch) as cm:
            zillow_extract(zillow_csv("city").splitlines(), Location(city="Austen", state="TX"),
                           "zhvi", "city")
        self.assertIn("Austin", str(cm.exception))

    def test_zillow_bad_layout(self):
        with self.assertRaises(SourceError):
            zillow_extract(["a,b", "1,2"], CITIES[0], "zhvi", "city")

    def test_redfin_filters_property_type_and_handles_na(self):
        p = redfin_extract(redfin_tsv("metro").splitlines(), CITIES[2], "metro")
        self.assertEqual(len(p["rows"]), 20)
        self.assertIsNone(p["rows"][0]["price_drops"])
        self.assertEqual(p["region"]["name"], "Phoenix, AZ metro area")

    def test_redfin_errors(self):
        with self.assertRaises(SourceError):
            redfin_extract(["a\tb", "1\t2"], CITIES[0], "metro")
        with self.assertRaises(RedfinNoMatch):
            redfin_extract(redfin_tsv("metro").splitlines(), Location(city="Boise", state="ID"), "metro")


class LocalRedfinFileTests(unittest.TestCase):
    CSV = ("Period End,Region,State,Median Sale Price,Homes Sold,Inventory,Months of Supply,"
           "Median DOM,Share of Listings with Price Drops\n"
           + "".join(f'{d.strftime("%B %Y")},"Seattle, WA",WA,"$850,000",100,"1,200",{2+k*0.3},20,{5+k}%\n'
                     for k, d in enumerate(month_ends(9, date(2026, 9, 30))))
           + 'January 2026,"Tacoma, WA",WA,"$500,000",50,400,3.0,25,10%\n')

    def test_hand_downloaded_csv_by_filename_in_data_folder(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "redfin_housing_market_monthly_all_cities_key_metrics_2026_Jan_to_2026_Sep.csv"
             ).write_text(self.CSV, encoding="utf-8")
            p = RedfinLoader(d).extract(CITIES[0], "city")
            self.assertEqual(len(p["rows"]), 9)
            self.assertEqual(p["quality"], "exact")
            r = p["rows"][-1]
            self.assertEqual(r["period_end"], "2026-09-30")
            self.assertEqual(r["median_sale_price"], 850000)
            self.assertEqual(r["inventory"], 1200)
            self.assertAlmostEqual(r["price_drops"], 0.13)

    def test_env_path_and_gzip_tsv(self):
        import os
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "whatever.tsv.gz"
            f.write_bytes(gzip.compress(redfin_tsv("metro").encode()))
            os.environ["REDFIN_METRO_FILE"] = str(f)
            try:
                p = RedfinLoader(Path(d) / "empty").extract(CITIES[1], "metro")
            finally:
                del os.environ["REDFIN_METRO_FILE"]
            self.assertEqual(len(p["rows"]), 20)

    def test_short_history_still_gives_a_report(self):
        rows = redfin_extract(LocalRedfinFileTests.CSV.splitlines(), CITIES[0], "city",
                              delimiter=",")["rows"]
        m = supply_metrics.compute_metrics(rows)
        self.assertIsNone(m["inventory_yoy_pct"])
        self.assertIsNotNone(m["months_of_supply"])


REAL_HEADER = ("LAST UPDATED,FREQUENCY,PERIOD BEGIN,PERIOD END,REGION ID,REGION TYPE,REGION NAME,"
               "HOMES SOLD,HOMES SOLD YOY (%),MEDIAN SALE PRICE NSA ($),MEDIAN SALE PRICE NSA YOY (%),"
               "MEDIAN DAYS ON MARKET (DAYS),MEDIAN DAYS ON MARKET YOY (DAYS),NEW LISTINGS,"
               "NEW LISTINGS YOY (%),ACTIVE LISTINGS,ACTIVE LISTINGS YOY (%),PENDING SALES,"
               "PENDING SALES YOY (%),MEDIAN NEW LISTING PRICE PER SQ.FT. ($),"
               "MEDIAN NEW LISTING PRICE PER SQ.FT. YOY (%)")


def real_style_csv(rtype, names):
    lines = [REAL_HEADER]
    for name in names:
        for k, end in enumerate(month_ends(9, date(2026, 9, 30))):
            lines.append(",".join(str(x) for x in [
                "2026-10-10", "Monthly", end.replace(day=1), end, 1000 + k, rtype, f'"{name}"',
                100, -5.0, 800000, 2.0, 25, 3, 150, 4.0, 400 + k * 10, 20.0, 90, 1.5, 450, 1.0]))
    return "\n".join(lines) + "\n"


class RealRedfinExportTests(unittest.TestCase):
    def test_real_header_city_and_metro(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "redfin_housing_market_monthly_all_cities_key_metrics_2026_Jan_to_2026_Sep.csv"
             ).write_text(real_style_csv("place", ["Seattle, WA", "Austin, TX", "Phoenix, AZ"]))
            (Path(d) / "redfin_housing_market_monthly_all_metros_key_metrics_2026_Jan_to_2026_Sep.csv"
             ).write_text(real_style_csv("metro", ["Seattle, WA", "Austin, TX", "Phoenix, AZ"]))
            loader = RedfinLoader(d)
            for c in CITIES:
                for geo in ("city", "metro"):
                    p = loader.extract(c, geo)
                    self.assertEqual(len(p["rows"]), 9, (c, geo))
            m = supply_metrics.compute_metrics(loader.extract(CITIES[0], "metro")["rows"])
            self.assertEqual(m["inventory_yoy_pct"], 20.0)
            self.assertEqual(m["homes_sold_yoy_pct"], -5.0)
            self.assertEqual(m["median_dom_change_yoy"], 3)
            self.assertEqual(m["months_of_supply_source"], "derived_active_listings_over_homes_sold")
            self.assertEqual(m["months_of_supply"], 4.7)  # 470 / 100
            self.assertIsNone(m["price_drops_pct"])

    def test_worker_runs_on_real_style_file(self):
        with tempfile.TemporaryDirectory() as d, RawStore(clock=Clock()) as store:
            (Path(d) / "x_all_metros_y.csv").write_text(real_style_csv("metro", ["Seattle, WA"]))
            w = SupplyWorker(store, loader=RedfinLoader(d), judge=JsonJudge(StubChat(), store=store))
            r = w.run(CITIES[0])
            self.assertEqual(r.data_as_of, date(2026, 9, 30))
            self.assertTrue(any("estimated" in c for c in r.caveats))


class MetricsTests(unittest.TestCase):
    def test_housing_metrics_known_growth(self):
        pts = [Point(d, 100.0 * 1.01 ** k) for k, d in enumerate(month_ends(40))]
        m = housing_metrics.compute_metrics(pts, None)
        self.assertEqual(m["zhvi_yoy_pct"], 12.7)
        self.assertEqual(m["zhvi_3m_annualized_pct"], 12.7)
        self.assertEqual(m["zhvi_drawdown_from_peak_pct"], 0.0)
        self.assertEqual(m["baseline_direction"], "up")
        self.assertIsNone(m["zori_yoy_pct"])

    def test_housing_falling_and_short_history(self):
        pts = [Point(d, 100.0 * 0.99 ** k) for k, d in enumerate(month_ends(24))]
        self.assertEqual(housing_metrics.compute_metrics(pts, None)["baseline_direction"], "down")
        short = housing_metrics.compute_metrics(pts[-5:], None)
        self.assertIsNone(short["zhvi_yoy_pct"])
        self.assertEqual(short["baseline_direction"], "flat")

    def test_supply_metrics_and_baseline(self):
        p = redfin_extract(redfin_tsv("metro", 3.0).splitlines(), CITIES[0], "metro")
        m = supply_metrics.compute_metrics(p["rows"])
        self.assertEqual(m["months_of_supply_change_yoy"], 1.2)
        self.assertEqual(m["price_drops_pct"], 10.0)
        self.assertEqual(m["baseline_direction"], "flat")  # 3.0 start +~2 → balanced
        hi = supply_metrics.compute_metrics(
            redfin_extract(redfin_tsv("metro", 7.0).splitlines(), CITIES[0], "metro")["rows"])
        self.assertEqual(hi["baseline_direction"], "down")
        short = supply_metrics.compute_metrics(
            redfin_extract(redfin_tsv("metro").splitlines(), CITIES[0], "metro")["rows"][-4:])
        self.assertIsNone(short["inventory_yoy_pct"])


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.clock = Clock()
        self.store = RawStore(clock=self.clock)
        self.op = opener(all_files())

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def judge(self):
        return JsonJudge(StubChat(), store=self.store)

    def housing(self, judge="stub", **kw):
        loader = ZillowLoader(self.dir / "z", open_url=self.op, clock=self.clock)
        return HousingPriceWorker(self.store, loader=loader,
                                  judge=self.judge() if judge == "stub" else judge, **kw)

    def supply(self, judge="stub", **kw):
        loader = RedfinLoader(self.dir / "r", open_url=self.op, clock=self.clock)
        return SupplyWorker(self.store, loader=loader,
                            judge=self.judge() if judge == "stub" else judge, **kw)

    def test_schema_valid_reports_for_three_cities(self):
        for c in CITIES:
            for worker in (self.housing(), self.supply()):
                r = worker.run(c)
                WorkerReport.model_validate(r.model_dump())
                self.assertEqual(r.location, c)
                self.assertTrue(r.sources)
                self.assertTrue(r.summary)
                self.assertIn(r.direction, ("up", "flat", "down"))
                self.assertTrue(r.key_metrics["point_in_time"])

    def test_housing_exact_city_and_data_as_of(self):
        r = self.housing().run(CITIES[0])
        self.assertEqual(r.geography_match, "exact")
        self.assertEqual(r.data_as_of, date(2026, 8, 31))  # Aug 31 + 20d <= Oct 1 → usable
        self.assertEqual(r.key_metrics["baseline_direction"], "up")
        self.assertEqual(len(r.sources), 2)

    def test_publication_lag_hides_recent_month(self):
        r = self.housing().run(CITIES[0], as_of=datetime(2026, 9, 10, tzinfo=timezone.utc))
        self.assertEqual(r.data_as_of, date(2026, 7, 31))  # Aug 31 not yet published
        self.assertFalse(r.key_metrics["point_in_time"])
        self.assertTrue(any("Not point-in-time" in c for c in r.caveats))

    def test_as_of_uses_stored_snapshot(self):
        self.housing().run(CITIES[0])  # stores a snapshot at T0
        r = self.housing().run(CITIES[0], as_of=datetime(2026, 10, 2, tzinfo=timezone.utc))
        self.assertTrue(r.key_metrics["point_in_time"])

    def test_city_missing_falls_back_to_metro_approximate(self):
        files = all_files()
        files["City_zhvi"] = zillow_csv("city").replace("Austin", "Elsewhere").encode()
        self.op = opener(files)
        r = self.housing().run(CITIES[1])
        self.assertEqual(r.geography_match, "approximate")
        self.assertTrue(any("No exact city series" in c for c in r.caveats))

    def test_rent_failure_is_a_caveat_not_a_crash(self):
        files = all_files()
        del files["Metro_zori"], files["City_zori"]
        self.op = opener(files)
        r = self.housing().run(CITIES[0])
        self.assertIsNone(r.key_metrics["zori_yoy_pct"])
        self.assertTrue(any("Rent data unavailable" in c for c in r.caveats))

    def test_supply_metro_default_is_approximate_and_city_is_exact(self):
        self.assertEqual(self.supply().run(CITIES[0]).geography_match, "approximate")
        r = self.supply(geos=("city", "metro")).run(CITIES[0])
        self.assertEqual(r.geography_match, "exact")
        self.assertEqual(r.key_metrics["baseline_direction"], "down")  # 7+ months

    def test_baseline_and_model_failure_fallback(self):
        r = self.housing(judge=None).run(CITIES[0])
        self.assertEqual(r.judged_by, "rule:zhvi-momentum")
        bad = JsonJudge(StubChat(["not json"] * 10), store=self.store, max_attempts=1) \
            if "max_attempts" in JsonJudge.__init__.__code__.co_varnames else \
            JsonJudge(StubChat(["not json"] * 10), store=self.store)
        r = self.supply(judge=bad).run(CITIES[0])
        self.assertIn("fallback", r.judged_by)

    def test_files_downloaded_once_across_cities(self):
        w = self.housing()
        for c in CITIES:
            w.run(c)
        self.assertEqual(len([u for u in self.op.calls if "Metro" in u]), 0)
        self.assertEqual(len(self.op.calls), 2)  # City zhvi + City zori only


if __name__ == "__main__":
    unittest.main()

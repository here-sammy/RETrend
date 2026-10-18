"""Urban Institute downloader (fake API) and its CSV being readable by the schools worker."""

import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path

from retrend import cli
from retrend.location import Location
from retrend.sources import urban
from retrend.sources.files import SourceError
from retrend.sources.schools import SchoolsLoader, extract_district


def fake_api(last=2022, fail_year=None):
    calls = []

    def get(url):
        calls.append(url)
        year = int(url.split("/directory/")[1].split("/")[0])
        if year > last:
            raise urllib.error.HTTPError(url, 404, "nf", {}, None)
        rec = lambda lid, name, n: {"leaid": lid, "lea_name": name, "state_location": "WA", "year": year, "enrollment": n}
        page2 = f"https://educationdata.urban.org/api/v1/school-districts/ccd/directory/{year}/?limit=10000&page=2"
        if "page=2" in url:
            return json.dumps({"results": [rec("5300002", "Tacoma Public Schools", 29000 + year)], "next": None}).encode()
        return json.dumps({"results": [rec("5307710", "Seattle Public Schools", 50000 + 100 * (year - 2013)),
                                       rec("5300003", "Tiny SD", -1)], "next": page2}).encode()
    return get, calls


class Resp(io.BytesIO):
    pass


def opener(get):
    return lambda url: Resp(get(url))


class UrbanTests(unittest.TestCase):
    def run_cli(self, d, open_url, *extra):
        out = io.StringIO()
        rc = cli.main(["--env-file", d + "/.env", "--db", d + "/data/c.db", "--log", d + "/l.jsonl",
                       "schools-fetch", *extra], open_url=open_url, out=out)
        return rc, out.getvalue()

    def test_ten_years_saved_in_database_and_csv_readable_by_worker(self):
        with tempfile.TemporaryDirectory() as d:
            get, calls = fake_api(last=2022)
            rc, text = self.run_cli(d, opener(get))
            self.assertEqual(rc, 0, text)
            self.assertIn("Fetching school years 2003 to 2022", text)
            csv_path = Path(d) / "data" / "files" / "schools" / "urban_enrollment.csv"
            body = csv_path.read_text()
            self.assertTrue(body.startswith("District Name,State,School Year,Enrollment,LEAID"))
            self.assertNotIn("Tiny SD", body)  # negative code skipped
            self.assertIn("Seattle Public Schools,WA,2003-04,49000", body)
            self.assertEqual(body.count("Tacoma"), 20)  # followed 'next' in every year
            payload = extract_district(io.StringIO(body), Location(city="Seattle", state="WA"))
            self.assertEqual([r["year"] for r in payload["rows"]], list(range(2003, 2023)))
            self.assertEqual(SchoolsLoader(csv_path.parent).extract(
                Location(city="Tacoma", state="WA"))["district"], "Tacoma Public Schools")

    def test_rerun_skips_saved_years_and_force_refetches(self):
        with tempfile.TemporaryDirectory() as d:
            get, calls = fake_api()
            self.run_cli(d, opener(get), "--start", "2020", "--end", "2022")
            n = len(calls)
            rc, text = self.run_cli(d, opener(get), "--start", "2020", "--end", "2022")
            self.assertEqual(len(calls), n)
            self.assertIn("skipped 3", text)
            self.run_cli(d, opener(get), "--start", "2020", "--end", "2022", "--force")
            self.assertGreater(len(calls), n)

    def test_failed_year_is_reported_and_exit_code_nonzero(self):
        with tempfile.TemporaryDirectory() as d:
            get, _ = fake_api(last=2021)
            rc, text = self.run_cli(d, opener(get), "--start", "2021", "--end", "2022")
            self.assertEqual(rc, 1)
            self.assertIn("failed 1", text)

    def test_probe_survives_a_timeout_on_an_unpublished_year(self):
        get0, _ = fake_api(last=2022)

        def get(url):
            if "/directory/2024/" in url:
                raise TimeoutError("read timed out")
            return get0(url)
        with tempfile.TemporaryDirectory() as d:
            rc, text = self.run_cli(d, opener(get))
            self.assertEqual(rc, 0, text)
            self.assertIn("2003 to 2022", text)

    def test_layout_change_is_reported(self):
        def get(url):
            return json.dumps({"results": [{"foo": 1, "bar": 2}], "next": None}).encode()
        with tempfile.TemporaryDirectory() as d:
            rc, text = self.run_cli(d, opener(get), "--end", "2022")
            self.assertEqual(rc, 1)
            self.assertIn("fields found", text)


if __name__ == "__main__":
    unittest.main()

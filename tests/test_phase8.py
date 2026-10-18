"""Mortgage affordability rule (alternative baseline)."""

import unittest
from datetime import date, timedelta

from retrend.fred import Observation
from retrend.store import RawStore
from retrend.workers.mortgage import MortgageRateWorker
from retrend.workers.mortgage_metrics import affordability_direction, compute_metrics
from retrend.location import Location
from test_phase4 import fake_fred


def series(rates, end=date(2026, 9, 25)):
    n = len(rates)
    return [Observation(day=end - timedelta(weeks=n - 1 - i), value=r) for i, r in enumerate(rates)]


class AffordabilityTests(unittest.TestCase):
    def test_thresholds(self):
        self.assertEqual(affordability_direction(0.9, 8.0)[0], "down")
        self.assertEqual(affordability_direction(0.1, -8.0)[0], "up")
        self.assertEqual(affordability_direction(0.9, 1.0)[0], "flat")   # high but not rising
        self.assertEqual(affordability_direction(0.5, 9.0)[0], "flat")   # rising but mid-range
        self.assertEqual(affordability_direction(None, 9.0)[0], "flat")

    def test_metrics_carry_both_rules(self):
        rates = [3.0] * 150 + [3.0 + 0.03 * i for i in range(110)]  # long low period, then a climb
        m = compute_metrics(series(rates), date(2026, 9, 28))
        self.assertEqual(m["affordability_direction"], "down")
        self.assertIn("percentile", m["affordability_rule"])
        self.assertIn("baseline_direction", m)

    def test_worker_rule_choice(self):
        loc = Location(city="Seattle", state="WA")
        with RawStore() as st:
            old = MortgageRateWorker(st, fred_source=fake_fred, judge=None).run(loc)
            new = MortgageRateWorker(st, fred_source=fake_fred, judge=None, rule="affordability").run(loc)
            self.assertEqual(old.judged_by, "rule:baseline-13w")
            self.assertEqual(new.judged_by, "rule:affordability-level")
            with self.assertRaises(ValueError):
                MortgageRateWorker(st, fred_source=fake_fred, judge=None, rule="nope")


if __name__ == "__main__":
    unittest.main()

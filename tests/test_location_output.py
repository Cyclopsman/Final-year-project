"""Guard against assigning a simulated outage to the wrong place or hour."""
import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from src.environment.grid_env import load_config
from src.evaluation.location_output import export_locations, location_rows


class LocationOutputTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        with open("config/zone_locations.yaml") as f:
            self.metadata = yaml.safe_load(f)
        self.trace = {"shed": np.zeros((5, 168))}
        self.trace["shed"][0, 23] = 0.25
        self.trace["shed"][4, 24] = 1.0

    def test_name_mapping_and_midnight(self):
        # Deliberately reverse location metadata to catch positional joins.
        self.metadata["locations"] = dict(reversed(list(self.metadata["locations"].items())))
        rows = location_rows(self.cfg, self.trace, self.metadata, "test", 7)
        accra = next(r for r in rows if r["zone"] == "Greater Accra" and r["episode_hour_start"] == 23)
        north = next(r for r in rows if r["zone"] == "Northern" and r["episode_hour_start"] == 24)
        self.assertEqual((accra["reference_place"], accra["shed_fraction"], accra["episode_hour_end"]),
                         ("Accra", 0.25, 24))
        self.assertEqual((north["episode_day"], north["hour_of_day"], north["shed_fraction"]), (2, 0, 1.0))
        self.assertEqual(sum(r["shed_fraction"] for r in rows), self.trace["shed"].sum())

    def test_reject_bad_metadata_and_trace(self):
        bad = copy.deepcopy(self.metadata)
        bad["locations"]["Ashanti"]["latitude"] = float("nan")
        with self.assertRaises(ValueError):
            location_rows(self.cfg, self.trace, bad, "test", 0)
        del bad["locations"]["Ashanti"]
        with self.assertRaises(ValueError):
            location_rows(self.cfg, self.trace, bad, "test", 0)
        self.trace["shed"][0, 0] = -0.25
        with self.assertRaises(ValueError):
            location_rows(self.cfg, self.trace, self.metadata, "test", 0)

    def test_export_coordinates_and_full_schedule(self):
        original = self.trace["shed"].copy()
        with tempfile.TemporaryDirectory() as directory:
            paths = export_locations(self.cfg, self.trace, "test", 7, 23, Path(directory))
            with paths[0].open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 5 * 168)
            features = json.loads(paths[1].read_text())["features"]
            self.assertEqual(len(features), 5)
            accra = features[0]
            self.assertEqual(accra["geometry"]["coordinates"], [-0.197, 5.556])
            self.assertEqual(accra["properties"]["shed_fraction"], 0.25)
            self.assertIn("not affected premises", accra["properties"]["scope"])
            self.assertTrue(paths[2].stat().st_size > 0)
            with self.assertRaises(ValueError):
                export_locations(self.cfg, self.trace, "test", 7, 168, Path(directory))
        np.testing.assert_array_equal(original, self.trace["shed"])


if __name__ == "__main__":
    unittest.main()

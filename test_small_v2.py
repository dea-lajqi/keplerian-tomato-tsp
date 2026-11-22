"""Tests for the two-window repair and original ESA validation."""
import unittest
import random
import json

import numpy as np

from small_v2 import SEED, repair_orders
from global_dp_easy import evaluator
from kttsp_esa_solver import validate


class SmallV2Tests(unittest.TestCase):
    def test_moves_membership_between_holes_and_preserves_visits(self):
        route = tuple(range(12))
        normal = np.ones((12,12),dtype=bool)
        paths = repair_orders(route,(2,5),(7,10),normal,normal,20,random.Random(5))
        self.assertTrue(paths)
        for path in paths:
            self.assertEqual(sorted(path),list(route))
            self.assertEqual(path[:2]+path[5:7]+path[10:],(0,1,5,6,10,11))
        self.assertTrue(any(set(p[2:5]) != set(route[2:5]) for p in paths))

    def test_reserves_exception_budget_across_both_holes(self):
        route = tuple(range(12))
        geometry = np.ones((12,12),dtype=bool)
        normal = np.zeros_like(geometry)
        self.assertEqual(repair_orders(route,(2,5),(7,10),normal,geometry,20,random.Random(5)),[])

    def test_original_evaluator_and_duplicate_rejection(self):
        udp = evaluator().udp
        row = json.loads(SEED.read_text())
        valid = validate(udp,row["path"],row["transfers"])
        self.assertEqual(valid,{"valid":True,"fitness":[111.75601410426336,0.,0.,0.,0.]})
        row["path"][1] = row["path"][0]
        self.assertIsNone(validate(udp,row["path"],row["transfers"]))


if __name__=="__main__":
    unittest.main()

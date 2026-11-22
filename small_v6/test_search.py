"""Regression tests for Small V6's reverse-time transfer filter."""
from pathlib import Path
from types import SimpleNamespace
import time
import unittest

from small_v6.search import Search, SEED, evaluator, validate
import json
import numpy as np


class ReverseTimeTests(unittest.TestCase):
    def test_directional_screen_matches_reflected_velocity(self):
        search=object.__new__(Search)
        search.args=SimpleNamespace(phase=.013)
        search.module=evaluator()
        search.grids={}
        search.deadline=time.monotonic()+20.
        forward=search.grid(False)
        backward=search.grid(True)
        np.testing.assert_equal(backward.z_speed,-forward.z_speed)
        for node in (3,7,14,32):
            r,v=backward.state_at(node,2.)
            self.assertEqual(np.sign(np.cross(r,v)[2]),np.sign(backward.z_speed[node]))

    def test_original_esa_incumbent(self):
        raw=json.loads(Path(SEED).read_text())
        result=validate(evaluator().udp,raw["path"],raw["transfers"])
        self.assertEqual(result,{"valid":True,"fitness":[111.75601410426336,0.,0.,0.,0.]})


if __name__=="__main__":
    unittest.main()

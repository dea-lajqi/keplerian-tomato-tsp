"""Offline local tests; no PyKEP required."""
import json
import math
import random
import sys
import tempfile
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from phase_search import State, select_states, retime_suffix, choose_start

class FakeStats: calls=0
class FakeCache:
    def __init__(self):self.stats=FakeStats()
    def cost(self,a,b,d,f):
        self.stats.calls+=1
        # Baseline tof=1.0 feasible. A 0.5-day transfer is also feasible only
        # near orbital alignment; prevents purely arithmetic test shortcuts.
        return 90.0 if .49<=f<=1.1 and abs(d*4-round(d*4))<.001 else 1000.0
class FakeUDP:
    min_tof=.01;max_time=500;dv_threshold=100.;dv_threshold_exception=600.;n_dv_exception=5

class Checks(unittest.TestCase):
    def test_choose_start_valid(self):
        sol={'path':list(range(181)),'dep':list(range(180)), 'tof':[1.]*180}
        r=random.Random(4)
        for _ in range(100):self.assertTrue(0<=choose_start(sol,r,'fine')<180)
    def test_phase_diversity(self):
        s=[State(x,0,(),()) for x in [1.0,1.125,1.25,1.5,1.75,2.0,2.25]]
        out=select_states(s,5,1.75,random.Random(4))
        self.assertLessEqual(len(out),5)
        self.assertTrue(any(x.arrival>=1.5 for x in out))
    def test_suffix_completes_and_is_valid(self):
        sol={'path':list(range(8)),'dep':[0.,1.,2.,3.,4.,5.,6.], 'tof':[1.]*7}
        udp=FakeUDP();cache=FakeCache()
        candidate,meta=retime_suffix(udp,cache,sol,4,random.Random(22),
                     beam=12,mode='balanced',best_bound=10.,max_calls=100000)
        self.assertIsNotNone(candidate,meta)
        self.assertEqual(len(candidate['dep']),7)
        self.assertEqual(candidate['path'],sol['path'])
        for i,(d,f) in enumerate(zip(candidate['dep'],candidate['tof'])):
            self.assertLessEqual(cache.cost(i,i+1,d,f),100.)
            if i:self.assertGreaterEqual(d+1e-8,candidate['dep'][i-1]+candidate['tof'][i-1])
    def test_too_small_budget_returns_no_bogus_record(self):
        sol={'path':list(range(8)),'dep':[0.,1.,2.,3.,4.,5.,6.], 'tof':[1.]*7}
        cand,meta=retime_suffix(FakeUDP(),FakeCache(),sol,4,random.Random(1),
                max_calls=1)
        self.assertIsNone(cand)
        self.assertTrue(meta['limit_exceeded'])

if __name__=='__main__':unittest.main(verbosity=2)

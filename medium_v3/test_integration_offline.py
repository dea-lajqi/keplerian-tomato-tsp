#!/usr/bin/env python3
"""Supervisor, checkpoint and resume integration, with synthetic fake ESA API.

No results from this file are physical ESA scores.
"""
import json, tempfile, unittest, sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import medium_v3.run as campaign

class FakeUDP:
    n_tomatoes=181
    min_tof=.01
    max_time=500
    dv_threshold=100
    dv_threshold_exception=600
    n_dv_exception=5
    tomato_orbital_parameters=[[3.75e6,.001,.0,0,0,0] for _ in range(181)]
    def compute_transfer(self,a,b,t,f):
        return 30.

FAKE=FakeUDP()
SEED={'path':list(range(181)),'dep':[i*.125 for i in range(180)],'tof':[.125]*180}

def fake_verify(u,sol):
    assert len(sol['path'])==181 and sorted(sol['path'])==list(range(181))
    assert len(sol['dep'])==len(sol['tof'])==180
    assert all(sol['tof'][i]>=u.min_tof for i in range(180))
    assert all(sol['dep'][i+1]+1e-6>=sol['dep'][i]+sol['tof'][i] for i in range(179))
    return [sol['dep'][-1]+sol['tof'][-1],0,0,0,0]

class Integration(unittest.TestCase):
    def test_checkpoint_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)
            def args(trials):
                return SimpleNamespace(directory=dest,minutes=.05,seed=141,
                  checkpoint_seconds=.025,beam=2,max_calls=300,
                  trial_seconds=.5,pool_slack=2,slsqp_iters=1,trials=trials)
            with patch.object(campaign,'official_udp',return_value=FAKE),\
                 patch.object(campaign,'seed_from_files',return_value=(SEED,[22.5,0,0,0,0],'fake-test')),\
                 patch.object(campaign,'verify',side_effect=fake_verify),\
                 patch.object(campaign,'solution_payload',side_effect=lambda sol,f,**kw: {'solution':sol,'fitness':f}):
                campaign.run(args(1))
                initial=json.loads((dest/'checkpoint.json').read_text())
                self.assertEqual(initial['stats']['trials'],1)
                self.assertTrue((dest/'best.json').exists())
                campaign.run(args(2))
                second=json.loads((dest/'checkpoint.json').read_text())
                self.assertEqual(second['stats']['trials'],2)
                self.assertTrue(second['elapsed_seconds']>=initial['elapsed_seconds'])
                self.assertFalse((dest/'pid').exists())

if __name__=='__main__':unittest.main(verbosity=2)

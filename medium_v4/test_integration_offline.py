#!/usr/bin/env python3
"""End-to-end synthetic driver integration, including checkpoint and resume.
This does not establish any ESA Medium record. Uses a fake 181-target oracle.
"""
from __future__ import annotations
import argparse,tempfile,unittest,json,sys,io,contextlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from medium_v4 import driver
from medium_v4.test_offline import FakeUDP,fake_seed

class Integration(unittest.TestCase):
    def test_start_checkpoint_resume(self):
        udp=FakeUDP(181)
        sol=fake_seed(181)
        fit=udp.fitness(sol['dep']+sol['tof']+sol['path'])
        old_udp=driver.official_udp;old_seeds=driver.seeds
        driver.official_udp=lambda root:udp
        driver.seeds=lambda obj,folder:((sol,fit,'FAKE ORACLE'),[(sol,fit,'FAKE ORACLE')])
        try:
            with tempfile.TemporaryDirectory() as td:
                args=argparse.Namespace(directory=Path(td),minutes=.1,seed=341,trials=3,
                  beam=7,call_cap=5000,trial_seconds=1,checkpoint_seconds=.05,
                  esa_checks_per_trial=3,pool_slack=2.5,suffix_probability=0.)
                with contextlib.redirect_stdout(io.StringIO()):
                    driver.run(args)
                ck=driver.read(Path(td)/'checkpoint.json')
                self.assertEqual(ck['stats']['trials'],3)
                self.assertTrue((Path(td)/'best.json').exists())
                self.assertFalse((Path(td)/'pid').exists())
                # Explicit resume: same random generator state, increasing trial cap.
                args.trials=5
                with contextlib.redirect_stdout(io.StringIO()):
                    driver.run(args)
                after=driver.read(Path(td)/'checkpoint.json')
                self.assertEqual(after['stats']['trials'],5)
                self.assertGreaterEqual(after['elapsed'],ck['elapsed'])
                self.assertEqual(after['version'],1)
        finally:
            driver.official_udp=old_udp;driver.seeds=old_seeds

if __name__=='__main__':unittest.main(verbosity=2)

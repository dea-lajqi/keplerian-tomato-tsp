#!/usr/bin/env python3
"""Unit and smoke tests not requiring PyKEP; physics parity requires WSL."""
import math,random,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(Path(__file__).resolve().parent))
from topology import candidate_moves, geometry_cost, shell_crossings, long_transfer_analysis
from timed_search import repair_window,select_phases,State
from analysis import problem_elements,load_sol,diagnostics

class FakeUDP:
    n_tomatoes=12
    min_tof=.01
    max_time=500
    n_dv_exception=5
    dv_threshold=100
    dv_threshold_exception=600
    def __init__(self):
        self.tomato_orbital_parameters=[[3.75e6,.001,0,0,0,0] for _ in range(6)] +\
                                       [[15e6,.001,.01*(i-6),0,0,0] for i in range(6,12)]
    def compute_transfer(self,a,b,t,f):
        return 200. if (a<6)!=(b<6) else 35.

class Cache:
    def __init__(self,udp):
        from types import SimpleNamespace
        self.udp=udp
        self.stats=SimpleNamespace(calls=0)
    def exception(self,a,b,t,tof):
        self.stats.calls+=1
        v=self.udp.compute_transfer(a,b,t,tof)
        return None if v>600 else int(v>100)

class Tests(unittest.TestCase):
    def setUp(self):
        self.u=FakeUDP()
        self.sol={'path':list(range(12)),'dep':[i*.25 for i in range(11)],'tof':[.25]*11}
    def test_shell_count(self):
        self.assertEqual(shell_crossings(self.sol['path'],[x[0] for x in self.u.tomato_orbital_parameters]),1)
    def test_geometry_symmetric(self):
        e=self.u.tomato_orbital_parameters
        self.assertAlmostEqual(geometry_cost(0,1,e),geometry_cost(1,0,e))
    def test_generated_permutations_and_shell_bridges(self):
        rng=random.Random(14)
        moves=candidate_moves(self.sol,self.u.tomato_orbital_parameters,rng,32)
        self.assertTrue(moves)
        for m in moves:
            self.assertEqual(sorted(m.route),list(range(12)))
            self.assertEqual(shell_crossings(m.route,[x[0] for x in self.u.tomato_orbital_parameters]),1)
            self.assertEqual(m.route[:m.left+1],self.sol['path'][:m.left+1])
            self.assertEqual(m.route[m.right:],self.sol['path'][m.right:])
    def test_repair_creates_full_feasible_window(self):
        rng=random.Random(19)
        route=self.sol['path'][:]
        route[2],route[3]=route[3],route[2]
        cache=Cache(self.u)
        partial=repair_window(self.u,cache,self.sol,route,1,5,rng,beam=4,max_calls=3500)
        self.assertTrue(partial)
        for s,d,f in partial:
            self.assertEqual(len(d),5)
            self.assertEqual(len(f),5)
            self.assertLessEqual(s.exceptions,5)
            self.assertGreaterEqual(s.arrival,0)
            self.assertTrue(all(d[i]+f[i]<=d[i+1]+1e-8 for i in range(4)))
    def test_phase_diverse(self):
        x=[State(.125,0,(),()),State(.25,0,(),()),State(1.,1,(),()),State(1.5,0,(),())]
        y=select_phases(x,3,random.Random(8))
        self.assertEqual(len(y),3)
    def test_real_orbital_data_without_pykep(self):
        inst=ROOT/'data/spoc4/kttsp/problems/medium.kttsp'
        solfile=ROOT/'results/esa/medium_20261120T080509Z_bc1f8d8f_4_verified.json'
        if not inst.exists() or not solfile.exists():
            self.skipTest('Run in the original project directory with Medium data')
        elements,_=problem_elements(inst)
        stats=diagnostics(elements,load_sol(solfile))
        self.assertEqual(stats['tomatoes'],181)
        self.assertEqual(stats['shell_crossings'],5)
        self.assertEqual(stats['flights_over_3_days'],14)
        self.assertAlmostEqual(stats['arrival_day'],202.8125)
        rng=random.Random(2026)
        moves=candidate_moves(load_sol(solfile),elements,rng,64)
        self.assertGreater(len(moves),0)
        for m in moves:self.assertEqual(shell_crossings(m.route,[e[0] for e in elements]),5)

if __name__=='__main__':unittest.main(verbosity=2)

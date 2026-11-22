#!/usr/bin/env python3
"""V4 offline tests: fake transfer oracle only; NOT real ESA validation."""
from __future__ import annotations
import json,random,sys,unittest,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from medium_v4.joint import phases,times_of_flight,joint_window,prune,Node,edge_exceptions,window_choices,terminal_crossing
from medium_overnight.search import DVCache
from medium_v4.driver import diversify,signature,atomic_json,read

class FakeUDP:
    def __init__(self,n=9, hard=False):
        self.n_tomatoes=n; self.min_tof=.01;self.max_time=500
        self.dv_threshold=100;self.dv_threshold_exception=600;self.n_dv_exception=5
        self.tomato_orbital_parameters=[[7000000+i*10,0,0,0,0,0] for i in range(n)]
        self.hard=hard
    def compute_transfer(self,a,b,d,flight):
        if a==b or flight<self.min_tof:return float('inf')
        if self.hard and abs(a-b)>2:return 700.
        return 40.
    def fitness(self,x):
        n=self.n_tomatoes
        dep=x[:n-1]; tof=x[n-1:2*n-2]; path=x[2*n-2:]
        perm=len(set(path))-n
        dvs=[self.compute_transfer(path[i],path[i+1],dep[i],tof[i]) for i in range(n-1)]
        bad=sum(v<=600+1e-6 for v in dvs)-(n-1)
        timeok=sum(dep[i]+tof[i]<=dep[i+1]+1e-6 for i in range(n-2))-(n-2)
        exceed=sum(v>100 and v<=600 for v in dvs)-5
        return [dep[-1]+tof[-1],perm,bad,timeok,exceed]

def fake_seed(n=9):
    dep=[float(i) for i in range(n-1)]
    return {'path':list(range(n)),'dep':dep,'tof':[.75]*(n-1)}

class V4Checks(unittest.TestCase):
    def test_baseline_guaranteed_and_preserves_outer_tour(self):
        u=FakeUDP();c=DVCache(u);s=fake_seed()
        got=joint_window(u,c,s,2,5,random.Random(0),beam=12,call_cap=60000,deadline=None)
        self.assertTrue(got,'At least original schedule must survive')
        for q,m in got:
            self.assertEqual(sorted(q['path']),list(range(u.n_tomatoes)))
            self.assertEqual(q['path'][:3],s['path'][:3]);self.assertEqual(q['path'][5:],s['path'][5:])
            self.assertEqual(q['dep'][:2],s['dep'][:2]);self.assertEqual(q['dep'][5:],s['dep'][5:])
            self.assertEqual(u.fitness(q['dep']+q['tof']+q['path'])[1:4],[0,0,0])
    def test_construct_new_full_order(self):
        u=FakeUDP();c=DVCache(u);s=fake_seed()
        got=joint_window(u,c,s,3,6,random.Random(123),beam=20,call_cap=80000,max_results=20)
        self.assertTrue(any(q['path']!=s['path'] for q,_ in got),'Must construct a novel *complete* order')
    def test_endgame_can_improve_valid_fitness(self):
        u=FakeUDP();c=DVCache(u);s=fake_seed()
        got=joint_window(u,c,s,5,8,random.Random(3),beam=20,call_cap=80000,max_results=20)
        self.assertTrue(got)
        better=[q for q,_ in got if u.fitness(q['dep']+q['tof']+q['path'])[0]<7.75-1e-9]
        self.assertTrue(better)
        self.assertTrue(all(u.fitness(q['dep']+q['tof']+q['path'])[1:4]==[0,0,0] for q in better))
    def test_last_edge_shell_crossing_supported(self):
        u=FakeUDP(181)
        u.tomato_orbital_parameters[-1][0]=15000000
        base=u.compute_transfer
        def dv(a,b,d,flight):
            value=base(a,b,d,flight)
            return 400. if (a==180) != (b==180) else value
        u.compute_transfer=dv
        u.n_dv_exception=1
        c=DVCache(u);sol=fake_seed(181)
        # Reordering inner-shell nodes before the *outer-shell final target* is allowed.
        result=joint_window(u,c,sol,176,180,random.Random(21),beam=24,
                            call_cap=80000,max_results=20)
        self.assertTrue(result)
        self.assertTrue(all(q['path'][-1]==180 for q,_ in result))
        self.assertTrue(all(sum(edge_exceptions(q,c))<=1 for q,_ in result))
        shorten=terminal_crossing(u,c,sol,random.Random(3),eval_limit=10000)
        self.assertIsNotNone(shorten)
        self.assertLess(shorten['dep'][-1]+shorten['tof'][-1],179.75)

    def test_exception_budget(self):
        u=FakeUDP();c=DVCache(u);s=fake_seed()
        self.assertEqual(sum(edge_exceptions(s,c)),0)
        self.assertTrue(window_choices(s,u,random.Random(1)))
    def test_duplicate_pool_and_atomic_checkpoint(self):
        s=fake_seed(); f=[7.75,0,0,0,-5]
        clone={'solution':s,'fitness':f,'source':'one'}
        clone2={'solution':s,'fitness':f,'source':'two'}
        result=diversify([clone,clone2]);self.assertEqual(len(result),1)
        with tempfile.TemporaryDirectory() as temp:
            dst=Path(temp)/'checkpoint.json'
            atomic_json(dst,{'pool':result,'rng':random.Random(7).getstate()})
            self.assertEqual(len(read(dst)['pool']),1)
    def test_synthetic_reachability_not_confused_with_proof(self):
        u=FakeUDP(hard=True);c=DVCache(u);s=fake_seed()
        got=joint_window(u,c,s,2,5,random.Random(123),beam=12,call_cap=60000)
        self.assertTrue(got)
        for q,_ in got:
            fitness=u.fitness(q['dep']+q['tof']+q['path'])
            self.assertLessEqual(fitness[2],0)
    def test_real_medium_instance_shape_without_pykep(self):
        root=Path(__file__).resolve().parent.parent
        data=json.loads((root/'results/esa/medium_20261120T080509Z_bc1f8d8f_4_verified.json').read_text())
        self.assertEqual(len(data['path']),181)
        self.assertEqual(len(data['transfers']),180)
        self.assertEqual(len(set(data['path'])),181)
        self.assertEqual(data['validation']['fitness'],[202.8125,0,0,0,0])

if __name__=='__main__':unittest.main(verbosity=2)

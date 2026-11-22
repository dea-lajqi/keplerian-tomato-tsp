"""Offline deterministic unit tests: require only standard Python (not PyKEP)."""
import json
import random
import tempfile
import unittest
from pathlib import Path

from validation import verify, to_solution, as_chromosome
from search import DVCache, beam_block, greedy_suffix, try_mutation, mutation_order
from campaign import atomic_json, read_json, fingerprint, tuples


class FakeMedium:
    """Synthetic tiny orbit-like network to test search mechanics, not physics."""
    n_tomatoes=6
    min_tof=.01
    max_time=500
    dv_threshold=100
    dv_threshold_exception=600
    n_dv_exception=5
    tomato_orbital_parameters=[[1.5e7,0,0,0,0,0] for _ in range(6)]
    def compute_transfer(self,a,b,t,f):
        # Valid only for positive flights; departures and order may vary freely.
        if a==b or f<.05:return float('inf')
        return 50.0
    def fitness(self,x):
        n=self.n_tomatoes
        ds=x[:n-1];fs=x[n-1:2*n-2];order=x[2*n-2:]
        p=len(set(map(int,order)))-n
        dvs=[self.compute_transfer(int(order[i]),int(order[i+1]),ds[i],fs[i]) for i in range(n-1)]
        dv=sum(v<=self.dv_threshold_exception+1e-6 for v in dvs)-(n-1)
        late=sum(ds[i]+fs[i] <= ds[i+1]+1e-6 for i in range(n-2))-(n-2)
        return [ds[-1]+fs[-1],p,dv,late,sum(v>100 for v in dvs)-5]


class OfflineChecks(unittest.TestCase):
    def setUp(self):
        self.udp=FakeMedium()
        self.sol={'path':list(range(6)),'dep':[0.,1.,2.,3.,4.],'tof':[1.]*5}
    def test_baseline_fake_esa(self):
        f=verify(self.udp,self.sol)
        self.assertEqual(f[:4],[5.,0.,0.,0.])
    def test_reject_wrong_permutation(self):
        bad={**self.sol,'path':[0,1,2,3,3,5]}
        with self.assertRaises(ValueError):verify(self.udp,bad)
    def test_reject_chronology(self):
        bad={**self.sol,'dep':[0.,.1,2.,3.,4.]}
        with self.assertRaises(ValueError):verify(self.udp,bad)
    def test_reject_bounds(self):
        bad={**self.sol,'tof':[.005,1,1,1,1]}
        with self.assertRaises(ValueError):verify(self.udp,bad)
    def test_block_can_optimize(self):
        cache=DVCache(self.udp)
        new=[0,2,1,3,4,5]
        options=beam_block(self.udp,cache,self.sol['path'],new,
                self.sol['dep'],self.sol['tof'],0,3,random.Random(0),beam=6,eval_limit=5000)
        self.assertTrue(options)
        arrival,ex,ds,fs=min(options,key=lambda x:x[0])
        self.assertLess(arrival,4)
        tail=greedy_suffix(self.udp,cache,new,self.sol['dep'],self.sol['tof'],3,arrival,ex,random.Random(3))
        self.assertIsNotNone(tail)
        sd,sf,_=tail
        result={'path':new,'dep':ds+sd,'tof':fs+sf}
        self.assertLess(verify(self.udp,result)[0],5)
    def test_checkpoint_resume_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            name=Path(tmp)/'state.json'
            r=random.Random(5);r.random(); state={'sol':self.sol,'rng':r.getstate()}
            atomic_json(name,state)
            saved=read_json(name)
            self.assertEqual(saved['sol'],self.sol)
            r2=random.Random();r2.setstate(tuples(saved['rng']))
            self.assertEqual(r.random(),r2.random())
    def test_mutation_preserves_tomatoes(self):
        arr=[1.5e7]*6;rng=random.Random(2)
        for _ in range(10):
            move=mutation_order(self.sol['path'],rng,arr,'local')
            if move:self.assertEqual(sorted(move[2]),list(range(6)))


class OfflineSupervisorTests(unittest.TestCase):
    def test_independent_candidate_promoted(self):
        from campaign import process_candidates
        udp=FakeMedium()
        sol={'path':list(range(6)), 'dep':[0.,.125,.25,.375,.5], 'tof':[.125]*5}
        fit=verify(udp,sol)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'queue').mkdir()
            (root/'records').mkdir()
            old={'score':5.0,'solution':{'path':list(range(6)),
                      'dep':[0,1,2,3,4],'tof':[1]*5}, 'fitness':[5,0,0,0,-5]}
            atomic_json(root/'queue'/'candidate_0_123_1.json',
                {'solution':sol,'fitness':fit,'worker':0,'mode':'test','trial':1})
            new=process_candidates(root,udp,old)
            self.assertLess(new['score'],5.0)
            self.assertEqual(len(list((root/'queue').glob('candidate_*.json'))),0)
            self.assertEqual(len(list((root/'records').glob('record_*.json'))),1)
            self.assertEqual(verify(udp,new['solution'])[0],new['score'])
    def test_reject_invalid_worker_candidate(self):
        from campaign import process_candidates
        udp=FakeMedium()
        sol={'path':list(range(6)),'dep':[0,1,2,3,4],'tof':[.001]*5}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'queue').mkdir();(root/'records').mkdir()
            old={'score':5.0,'solution':{'path':list(range(6)),
                           'dep':[0,1,2,3,4],'tof':[1]*5},'fitness':[5,0,0,0,-5]}
            atomic_json(root/'queue'/'candidate_0_123_1.json',
                {'solution':sol,'fitness':[4,0,0,0,0], 'worker':0,'mode':'test','trial':1})
            new=process_candidates(root,udp,old)
            self.assertEqual(new['score'],5.0)
            self.assertEqual(len(list((root/'queue').glob('*.invalid'))),1)

    def test_zero_budget_supervisor_and_resume_metadata(self):
        from unittest.mock import patch
        import types
        import campaign as camp
        udp=FakeMedium()
        source={'path':list(range(6)),'dep':[0,1,2,3,4],'tof':[1]*5}
        with tempfile.TemporaryDirectory() as tmp:
            args=types.SimpleNamespace(directory=tmp,workers=1,seed=33,hours=0.0)
            with patch.object(camp,'official_udp',return_value=udp), \
                 patch.object(camp,'baseline',return_value=(source,verify(udp,source))):
                camp.campaign(args)
                state=read_json(Path(tmp)/'campaign.json')
                self.assertIsNone(state['supervisor_pid'])
                self.assertEqual(read_json(Path(tmp)/'best.json')['score'],5.0)
                self.assertFalse((Path(tmp)/'pid').exists())
                # Running again must load the previously validated saved record.
                camp.campaign(args)
                self.assertEqual(read_json(Path(tmp)/'best.json')['score'],5.0)

if __name__=='__main__':unittest.main(verbosity=2)

#!/usr/bin/env python3
"""Reproducible FAKE-PHYSICS benchmark for reachability logic, NOT ESA performance."""
from __future__ import annotations
import json,random,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from medium_overnight.search import DVCache
from medium_v4.joint import joint_window,terminal_crossing
from medium_v4.test_offline import FakeUDP,fake_seed

def experiment():
    udp=FakeUDP(181,hard=True)
    udp.tomato_orbital_parameters[-1][0]=15000000
    transfer=udp.compute_transfer
    def cost(a,b,d,tof):
        original=transfer(a,b,d,tof)
        if (a==180)!=(b==180):return 400. if original<600 else original
        return original
    udp.compute_transfer=cost
    sol=fake_seed(181)
    results={}
    for mode in ('fixed_order','joint_reorder','last_crossing'):
        rng=random.Random(100 + len(mode));cache=DVCache(udp)
        start=time.perf_counter();complete=0;novel=set();faster=0
        for step in range(18):
            if mode=='last_crossing':
                q=terminal_crossing(udp,cache,sol,rng,eval_limit=1500)
                got=[(q,{})] if q else []
            else:
                left=rng.choice([11,55,108,150,176]);right=min(180,left+rng.choice([3,4,5]))
                got=joint_window(udp,cache,sol,left,right,rng,
                   beam=14,call_cap=12000,max_results=10,reorder=mode=='joint_reorder')
            for q,_ in got:
                if q is None:continue
                f=udp.fitness(q['dep']+q['tof']+q['path'])
                if f[1]==0 and f[2]<=0 and f[3]<=0 and f[4]<=0:
                    complete+=1
                    if q['path']!=sol['path']:novel.add(tuple(q['path']))
                    if f[0]<179.75-1e-8:faster+=1
        results[mode]={'trials':18,'full_fake_valid_candidates':complete,
               'novel_fake_valid_orders':len(novel),'faster_fake_valid_timing_candidates':faster,
               'transfer_oracle_calls':cache.stats.calls,'seconds':round(time.perf_counter()-start,3)}
    return results

if __name__=='__main__':
    print('SYNTHETIC ONLY, NOT AN ESA MEDIUM FITNESS TEST')
    print(json.dumps(experiment(),indent=2))

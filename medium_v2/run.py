#!/usr/bin/env python3
"""ESA Medium V2: phase-diverse suffix beam search (independent of V1 and Easy)."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from medium_overnight.validation import official_udp, verify, baseline, solution_payload
from medium_overnight.search import DVCache
from phase_search import exceptions_by_leg, retime_suffix, choose_start

OLD=ROOT/'results/esa/medium_overnight/run_20261122/best.json'
DEFAULT=ROOT/'results/esa/medium_v2/run_01'
STOP=False

def halt(*_):
    global STOP
    STOP=True

def atom(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.parent/('.'+path.name+f'.{os.getpid()}.tmp')
    with temp.open('w') as fd:
        json.dump(data,fd,indent=2,allow_nan=False)
        fd.flush();os.fsync(fd.fileno())
    os.replace(temp,path)

def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,json.JSONDecodeError):return default

def timekey(sol):
    data={'p':sol['path'],'d':[round(x,5) for x in sol['dep']],
          'f':[round(x,5) for x in sol['tof']]}
    return hashlib.sha256(json.dumps(data,separators=(',',':')).encode()).hexdigest()[:20]

def valid_source(udp,source):
    if not source.is_file():
        raise FileNotFoundError('Expected the saved ESA-validated overnight record at '
              f'{source}. Run bash medium_overnight/status.sh first.')
    data=read(source)
    if not isinstance(data,dict) or 'solution' not in data:
        raise ValueError('Medium overnight best.json must contain solution')
    sol=data['solution'];fit=verify(udp,sol)
    if abs(float(data.get('score',fit[0]))-fit[0])>1e-6:
        raise ValueError('ESA score mismatch in saved record')
    return sol,fit

def phase_run(args):
    signal.signal(signal.SIGINT,halt);signal.signal(signal.SIGTERM,halt)
    os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
    try:os.nice(12)
    except (OSError,AttributeError):pass
    if hasattr(os,'sched_getaffinity'):
        try:
            cpus=sorted(os.sched_getaffinity(0))
            if len(cpus)>5:os.sched_setaffinity(0,{cpus[-3]})
        except (OSError,PermissionError):pass
    folder=args.directory.resolve();folder.mkdir(parents=True,exist_ok=True)
    source=Path(args.source).resolve()
    udp=official_udp(ROOT)
    seed,seed_fit=valid_source(udp,source)
    current=read(folder/'best.json')
    if current:
        initial=current['solution'];f=verify(udp,initial)
        if abs(f[0]-current['score'])>1e-7:raise ValueError('Checkpoint best ESA mismatch')
        if seed_fit[0]<f[0]:initial=seed;f=seed_fit
    else:initial=seed;f=seed_fit
    best={'solution':initial,'score':f[0],'fitness':f,'source':str(source)}
    atom(folder/'best.json',best)
    state=read(folder/'checkpoint.json',{})
    rng=random.Random(args.seed)
    if state.get('random_state'):
        def tuples(x):return tuple(tuples(y) for y in x) if isinstance(x,list) else x
        rng.setstate(tuples(state['random_state']))
    current_sol=initial
    if state.get('current_solution'):
        try:
            cfit=verify(udp,state['current_solution'])
            if cfit[0]<=best['score']+.3:current_sol=state['current_solution']
        except Exception:pass
    cache=DVCache(udp,limit=150000)
    flags=exceptions_by_leg(udp,current_sol,cache)
    trials=int(state.get('trials',0));valids=int(state.get('valids',0));savings=int(state.get('records',0))
    started=time.monotonic();previous=float(state.get('elapsed_seconds',0))
    duration=args.hours*3600
    last_save=0;best_seen={timekey(current_sol)}
    arms=['fine','balanced','phase']
    weights=state.get('weights',{a:{'tries':0,'complete':0,'records':0} for a in arms})
    for a in arms:weights.setdefault(a,{'tries':0,'complete':0,'records':0})
    def checkpoint():
        atom(folder/'checkpoint.json',{
          'version':1,'trials':trials,'valids':valids,'records':savings,'weights':weights,
          'random_state':rng.getstate(),'elapsed_seconds':previous+time.monotonic()-started,
          'current_solution':current_sol,'best':best['score'],
          'lambert_calls':cache.stats.calls,'cache_hits':cache.stats.cache_hits,
          'updated_unix':time.time()})
    print(f'MEDIUM V2: original ESA validated seed {seed_fit[0]:.10f}; best {best["score"]:.10f}',flush=True)
    print(f'V2 one low-priority worker, dedicated {folder}; budget {duration/3600:.3f} hours',flush=True)
    while not STOP and previous+time.monotonic()-started<duration and not (folder/'stop.request').exists():
        elapsed=previous+time.monotonic()-started
        if elapsed>=duration:break
        trials+=1
        # Balancing completes vs records. Always reserve exploration.
        if rng.random()<.18:
            arm=rng.choice(arms)
        else:
            def merit(a):
                s=weights[a]
                return (3*s['records']+s['complete']*.15)/(1+s['tries']) + .35/(1+s['tries'])**.5
            arm=max(arms,key=lambda a:merit(a)+rng.random()*.10)
        weights[arm]['tries']+=1
        if rng.random()<.18:
            current_sol=best['solution']
            flags=exceptions_by_leg(udp,current_sol,cache)
        start=choose_start(current_sol,rng,arm)
        try:
            candidate, info=retime_suffix(udp,cache,current_sol,start,rng,
               beam=args.beam,mode=arm,exception_flags=flags,
               best_bound=min(udp.max_time,best['score']+.30),max_calls=args.max_calls)
            if candidate:
                weights[arm]['complete']+=1
                val=candidate['dep'][-1]+candidate['tof'][-1]
                if val<best['score']+.3 and timekey(candidate) not in best_seen:
                    cf=verify(udp,candidate)
                    valids+=1
                    best_seen.add(timekey(candidate))
                    # Only retain alternative SAME-PATH phase schedules if a
                    # complete ESA-valid result was constructed.
                    if cf[0]<best['score']-1e-8:
                        payload=solution_payload(candidate,cf,udp=udp,provenance={
                          'optimizer':'medium_v2_phase_beam','seed':args.seed,
                          'trial':trials,'start':start,'arm':arm})
                        record=folder/'records'/f'v2_record_{cf[0]:.10f}_t{trials}.json'
                        atom(record,payload)
                        best={'solution':candidate,'score':cf[0],'fitness':cf,
                              'record':str(record.relative_to(folder)),'source':str(source)}
                        atom(folder/'best.json',best)
                        weights[arm]['records']+=1;savings+=1
                        print(f'NEW ESA-VALID MEDIUM V2 {cf[0]:.10f} via {arm} '
                              f'from edge {start}; trial {trials}',flush=True)
                    if cf[0]<=best['score']+.1 and rng.random()<.55:
                        current_sol=candidate
                        flags=exceptions_by_leg(udp,current_sol,cache)
        except Exception as exc:
            if trials%25==0:print(f'Candidate error {type(exc).__name__}: {exc}',flush=True)
        if time.monotonic()-last_save>60:
            checkpoint();last_save=time.monotonic()
            print(f'progress trials={trials} complete={sum(w["complete"] for w in weights.values())} '
                  f'ESAchecks={valids} Lambert={cache.stats.calls} best={best["score"]:.10f}',flush=True)
    checkpoint()
    print(f'FINAL ESA-valid Medium V2 {best["score"]:.10f} days; trials {trials}; '
          f'budget consumed {previous+time.monotonic()-started:.1f} seconds',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['run','status','verify-best'])
    p.add_argument('--directory',type=Path,default=DEFAULT)
    p.add_argument('--source',type=Path,default=OLD)
    p.add_argument('--hours',type=float,default=.5)
    p.add_argument('--seed',type=int,default=88421)
    p.add_argument('--beam',type=int,default=10)
    p.add_argument('--max-calls',type=int,default=90000)
    a=p.parse_args()
    if a.action=='run':
        if not (0<a.hours<=24):p.error('hours must be in (0,24]')
        phase_run(a)
    elif a.action=='status':
        folder=a.directory.resolve();state=read(folder/'checkpoint.json',{})
        best=read(folder/'best.json',{})
        if not best:
            print('No medium_v2 run found at',folder)
        else:
            print('Medium V2 best saved:',best.get('score'),'days')
            print('Records:',state.get('records'),'trials:',state.get('trials'),
                  'complete:',sum(x.get('complete',0) for x in state.get('weights',{}).values()))
            print('Elapsed:',state.get('elapsed_seconds'),'Lambert:',state.get('lambert_calls'))
            print('PID (when started via start.sh):',
                  (folder/'pid').read_text().strip() if (folder/'pid').exists() else 'not available')
    else:
        udp=official_udp(ROOT)
        best=read(a.directory/'best.json')
        if not best:best={'solution':valid_source(udp,a.source)[0]}
        print('Original ESA Medium fitness:',verify(udp,best['solution']))

if __name__=='__main__':main()

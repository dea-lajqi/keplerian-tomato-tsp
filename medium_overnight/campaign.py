#!/usr/bin/env python3
"""Standalone Medium-only overnight optimizer; no repository writes outside RUN_DIR.

Start from repository root:
  python -u medium_overnight/campaign.py run --hours 8 --workers 2 --seed 20261122
  python medium_overnight/campaign.py status
  python medium_overnight/campaign.py verify-best
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
import traceback
import hashlib
from pathlib import Path

from validation import baseline, official_udp, verify, solution_payload, to_solution
from search import DVCache, try_mutation, try_retime, score

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN = ROOT / 'results/esa/medium_overnight/run_20261122'
KEEP_GOING = True


def stop_handler(sig, _frame):
    global KEEP_GOING
    KEEP_GOING = False


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.'+path.name+f'.{os.getpid()}.tmp')
    with temp.open('w',encoding='utf-8') as stream:
        json.dump(obj,stream,indent=2,ensure_ascii=False)
        stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)


def read_json(path, fallback=None):
    try:
        with Path(path).open(encoding='utf-8') as f: return json.load(f)
    except (OSError,ValueError):return fallback


def rng_as_json(rng): return rng.getstate()

def tuples(value):
    if isinstance(value,list):return tuple(tuples(x) for x in value)
    return value


def fingerprint(sol):
    return hashlib.sha256(json.dumps(sol['path'],separators=(',',':')).encode()).hexdigest()[:16]


def pin_low_priority(worker_id, total_workers):
    # Best effort only. Keep most logical CPUs available to the Easy search.
    try:os.nice(12)
    except (OSError,AttributeError):pass
    if sys.platform.startswith('linux') and hasattr(os,'sched_getaffinity'):
        try:
            cpus=sorted(os.sched_getaffinity(0))
            # Reserve >= 3 CPUs for the Easy campaign; pin each Medium worker to
            # one of the highest-numbered CPUs. Pinning reduces accidental load.
            if len(cpus)>=total_workers+4:
                os.sched_setaffinity(0,{cpus[-total_workers+worker_id]})
                return cpus[-total_workers+worker_id]
        except (OSError,PermissionError):pass
    return 'system-managed'


def worker(args):
    rundir=Path(args.directory)
    cpu=pin_low_priority(args.worker_id,args.workers)
    udp=official_udp(ROOT)
    base, fit=baseline(ROOT,udp)
    axes=[float(x[0]) for x in udp.tomato_orbital_parameters]
    c=DVCache(udp)
    wid=args.worker_id
    checkpoint=rundir/f'worker_{wid}.json'
    saved=read_json(checkpoint,{})
    if saved.get('version')==1:
        pool=saved.get('pool',[]); counts=saved.get('counts',{})
        rng=random.Random()
        try:rng.setstate(tuples(saved['random_state']))
        except Exception:rng.seed(args.seed+wid*65537)
    else:
        pool=[];counts={};rng=random.Random(args.seed+wid*65537)
    pool=[p for p in pool if isinstance(p,dict) and 'sol' in p and 'fitness' in p]
    if not any(fingerprint(p['sol'])==fingerprint(base) for p in pool):
        pool.append({'sol':base,'fitness':fit,'mode':'baseline'})
    actions=['local','diversify','crossing','retime']
    for key in actions:counts.setdefault(key,{'tries':0,'feasible':0,'novel':0,'records':0})
    best_global=read_json(rundir/'best.json',{})
    best_limit=min(fit[0],best_global.get('score',fit[0]))
    last_save=0.; last_reload=0.; started=time.monotonic()
    trials=int(saved.get('trials',0)); full_checks=int(saved.get('full_checks',0));
    seen=set(fingerprint(p['sol']) for p in pool)
    print(f'Worker {wid} started; assigned CPU={cpu}; pool={len(pool)}; incumbent={best_limit:.8f}',flush=True)
    while KEEP_GOING and not (rundir/'stop.request').exists():
        now=time.monotonic()
        # Save a heartbeat even if every candidate is infeasible. Most failed
        # route searches exit early via `continue` below.
        if now-last_save >= 90:
            atomic_json(checkpoint,{'version':1,'worker_id':wid,'trials':trials,
                'full_checks':full_checks,'counts':counts,'pool':pool,
                'random_state':rng_as_json(rng),'cache_calls':c.stats.calls,
                'cache_hits':c.stats.cache_hits,'updated_unix':time.time()})
            print(f'Worker {wid}: trials={trials}, verified={full_checks}, '
                  f'Lambert calls={c.stats.calls}, pool={len(pool)}, best={best_limit:.8f}',flush=True)
            last_save=now
        if now-last_reload>25:
            global_best=read_json(rundir/'best.json',{})
            if global_best and global_best.get('score',1e9)<best_limit-1e-8:
                best_limit=global_best['score']
                if global_best.get('solution'):
                    new=global_best['solution']
                    try:
                        f=verify(udp,new)
                        if fingerprint(new) not in seen:
                            pool.append({'sol':new,'fitness':f,'mode':'shared-best'})
                            seen.add(fingerprint(new))
                    except Exception:pass
            last_reload=now
        # UCB-like strategy allocation: try methods that actually build feasible
        # complete schedules, plus exploration so we do not lock onto one search.
        alltries=1+sum(counts[k]['tries'] for k in actions)
        priorities=[]
        for action in actions:
            st=counts[action];t=st['tries']
            prior={'local':.3,'diversify':.2,'crossing':.05,'retime':.15}[action]
            success=(st['novel']*2+st['records']*8+st['feasible']*.3)/(t+5)
            priorities.append(prior+success+1.7*math.sqrt(math.log(alltries+1)/(t+1)))
        mode=actions[max(range(len(actions)),key=lambda i:priorities[i]+rng.random()*.12)]
        trials+=1;counts[mode]['tries']+=1
        # Diversity: sample non-record parent tours sometimes, but concentrate
        # the effort on the fastest pool members.
        pool.sort(key=lambda p:float(p['fitness'][0]))
        if rng.random()<.7:parent=rng.choice(pool[:min(4,len(pool))])
        else:parent=rng.choice(pool)
        sol=parent['sol']
        deadline=time.monotonic()+18.0 # bounded effort per proposed mutation
        try:
            if mode=='retime':
                proposed=try_retime(udp,c,sol,rng,deadline=deadline)
            else:
                proposed=try_mutation(udp,c,sol,axes,rng,mode,beam=6,deadline=deadline)
            if proposed is None:continue
            # Reject bad chronology before invoking ESA; this is only a cheap
            # screen, never the final validity authority.
            value=score(proposed)
            if not math.isfinite(value) or value>=min(udp.max_time,best_limit+12):continue
            hashid=fingerprint(proposed)
            novelty=hashid not in seen
            if not novelty and value>=min(p['fitness'][0] for p in pool)+1e-7:continue
            f=verify(udp,proposed)
            full_checks+=1;counts[mode]['feasible']+=1
            if novelty:
                counts[mode]['novel']+=1
                seen.add(hashid)
                pool.append({'sol':proposed,'fitness':f,'mode':mode})
                # Maintain a small elite + diverse library for escaping a local optimum.
                pool.sort(key=lambda p:p['fitness'][0])
                unique=[]; ids=set()
                for p in pool:
                    h=fingerprint(p['sol'])
                    if h not in ids:
                        unique.append(p);ids.add(h)
                pool=unique[:18]
            if f[0]<best_limit-1e-7:
                item={'solution':proposed,'fitness':f,'worker':wid,'mode':mode,
                      'created_unix':time.time(),'trial':trials}
                queue=rundir/'queue'
                atomic_json(queue/f'candidate_{wid}_{os.getpid()}_{trials}.json',item)
                counts[mode]['records']+=1
                best_limit=f[0]
                print(f'Worker {wid}: PROPOSED ESA-valid {f[0]:.9f} days via {mode}; '
                      f'trial={trials}',flush=True)
        except Exception as e:
            # Never kill the entire campaign for one bad numerical transfer.
            if trials%30==0:print(f'Worker {wid}: trial error {type(e).__name__}: {e}',flush=True)
        now=time.monotonic()
        if now-last_save>=90:
            atomic_json(checkpoint,{'version':1,'worker_id':wid,'trials':trials,
                'full_checks':full_checks,'counts':counts,'pool':pool,
                'random_state':rng_as_json(rng),'cache_calls':c.stats.calls,
                'cache_hits':c.stats.cache_hits,'updated_unix':time.time()})
            print(f'Worker {wid}: trials={trials}, verified={full_checks}, '
                  f'Lambert calls={c.stats.calls}, pool={len(pool)}, best={best_limit:.8f}',flush=True)
            last_save=now
    atomic_json(checkpoint,{'version':1,'worker_id':wid,'trials':trials,
                'full_checks':full_checks,'counts':counts,'pool':pool,
                'random_state':rng_as_json(rng),'cache_calls':c.stats.calls,
                'cache_hits':c.stats.cache_hits,'updated_unix':time.time()})
    print(f'Worker {wid}: safely stopped after {trials} trials',flush=True)


def process_candidates(rundir,udp,best):
    candidate_dir=rundir/'queue'
    for p in sorted(candidate_dir.glob('candidate_*.json'))[:30]:
        data=read_json(p)
        if not data:
            p.rename(p.with_suffix('.invalid'));continue
        try:
            sol=data['solution'];fitness=verify(udp,sol)
            if abs(fitness[0]-float(data['fitness'][0]))>1e-7:
                raise ValueError('Worker/supervisor fitness disagreement')
            if fitness[0] < best['score'] - 1e-7:
                # First save a permanent independently ESA-revalidated archive.
                name=f"record_{fitness[0]:.10f}_w{data['worker']}_{int(time.time())}.json"
                payload=solution_payload(sol,fitness,provenance={
                   'worker':data['worker'],'mode':data['mode'],'trial':data['trial'],
                   'independently_verified_by':'original ESA Medium evaluator (supervisor)'},udp=udp)
                atomic_json(rundir/'records'/name,payload)
                best={'score':fitness[0],'solution':sol,'fitness':fitness,
                      'record':str((rundir/'records'/name).relative_to(rundir)),
                      'updated_unix':time.time(),'worker':data['worker'],'mode':data['mode']}
                atomic_json(rundir/'best.json',best)
                print(f'NEW INDEPENDENTLY ESA-VALIDATED BEST {best["score"]:.10f} '
                      f'via {data["mode"]}, saved {name}',flush=True)
            p.unlink()
        except Exception as e:
            print(f'REJECTED proposed candidate {p.name}: {e}',flush=True)
            p.rename(p.with_suffix('.invalid'))
    return best


def campaign(args):
    global KEEP_GOING
    rundir=Path(args.directory).resolve();rundir.mkdir(parents=True,exist_ok=True)
    (rundir/'queue').mkdir(exist_ok=True);(rundir/'records').mkdir(exist_ok=True)
    stopfile=rundir/'stop.request'
    if stopfile.exists(): stopfile.unlink()
    udp=official_udp(ROOT)
    solution,fit=baseline(ROOT,udp)
    best=read_json(rundir/'best.json')
    if best:
        verify(udp,best['solution'])
    else:
        best={'score':fit[0],'solution':solution,'fitness':fit,
              'record':None,'updated_unix':time.time(),'mode':'ESA-validated incumbent'}
        atomic_json(rundir/'best.json',best)
    state_file=rundir/'campaign.json'
    state=read_json(state_file,{})
    active=float(state.get('active_elapsed_seconds',0))
    args.workers=max(1,min(2,int(args.workers))) # resource isolation
    requested=args.hours*3600
    print(f'MEDIUM ONLY: ESA-validated baseline {fit[0]:.9f}, saved best {best["score"]:.9f}',flush=True)
    print(f'Running with {args.workers} low-priority workers, remaining budget '
          f'{max(0,requested-active)/3600:.2f} hours; run={rundir}',flush=True)
    (rundir/'pid').write_text(str(os.getpid()))
    processes={}; logs={}; launches={}; failures={}
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'worker',
             '--directory',str(rundir),'--workers',str(args.workers),'--seed',str(args.seed)]
    env=os.environ.copy()
    env.update({'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
                'NUMEXPR_NUM_THREADS':'1','MPLCONFIGDIR':'/tmp'})
    start=time.monotonic();last_checkpoint=0.;heartbeat=time.monotonic()
    try:
        while KEEP_GOING and active+time.monotonic()-start < requested and not stopfile.exists():
            best=process_candidates(rundir,udp,best)
            for wid in range(args.workers):
                proc=processes.get(wid)
                if proc is None or proc.poll() is not None:
                    if proc is not None:
                        runtime=time.monotonic()-launches.get(wid,0)
                        failures[wid]=failures.get(wid,0)+1 if runtime<20 else 0
                        print(f'Worker {wid} exited code {proc.returncode}; '
                              f'last runtime={runtime:.1f}s; short failures={failures[wid]}',flush=True)
                        logs[wid].close()
                        if failures[wid]>=5:
                            print('ERROR: repeated worker crashes; stopping instead of restarting forever',flush=True)
                            KEEP_GOING=False
                            break
                    log=open(rundir/f'worker_{wid}.log','a',encoding='utf-8',buffering=1)
                    logs[wid]=log
                    processes[wid]=subprocess.Popen(command+['--worker-id',str(wid)],cwd=ROOT,
                          stdout=log,stderr=subprocess.STDOUT,env=env)
                    launches[wid]=time.monotonic()
                    print(f'Started worker {wid} pid={processes[wid].pid}',flush=True)
            now=time.monotonic()
            if now-last_checkpoint>20:
                state={'version':1,'seed':args.seed,'workers':args.workers,
                       'active_elapsed_seconds':active+now-start,'total_budget_seconds':requested,
                       'best_score':best['score'],'supervisor_pid':os.getpid(),
                       'worker_pids':{str(w):p.pid for w,p in processes.items()},
                       'heartbeat_unix':time.time()}
                atomic_json(state_file,state)
                last_checkpoint=now
            time.sleep(2)
        print('Supervisor: stopping or time budget reached; requesting graceful worker checkpoint',flush=True)
    finally:
        stopfile.touch()
        for proc in processes.values():
            if proc.poll() is None:proc.send_signal(signal.SIGTERM)
        for proc in processes.values():
            try:proc.wait(timeout=25)
            except subprocess.TimeoutExpired:
                proc.kill();proc.wait(timeout=10)
        for f in logs.values():f.close()
        best=process_candidates(rundir,udp,best)
        state={'version':1,'seed':args.seed,'workers':args.workers,
               'active_elapsed_seconds':active+time.monotonic()-start,
               'total_budget_seconds':requested,'best_score':best['score'],
               'supervisor_pid':None,'worker_pids':{},'heartbeat_unix':time.time(),
               'finished':active+time.monotonic()-start>=requested}
        atomic_json(state_file,state)
        (rundir/'pid').unlink(missing_ok=True)
        print(f'FINAL: best independently ESA-validated Medium {best["score"]:.9f} days; '
              f'improvement {fit[0]-best["score"]:.9f} days',flush=True)


def status(args):
    run=Path(args.directory)
    state=read_json(run/'campaign.json',{})
    best=read_json(run/'best.json',{})
    print('Medium campaign directory:',run)
    if not state and not best:
        print('No Medium campaign found yet.');return
    if best:
        print('Best ESA-validated saved:',best.get('score'),'days; record:',best.get('record'))
        print('Progress versus 202.8125:',round(202.8125-best.get('score',202.8125),9),'days')
    print('Active runtime:',round(state.get('active_elapsed_seconds',0)/3600,3),'hours')
    print('Last supervisor heartbeat:',state.get('heartbeat_unix'))
    print('Supervisor PID:',state.get('supervisor_pid'))
    print('Workers:',state.get('worker_pids',{}))
    for f in sorted(run.glob('worker_[0-9].json')):
        w=read_json(f,{})
        print(f.stem,'trials',w.get('trials'),'ESA-valid candidate checks',w.get('full_checks'),
              'Lambert calls',w.get('cache_calls'),'updated_unix',w.get('updated_unix'))
    print('Note: status is read from saved files; process liveness requires ps/pgrep.')


def verify_best(args):
    udp=official_udp(ROOT)
    sol,orig=baseline(ROOT,udp)
    state=read_json(Path(args.directory)/'best.json')
    if state and state.get('solution'):
        f=verify(udp,state['solution']);print('ORIGINAL ESA MEDIUM FITNESS:',f)
        print('IMPROVEMENT:',orig[0]-f[0],'days')
    else:print('No new record; original ESA-valid 202.8125 baseline persists.')


def main():
    ap=argparse.ArgumentParser()
    sub=ap.add_subparsers(dest='command',required=True)
    for cmd in ('run','worker','status','verify-best'):
        p=sub.add_parser(cmd)
        p.add_argument('--directory',default=str(DEFAULT_RUN))
        if cmd in ('run','worker'):
            p.add_argument('--hours',type=float,default=8.0)
            p.add_argument('--workers',type=int,default=2)
            p.add_argument('--seed',type=int,default=20261122)
        if cmd=='worker':p.add_argument('--worker-id',type=int,required=True)
    args=ap.parse_args()
    signal.signal(signal.SIGTERM,stop_handler)
    signal.signal(signal.SIGINT,stop_handler)
    if args.command=='run':campaign(args)
    elif args.command=='worker':worker(args)
    elif args.command=='status':status(args)
    else:verify_best(args)

if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

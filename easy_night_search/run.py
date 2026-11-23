"""Bounded local Easy search. Run from the repository root."""
import argparse, os, sys, json, time, random, math, signal, multiprocessing as mp
from pathlib import Path
from uuid import uuid4
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT)); os.chdir(ROOT)
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'

def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name('.'+p.name+'-'+uuid4().hex+'.tmp')
    with tmp.open('w') as f: json.dump(obj,f,allow_nan=False); f.flush(); os.fsync(f.fileno())
    import time
    for attempt in range(60):
        try:
            os.replace(tmp,p)
            break
        except PermissionError:
            if attempt == 59:
                raise
            time.sleep(0.5)

def load(p):
    d=json.loads(p.read_text()); return list(map(int,d['path'])),[tuple(x) for x in d['transfers']]

def worker(out, seconds, seed, incumbent):
    from global_dp_easy import Grid,evaluator
    from kttsp_esa_solver import validate,chromosome
    from focus_easy import mandatory
    from pareto_easy import BudgetExpired
    import numpy as np
    rng=random.Random(seed); module=evaluator(); udp=module.udp
    if udp.n_tomatoes!=49: raise RuntimeError('Not Easy')
    stop=time.monotonic()+seconds; out=Path(out); out.mkdir(parents=True,exist_ok=True)
    try: os.nice(10)
    except OSError: pass
    path,legs=load(Path(incumbent)); base=validate(udp,path,legs)
    if not base or not base['valid']: raise RuntimeError('Invalid seed')
    best=base['fitness'][0]; archive=[(best,path,legs)]; anchors={}; count=0; valid=0
    def learn(ls):
        for l in ls:
            k=tuple(l[:2]); a=anchors.setdefault(k,[]); d,t,c=l[2:5]; e=int(c>udp.dv_threshold)
            if any(q[2]>=d and q[2]+q[3]<=d+t and int(q[4]>udp.dv_threshold)<=e for q in a): continue
            a[:]=[q for q in a if not(d>=q[2] and d+t<=q[2]+q[3] and e<=int(q[4]>udp.dv_threshold))]; a.append(l)
    learn(legs)
    def submit(p,ls):
        nonlocal best,valid
        v=validate(udp,p,ls)
        if not v or not v['valid']: return
        valid+=1; score=v['fitness'][0]; learn(ls)
        if not any(p==q[1] and abs(score-q[0])<1e-7 for q in archive):
            archive.append((score,list(p),list(ls))); archive.sort(key=lambda q:q[0]); del archive[24:]
        if score<best-1e-8:
            best=score; save(out/f'record_{score:.12f}_{uuid4().hex[:8]}.json',dict(path=p,transfers=ls,chromosome=chromosome(p,ls),validation=v)); print('RECORD',score,flush=True)
    while time.monotonic()<stop:
        count+=1; _,p,ls=rng.choice(archive[:max(1,min(len(archive),8))]); p=p.copy()
        mode=count%5; i,j=sorted(rng.sample(range(49),2))
        if mode==0: p[i:j+1]=reversed(p[i:j+1])
        elif mode==1: p.insert(j,p.pop(i))
        elif mode==2: p[i],p[j]=p[j],p[i]
        elif mode==3:
            n=min(rng.randint(2,6),49-i); block=p[i:i+n]; del p[i:i+n]; k=rng.randrange(len(p)+1); p[k:k]=block
        # mode 4 retimes unchanged order with a new lattice phase.
        step=rng.choice([.125,.1875,.25,.3125]); grid=Grid(module,step,rng.random()*step,min(stop,time.monotonic()+90),directional=True)
        pairs=list(zip(p,p[1:])); required=[mandatory(grid,*a) for a in pairs]; budget=udp.n_dv_exception
        if sum(required)>budget: continue
        remaining=[sum(required[k:]) for k in range(49)]
        # Keep multiple readiness labels per budget, not just earliest sampled arrival.
        states={0:[(0.,())]}
        try:
            for k,(a,b) in enumerate(pairs):
                nxt={}; grid.check()
                for used,labels in states.items():
                    for day,sched in labels:
                        allow=used+1+remaining[k+1]<=budget
                        options=list(anchors.get((a,b),[]))
                        options+=list(grid.edges(a,b,day, rng.choice([4.,7.,10.]),rng.choice([8.,12.]),allow))
                        for l in options:
                            if l is None or l[2]<day-1e-8: continue
                            extra=int(l[4]>udp.dv_threshold); end=l[2]+l[3]; u=used+extra
                            if u+remaining[k+1]>budget or end>udp.max_time: continue
                            nxt.setdefault(u,[]).append((end,sched+(l,)))
                states={}
                for u,labels in nxt.items():
                    bins={}
                    for day,sched in sorted(labels,key=lambda x:x[0]): bins.setdefault(int(day/.35),(day,sched))
                    states[u]=list(bins.values())[:3]
                if not states: break
            else:
                for labels in states.values():
                    for _,sched in labels: submit(p,sched)
        except BudgetExpired: pass
        save(out/'status.json',dict(best=best,attempts=count,valid=valid,seed=seed))
    print('DONE',best,count,valid,flush=True)

def main():
    from global_dp_easy import evaluator
    from kttsp_esa_solver import validate
    ap=argparse.ArgumentParser(); ap.add_argument('--incumbent',type=Path); ap.add_argument('--hours',type=float,default=8); ap.add_argument('--workers',type=int,default=1); ap.add_argument('--output',type=Path,default=Path('results/esa/easy_night_search')); ap.add_argument('--seed',type=int,default=83191); a=ap.parse_args()
    if not 0<a.hours<=24 or not 1<=a.workers<=4: ap.error('hours 0–24, workers 1–4')
    out=a.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(out/'lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    udp=evaluator().udp; best=math.inf; seedfile=None
    # Rank saved complete schedules by actual final arrival, then verify with ESA.
    candidates=[]
    for f in ([a.incumbent] if a.incumbent else Path('results/esa').rglob('*.json')):
        try:
            if f.stat().st_size>2_000_000: continue
            p,l=load(f)
            if len(p)==49 and len(l)==48: candidates.append((l[-1][2]+l[-1][3],f))
        except (OSError,ValueError,KeyError,TypeError,IndexError): pass
    for _,f in sorted(candidates,key=lambda x:x[0]):
        p,l=load(f); v=validate(udp,p,l)
        if v and v['valid']:
            best=v['fitness'][0]; seedfile=f.resolve(); break
    if seedfile is None: raise SystemExit('No valid seed found. Supply --incumbent PATH to a saved path/transfers JSON.')
    print('START',best,seedfile,flush=True); end=time.monotonic()+a.hours*3600; batch=0; jobs=[]
    def stop(*_): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:
        while time.monotonic()<end:
            batch+=1; duration=min(480,end-time.monotonic()); jobs=[]
            for w in range(a.workers):
                folder=out/f'batch_{batch}_{w}_{uuid4().hex[:6]}'; proc=mp.Process(target=worker,args=(folder,duration,a.seed+batch*1009+w,str(seedfile))); proc.start(); jobs.append((proc,folder))
            for proc,_ in jobs: proc.join(max(0,end-time.monotonic()+5))
            for proc,folder in jobs:
                if proc.is_alive(): proc.terminate(); proc.join(5)
                for f in folder.glob('record_*.json'):
                    p,l=load(f); v=validate(udp,p,l)
                    if v and v['valid'] and v['fitness'][0]<best:
                        best=v['fitness'][0]; seedfile=f; print('VERIFIED',best,f,flush=True)
            save(out/'best.json',dict(best_days=best,record=str(seedfile),batch=batch)); print('BATCH',batch,best,flush=True)
            if all(proc.exitcode not in (0,None) for proc,_ in jobs): raise RuntimeError('All workers failed; inspect terminal output')
    except KeyboardInterrupt: pass
    finally:
        for proc,_ in jobs:
            if proc.is_alive(): proc.terminate(); proc.join(5)
        print('END',best,seedfile,flush=True)
if __name__=='__main__': main()

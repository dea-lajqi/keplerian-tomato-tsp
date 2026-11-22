#!/usr/bin/env python3
"""MEDIUM V4: adaptive, anchored, joint route-and-time local dynamic programming.

Run from the root of your repository. Only writes under --directory.
PyKEP 2.6.x required for ESA validation (provided by WSL kttsp-medium).
"""
from __future__ import annotations
import argparse, json, math, os, random, signal, sys, time, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from medium_overnight.validation import official_udp, baseline, verify, solution_payload
from medium_overnight.search import DVCache, try_retime
from medium_v4.joint import joint_window,window_choices,suffix_polish,terminal_crossing

STOP=False
DEFAULT_RUN=ROOT/'results/esa/medium_v4/run_01'

def signal_stop(*_):
    global STOP
    STOP=True

def atomic_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    t=path.with_name('.'+path.name+'.'+str(os.getpid())+'.tmp')
    with t.open('w') as fd:
        json.dump(data,fd,indent=2,ensure_ascii=False,allow_nan=False)
        fd.flush();os.fsync(fd.fileno())
    os.replace(t,path)

def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return default

def unpack(item):
    if not isinstance(item,dict):return None
    sol=item.get('solution',item.get('sol',item))
    return sol if isinstance(sol,dict) and all(k in sol for k in ('path','dep','tof')) else None

def signature(sol):
    return hashlib.sha256(json.dumps(sol['path'],separators=(',',':')).encode()).hexdigest()[:20]

def rngrestore(v):return tuple(rngrestore(x) for x in v) if isinstance(v,list) else v

def seeds(udp,run):
    allfiles=[ROOT/'results/esa/medium_overnight/run_20261122/best.json',
              ROOT/'results/esa/medium_v2/run_01/best.json',
              ROOT/'results/esa/medium_v3/run_01/best.json',
              ROOT/'results/esa/medium_v4/smoke/best.json',
              run/'best.json']
    base,f=baseline(ROOT,udp)
    found=[(base,f,'Original ESA baseline')]
    for fn in allfiles:
        sol=unpack(read(fn))
        if sol:
            try:found.append((sol,verify(udp,sol),str(fn)))
            except Exception as e:print('REJECTED seed',fn,type(e).__name__,str(e)[:100],flush=True)
    best=min(found,key=lambda x:x[1][0])
    return best,found

def diversify(pool,best_slack=3.0,keep=32):
    by_order={}
    for p in pool:
        if p['fitness'][0]>min(t['fitness'][0] for t in pool)+best_slack:continue
        k=signature(p['solution'])
        if k not in by_order or p['fitness'][0]<by_order[k]['fitness'][0]:by_order[k]=p
    return sorted(by_order.values(),key=lambda v:v['fitness'][0])[:keep]

def run(args):
    global STOP
    STOP=False
    signal.signal(signal.SIGTERM,signal_stop);signal.signal(signal.SIGINT,signal_stop)
    try:os.nice(12)
    except (OSError,AttributeError):pass
    try:
        if hasattr(os,'sched_getaffinity') and len(os.sched_getaffinity(0))>=5:
            cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpus[-2]})
    except (OSError,PermissionError):pass
    folder=args.directory.resolve();folder.mkdir(parents=True,exist_ok=True)
    pidfile=folder/'pid';stopfile=folder/'stop.request'
    if pidfile.exists():
        try:
            pid=int(pidfile.read_text())
            os.kill(pid,0)
            raise SystemExit(f'Already running PID={pid} in {folder}; will NOT launch duplicate')
        except ProcessLookupError:pass
        except ValueError:pass
    stopfile.unlink(missing_ok=True)
    pidfile.write_text(str(os.getpid()))
    try:
        udp=official_udp(ROOT)
        (inc_sol,inc_fit,inc_from),source=seeds(udp,folder)
        print('MEDIUM V4 independently verified ESA seed',inc_fit,'from',inc_from,flush=True)
        if len(inc_sol['path'])!=181:raise ValueError('Not 181 tomatoes')
        checkpoint=read(folder/'checkpoint.json',{})
        rng=random.Random(args.seed)
        if 'rng' in checkpoint:rng.setstate(rngrestore(checkpoint['rng']))
        stats=checkpoint.get('stats',{'trials':0,'esa_checks':0,'complete':0,'novel':0,'records':0,'calls':0,
             'by_mode':{},'failures':{},'last_best_at':0})
        modes=['joint_reorder','joint_phase','endgame','terminal_crossing','v1_retime']
        for name in modes:stats['by_mode'].setdefault(name,{'tries':0,'complete':0,'novel':0,'records':0})
        pool=[{'solution':s,'fitness':f,'source':where} for s,f,where in source]
        # Import selected ESA-valid alternatives from prior V1 worker checkpoint libraries.
        if not checkpoint:
            for w in (0,1):
                src=read(ROOT/f'results/esa/medium_overnight/run_20261122/worker_{w}.json',{})
                for entry in src.get('pool',[])[:24]:
                    s=unpack(entry)
                    if s:
                        try:
                            fitness=verify(udp,s)
                            if fitness[0]<=inc_fit[0]+3.0:pool.append({'solution':s,'fitness':fitness,'source':'V1 library'})
                        except Exception:pass
        for item in checkpoint.get('pool',[]):
            s=unpack(item)
            if s:
                try:pool.append({'solution':s,'fitness':verify(udp,s),'source':'V4 resumed'})
                except Exception:pass
        pool=diversify(pool)
        best=min(pool,key=lambda x:x['fitness'][0])
        if best['fitness'][0]<inc_fit[0]-1e-9:
            inc_sol,inc_fit=best['solution'],best['fitness']
        record=read(folder/'best.json')
        if not record or record.get('score',1e9)>inc_fit[0]+1e-9:
            atomic_json(folder/'best.json',{'solution':inc_sol,'score':inc_fit[0],
                        'fitness':inc_fit,'source':inc_from,'updated':time.time()})
        cache=DVCache(udp,limit=160000)
        lastsave=0;start=time.monotonic();oldtime=float(checkpoint.get('elapsed',0))
        print(f'MEDIUM V4 started: {args.minutes:g} minute total budget, 1 low-priority CPU, '
              f'pool={len(pool)}, {args.seed=} directory={folder}',flush=True)
        while not STOP and not stopfile.exists() and oldtime+time.monotonic()-start < args.minutes*60:
            if args.trials and stats['trials']>=args.trials:break
            now=time.monotonic()
            if now-lastsave>args.checkpoint_seconds:
                stats['calls']=cache.stats.calls
                atomic_json(folder/'checkpoint.json',{'version':1,'stats':stats,'pool':pool,
                            'rng':rng.getstate(),'elapsed':oldtime+now-start,'updated':time.time()})
                print(f"CHECKPOINT trials={stats['trials']} valid_full={stats['complete']} "
                      f"novel={stats['novel']} ESA={stats['esa_checks']} Lambert={cache.stats.calls} "
                      f"best={inc_fit[0]:.10f}",flush=True)
                lastsave=now
            total=1+sum(stats['by_mode'][x]['tries'] for x in modes)
            def merit(x):
                z=stats['by_mode'][x]
                return (z['records']*10+z['novel']*.4+z['complete']*.15)/(z['tries']+5)+\
                       .9*math.sqrt(math.log(total+1)/(z['tries']+1))
            mode=max(modes,key=lambda x:merit(x)+rng.random()*.20)
            stats['trials']+=1;stats['by_mode'][mode]['tries']+=1
            parent=rng.choice(pool[:min(4,len(pool))]) if rng.random()<.75 else rng.choice(pool)
            sol=parent['solution']; deadline=min(time.monotonic()+args.trial_seconds,
                                                 start+args.minutes*60-oldtime)
            candidates=[]
            try:
                if mode=='terminal_crossing':
                    c=terminal_crossing(udp,cache,sol,rng,deadline=deadline)
                    if c:candidates.append((c,{'reordered':False,'window':None}))
                elif mode=='v1_retime':
                    c=try_retime(udp,cache,sol,rng,deadline=deadline)
                    if c:candidates.append((c,{'reordered':False,'window':None}))
                else:
                    win=window_choices(sol,udp,rng,(2,4) if mode=='joint_reorder' else (2,5),
                            .9 if mode=='endgame' else .25)
                    if not win:continue
                    l,r=win
                    candidates=joint_window(udp,cache,sol,l,r,rng,beam=args.beam,
                         call_cap=args.call_cap,deadline=deadline,
                         reorder=(mode!='joint_phase'),
                         mode='fine' if mode=='endgame' else 'broad',max_results=8)
                if not candidates:continue
                # All candidates are complete tours with unchanged outside edges.
                stats['complete']+=len(candidates)
                stats['by_mode'][mode]['complete']+=len(candidates)
                for c,info in candidates[:args.esa_checks_per_trial]:
                    if STOP or time.monotonic()>deadline:break
                    if c['dep'][-1]+c['tof'][-1] > inc_fit[0]+args.pool_slack:continue
                    f=verify(udp,c)
                    stats['esa_checks']+=1
                    k=signature(c)
                    known={signature(x['solution']) for x in pool}
                    novelty=k not in known
                    if novelty:
                        stats['novel']+=1;stats['by_mode'][mode]['novel']+=1
                        pool.append({'solution':c,'fitness':f,'source':mode})
                    else:
                        for p in pool:
                            if signature(p['solution'])==k and f[0]<p['fitness'][0]-1e-9:
                                p.update(solution=c,fitness=f,source=mode)
                                break
                    if f[0]<inc_fit[0]-1e-8:
                        inc_sol,inc_fit=c,f
                        rname=f'record_{f[0]:.10f}_trial_{stats["trials"]:07d}.json'
                        record_path=folder/'records'/rname
                        atomic_json(record_path,solution_payload(c,f,udp=udp,provenance={
                            'method':mode,'trial':stats['trials'],'source':'ESA medium original evaluator'}))
                        atomic_json(folder/'best.json',{'solution':c,'fitness':f,'score':f[0],
                            'source':str(record_path),'updated':time.time()})
                        stats['records']+=1;stats['by_mode'][mode]['records']+=1
                        stats['last_best_at']=oldtime+time.monotonic()-start
                        print(f'NEW ESA-VALID MEDIUM V4 {f[0]:.10f} via {mode}',flush=True)
                    # High-value variant: attempt to pull the entire suffix earlier
                    # using the validated parent candidate; no unvalidated claims.
                    if info.get('reordered') and rng.random()<args.suffix_probability and time.monotonic()<deadline:
                        anchor=info['window'][1]
                        polished=suffix_polish(udp,cache,c,rng,anchor,deadline=deadline)
                        if polished and polished['dep'][-1]+polished['tof'][-1]<inc_fit[0]-1e-8:
                            ff=verify(udp,polished)
                            stats['esa_checks']+=1
                            if ff[0]<inc_fit[0]-1e-8:
                                inc_sol,inc_fit=polished,ff
                                fn=folder/'records'/f'record_{ff[0]:.10f}_trial_{stats["trials"]:07d}_suffix.json'
                                atomic_json(fn,solution_payload(polished,ff,udp=udp,provenance={'method':'joint+suffix','trial':stats['trials']}))
                                atomic_json(folder/'best.json',{'solution':polished,'fitness':ff,
                                             'score':ff[0],'source':str(fn),'updated':time.time()})
                                stats['records']+=1;stats['by_mode'][mode]['records']+=1
                                print(f'NEW ESA-VALID MEDIUM V4 {ff[0]:.10f} via joint+suffix',flush=True)
                pool=diversify(pool,best_slack=args.pool_slack)
            except Exception as exc:
                name=type(exc).__name__+': '+str(exc)[:100]
                stats['failures'][name]=stats['failures'].get(name,0)+1
                if stats['failures'][name]<=2:print('candidate rejected:',name,flush=True)
        print(f"FINISHED MEDIUM V4: {inc_fit[0]:.10f} days, trials={stats['trials']}, "
              f"new_records={stats['records']}, complete={stats['complete']}",flush=True)
    finally:
        if 'stats' in locals() and 'rng' in locals():
            stats['calls']=cache.stats.calls if 'cache' in locals() else 0
            atomic_json(folder/'checkpoint.json',{'version':1,'stats':stats,'pool':pool,
                        'rng':rng.getstate(),'elapsed':oldtime+time.monotonic()-start,
                        'updated':time.time()})
        pidfile.unlink(missing_ok=True)

def status(args):
    folder=args.directory
    best=read(folder/'best.json',{})
    ck=read(folder/'checkpoint.json',{})
    if not best:return print('No run found:',folder)
    st=ck.get('stats',{})
    print('ESA-valid best:',best.get('score'),'source:',best.get('source'))
    print('Runtime minutes:',round(ck.get('elapsed',0)/60,2),'trials:',st.get('trials'),
          'complete:',st.get('complete'),'new orders:',st.get('novel'),
          'ESA checks:',st.get('esa_checks'),'Lambert:',st.get('calls'),
          'records:',st.get('records'))
    for method,v in st.get('by_mode',{}).items():print(method,v)
    pidfile=folder/'pid'
    if pidfile.exists():
        pid=int(pidfile.read_text())
        try:os.kill(pid,0);print('PID alive:',pid)
        except ProcessLookupError:print('PID not active:',pid)
    else:print('No active PID file')

def check_best(args):
    udp=official_udp(ROOT)
    (best,fitness,src),_all=seeds(udp,args.directory)
    print('INDEPENDENT ORIGINAL ESA FITNESS:',fitness,'FROM:',src)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['run','status','verify-best'])
    p.add_argument('--directory',type=Path,default=DEFAULT_RUN)
    p.add_argument('--minutes',type=float,default=90)
    p.add_argument('--seed',type=int,default=20261122)
    p.add_argument('--trials',type=int,default=0)
    p.add_argument('--beam',type=int,default=12)
    p.add_argument('--call-cap',type=int,default=15000)
    p.add_argument('--trial-seconds',type=float,default=10)
    p.add_argument('--checkpoint-seconds',type=float,default=60)
    p.add_argument('--esa-checks-per-trial',type=int,default=4)
    p.add_argument('--pool-slack',type=float,default=2.5)
    p.add_argument('--suffix-probability',type=float,default=.3)
    args=p.parse_args()
    if args.command=='status':status(args)
    elif args.command=='verify-best':check_best(args)
    else:
        if args.minutes<=0 or args.minutes>1200:p.error('minutes must be (0,1200]')
        run(args)

if __name__=='__main__':main()

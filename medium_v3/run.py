#!/usr/bin/env python3
"""Medium V3: guided route repair + exact ESA timing + optional continuous smoothing.

Only Medium. Never edits the original instances, evaluator or incumbent.
Uses existing medium_overnight.validation and search utilities from Medium V1.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, signal, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
from medium_overnight.validation import official_udp, verify, baseline, solution_payload
from medium_overnight.search import DVCache, greedy_suffix
from topology import candidate_moves, long_transfer_analysis
from timed_search import repair_window, join_candidates
from continuous import smooth_segment

DEFAULT=ROOT/'results/esa/medium_v3/run_01'
SOURCES=[ROOT/'results/esa/medium_overnight/run_20261122/best.json',
         ROOT/'results/esa/medium_v2/run_01/best.json']
STOP=False

def stop(*_):
    global STOP
    STOP=True

def atomic_write(path, x):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name('.'+p.name+f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return default

def rngrestore(x):
    return tuple(rngrestore(y) for y in x) if isinstance(x,list) else x

def fingerprint(sol):
    return hashlib.sha256(json.dumps(sol['path'],separators=(',',':')).encode()).hexdigest()[:20]

def classify_source(d):
    if d is None:return None
    sol=d.get('solution',d) if isinstance(d,dict) else None
    return sol if isinstance(sol,dict) and all(k in sol for k in ('path','dep','tof')) else None

def seed_from_files(udp,run=None):
    base,bfit=baseline(ROOT,udp)
    best=(base,bfit,'ESA original 202.8125')
    for p in SOURCES+([Path(run)/'best.json'] if run else []):
        sol=classify_source(read(p))
        if sol:
            try:
                f=verify(udp,sol)
                if f[0]<best[1][0]-1e-9:best=(sol,f,str(p))
            except Exception as e:
                print('WARNING rejected seed from',p,':',e,flush=True)
    return best

def stats_checkpoint(folder, rng, stats, pool, best, elapsed):
    atomic_write(folder/'checkpoint.json',{
      'version':1,'stats':stats,'pool':pool[:12],
      'best':best['score'],'random_state':rng.getstate(),
      'elapsed_seconds':elapsed,'updated_unix':time.time()})

def candidate_check(udp, sol, limit):
    if len(sol['path'])!=181 or len(sol['dep'])!=180 or len(sol['tof'])!=180:
        return None
    if sol['dep'][-1]+sol['tof'][-1]>limit+1e-9:return None
    return verify(udp,sol)

def run(args):
    global STOP
    STOP=False
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    try:os.nice(12)
    except (AttributeError,PermissionError,OSError):pass
    if hasattr(os,'sched_getaffinity'):
        try:
            cpus=sorted(os.sched_getaffinity(0))
            if len(cpus)>5:os.sched_setaffinity(0,{cpus[-3]})
        except (OSError,PermissionError):pass
    folder=args.directory.resolve();folder.mkdir(parents=True,exist_ok=True)
    stopper=folder/'stop.request'
    if stopper.exists():stopper.unlink()
    (folder/'pid').write_text(str(os.getpid()))
    udp=official_udp(ROOT)
    source,source_fit,source_path=seed_from_files(udp,folder)
    print('MEDIUM V3 only: independent ESA validated seed',source_fit,
          'from',source_path,flush=True)
    if len(source['path'])!=181:raise ValueError('Medium 181 tomatoes required')
    els=[[float(z) for z in row] for row in udp.tomato_orbital_parameters]
    print('Orbital diagnostics:',long_transfer_analysis(source,els),flush=True)
    state=read(folder/'checkpoint.json',{})
    rng=random.Random(args.seed)
    if state.get('random_state'):
        rng.setstate(rngrestore(state['random_state']))
    stats=state.get('stats',{'trials':0,'complete':0,'esa_checked':0,'different_orders':0,
                              'records':0,'modes':{},'failures':{},'lambert':0})
    pool=[];seen=set()
    for item in state.get('pool',[]):
        sol=classify_source(item)
        if not sol:continue
        try:
            f=verify(udp,sol)
            k=fingerprint(sol)
            if f[0]<=source_fit[0]+2.0 and k not in seen:
                pool.append({'solution':sol,'score':f[0]});seen.add(k)
        except Exception:pass
    key=fingerprint(source)
    if key not in seen:
        pool.append({'solution':source,'score':source_fit[0]});seen.add(key)
    best={'solution':source,'score':source_fit[0],'fitness':source_fit,
          'source':source_path,'updated_unix':time.time()}
    atomic_write(folder/'best.json',best)
    cache=DVCache(udp,limit=240000)
    start=time.monotonic();elapsed_old=float(state.get('elapsed_seconds',0.))
    last_save=0.;modes=['bottleneck_insert','adjacent','small_reverse','near_plane_swap','block_shuffle','endgame','continuous']
    for m in modes:stats['modes'].setdefault(m,{'tries':0,'complete':0,'records':0,'novel':0})
    duration=float(args.minutes)*60
    print(f'Medium V3: {args.minutes:g}-minute budget; 1 low-priority CPU; '
          f'run={folder}',flush=True)
    try:
        while not STOP and not stopper.exists() and elapsed_old+time.monotonic()-start<duration:
            if args.trials and stats['trials']>=args.trials:break
            now=time.monotonic()
            if now-last_save>=args.checkpoint_seconds:
                stats['lambert']=cache.stats.calls
                stats_checkpoint(folder,rng,stats,pool,best,elapsed_old+now-start)
                print(f'checkpoint trials={stats["trials"]} complete={stats["complete"]} '
                      f'ESAchecks={stats["esa_checked"]} Lambert={cache.stats.calls} '
                      f'best={best["score"]:.9f}',flush=True)
                last_save=now
            stats['trials']+=1
            if rng.random()<.15:mode=rng.choice(modes)
            else:
                # Rewards successful complete routes and novel visits, not
                # unproductive raw Lambert counts. Explore each mode sometimes.
                alltries=1+sum(stats['modes'][x]['tries'] for x in modes)
                def merit(m):
                    q=stats['modes'][m]
                    return (q['records']*9+q['novel']*.6+q['complete']*.2)/(q['tries']+4)+\
                            1.3*math.sqrt(math.log(alltries+2)/(q['tries']+1))
                mode=max(modes,key=lambda m:merit(m)+rng.random()*.15)
            stats['modes'][mode]['tries']+=1
            pool.sort(key=lambda item:item['score'])
            p=rng.choice(pool[:min(len(pool),4)]) if rng.random()<.7 else rng.choice(pool)
            sol=p['solution']; deadline=time.monotonic()+args.trial_seconds
            try:
                attempts=[]
                if mode=='continuous':
                    # First polish the LAST few legs, which can directly lower
                    # mission completion time without requiring suffix repair.
                    right=len(sol['dep'])
                    left=max(0,right-rng.randint(2,5))
                    tmp,msg=smooth_segment(udp,sol,left,right,maxiter=args.slsqp_iters)
                    if tmp:attempts.append(tmp)
                else:
                    moves=candidate_moves(sol,els,rng,count=5,mode=mode)
                    for mv in moves[:2]:
                        if time.monotonic()>=deadline:break
                        partial=repair_window(udp,cache,sol,mv.route,mv.left,mv.right,rng,
                              beam=args.beam,max_calls=args.max_calls,deadline=deadline)
                        if not partial:continue
                        attempts.extend(join_candidates(udp,cache,sol,mv.route,mv.right,
                              partial,rng,deadline=deadline,max_suffix=2))
                        if attempts:break
                for candidate in attempts:
                    if time.monotonic()>deadline:break
                    if candidate['dep'][-1]+candidate['tof'][-1]>=min(udp.max_time,best['score']+args.pool_slack):
                        continue
                    f=candidate_check(udp,candidate,best['score']+args.pool_slack)
                    if not f:continue
                    stats['complete']+=1;stats['esa_checked']+=1
                    h=fingerprint(candidate)
                    if h not in seen:
                        stats['different_orders']+=1;stats['modes'][mode]['novel']+=1
                        seen.add(h)
                        pool.append({'solution':candidate,'score':f[0]})
                    else:
                        # Same order can have a strictly faster timing schedule.
                        for item in pool:
                            if fingerprint(item['solution'])==h and f[0]<item['score']-1e-9:
                                item.update(solution=candidate,score=f[0])
                                break
                    # Keep the elite, plus alternative verified orders.
                    pool=sorted(pool,key=lambda x:x['score'])[:20]
                    stats['modes'][mode]['complete']+=1
                    if f[0]<best['score']-1e-8:
                        newrec=folder/'records'/f'record_{f[0]:.10f}_{stats["trials"]:07d}.json'
                        payload=solution_payload(candidate,f,udp=udp,provenance={
                          'optimizer':'medium_v3','mode':mode,'trial':stats['trials'],
                          'original_ESA_evaluator':True})
                        atomic_write(newrec,payload)
                        best={'solution':candidate,'score':f[0],'fitness':f,'source':str(newrec),
                              'record':str(newrec.relative_to(folder)),'updated_unix':time.time()}
                        atomic_write(folder/'best.json',best)
                        stats['records']+=1;stats['modes'][mode]['records']+=1
                        print(f'NEW INDEPENDENT ESA MEDIUM V3 {f[0]:.10f} days '
                              f'via {mode}, trial {stats["trials"]}',flush=True)
            except Exception as e:
                why=f'{type(e).__name__}: {str(e)[:120]}'
                stats['failures'][why]=stats['failures'].get(why,0)+1
                if stats['failures'][why]==1:print('WARNING candidate:',why,flush=True)
    finally:
        stats['lambert']=cache.stats.calls
        stats_checkpoint(folder,rng,stats,pool,best,elapsed_old+time.monotonic()-start)
        print(f'FINAL MEDIUM V3 best independently ESA-valid {best["score"]:.10f} days; '
              f'trials={stats["trials"]}; new_records={stats["records"]}',flush=True)
        (folder/'pid').unlink(missing_ok=True)

def show_status(args):
    a=read(args.directory/'best.json',{});b=read(args.directory/'checkpoint.json',{})
    if not a:return print('No Medium V3 run found in',args.directory)
    print('Best ESA-validated Medium V3:',a.get('score'),'days; source:',a.get('source'))
    print('Trials:',b.get('stats',{}).get('trials'),'complete:',b.get('stats',{}).get('complete'),
          'novel orders:',b.get('stats',{}).get('different_orders'),
          'Lambert:',b.get('stats',{}).get('lambert'))
    print('Elapsed seconds:',b.get('elapsed_seconds'))
    print('Per-method:',json.dumps(b.get('stats',{}).get('modes'),indent=2))
    pidfile=args.directory/'pid'
    if pidfile.exists():
        pid=int(pidfile.read_text())
        try:os.kill(pid,0);print('Process alive:',pid)
        except ProcessLookupError:print('Process stopped; stale PID file:',pid)
    else:print('No active PID file')

def do_verify(args):
    udp=official_udp(ROOT)
    sol,f,where=seed_from_files(udp,args.directory)
    print('Original ESA Medium fitness:',f,'source:',where)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['run','status','verify-best'])
    p.add_argument('--directory',type=Path,default=DEFAULT)
    p.add_argument('--minutes',type=float,default=45)
    p.add_argument('--seed',type=int,default=20261122)
    p.add_argument('--beam',type=int,default=8)
    p.add_argument('--max-calls',type=int,default=10000)
    p.add_argument('--trial-seconds',type=float,default=20)
    p.add_argument('--checkpoint-seconds',type=float,default=60)
    p.add_argument('--pool-slack',type=float,default=2.0)
    p.add_argument('--slsqp-iters',type=int,default=25)
    p.add_argument('--trials',type=int,default=0)
    args=p.parse_args()
    if args.command=='status':show_status(args)
    elif args.command=='verify-best':do_verify(args)
    else:
        if args.minutes<=0 or args.minutes>600:p.error('minutes must be between 0 and 600')
        run(args)
if __name__=='__main__':main()

"""Bounded low-priority portfolio for the Small two-window repair experiment."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from continuous_windows_easy import SEED
from global_dp_easy import evaluator
from kttsp_esa_solver import validate
from small_v2 import atomic_json


def run(args):
    folder=args.output.resolve()
    folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/"lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    file=folder/"state.json"
    state=json.loads(file.read_text()) if file.exists() else {
        "next":0,"elapsed":0.,"history":[],"active":[],"orders":[],"novel":[],
        "best_path":str(SEED),"best":111.75601410426336,"seed":args.seed,"records":[],
        "checkpoint_by_lane":{},"last_gain_elapsed":0.,"processed_records":[]}
    udp=evaluator().udp
    init=json.loads(Path(state["best_path"]).read_text())
    verified=validate(udp,init["path"],init["transfers"])
    if not verified or not verified["valid"]:
        raise ValueError("ESA rejected campaign start")
    state["best"]=verified["fitness"][0]
    start=time.monotonic()
    previous=state["elapsed"]
    stop_at=start+max(0.,args.seconds-previous)
    jobs={}
    orders={tuple(p) for p in state["orders"]}
    novel={tuple(p) for p in state["novel"]}
    processed=set(state.get("processed_records",[]))
    diverse={tuple(p) for p in state.get("diverse_orders",[init["path"]])}
    diversity_edges=[set(zip(p,p[1:])) for p in sorted(diverse)]
    stopped=False
    pending={row["lane"]:row for row in state["active"]}
    def intercept(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,intercept)
    signal.signal(signal.SIGINT,intercept)

    def save(status):
        state.update(status=status,pid=os.getpid(),active=[x[0] for x in jobs.values()]+list(pending.values()),
                     utc=datetime.now(timezone.utc).isoformat(),elapsed=previous+time.monotonic()-start,
                     orders=sorted(orders),novel=sorted(novel),processed_records=sorted(processed),
                     diverse_orders=sorted(diverse))
        atomic_json(file,state)

    def latest(directory):
        return max(directory.glob("*checkpoint.json"),key=lambda p:p.stat().st_mtime_ns,default=None)

    def harvest(row,code=None):
        directory=Path(row["directory"])
        for f in sorted(directory.glob("easy_v2_*.json"),key=lambda p:p.stat().st_mtime_ns):
            if str(f) in processed:
                continue
            raw=json.loads(f.read_text())
            result=validate(udp,raw["path"],raw["transfers"])
            if result and result["valid"] and result["fitness"][0]<state["best"]:
                value=result["fitness"][0]
                dest=folder/f"record_{len(state['records'])+1:04d}.json"
                atomic_json(dest,{**raw,"validation":result,"worker_record":str(f)})
                state["best"],state["best_path"]=value,str(dest)
                state["records"].append(str(dest))
                state["last_gain_elapsed"]=previous+time.monotonic()-start
                print("independent ESA record",value,dest,flush=True)
            processed.add(str(f))
        if code is None:
            return
        checkpoint=latest(directory)
        if checkpoint:
            data=json.loads(checkpoint.read_text())
            state["checkpoint_by_lane"][str(row["lane"])]=str(checkpoint)
            old=len(novel)
            orders.update(map(tuple,data["orders"]))
            newcomers=sorted(set(map(tuple,data["novel_orders"]))-novel)
            novel.update(newcomers)
            substantially_different=0
            for p in newcomers:
                edges=set(zip(p,p[1:]))
                if all(len(edges-ref)>=5 for ref in diversity_edges):
                    diverse.add(p)
                    diversity_edges.append(edges)
                    substantially_different+=1
            row["diversity_representatives_added"]=substantially_different
            row["counts"]=data["stats"]
            row["worker_seconds"]=data["elapsed_seconds"]
            row["best_generated"]=data["best_generated"]
            row["new_global_orders"]=len(novel)-old
            if substantially_different:
                state["last_gain_elapsed"]=previous+time.monotonic()-start
        state["history"].append({**row,"exit":code})
        print("episode",row["number"],"exit",code,row.get("counts",{}),flush=True)

    status="budget finished"
    try:
        save("running")
        while not stopped and time.monotonic()+3.<stop_at and state["best"]>101.:
            for lane,(row,p) in list(jobs.items()):
                code=p.poll()
                harvest(row,code)
                if code is not None:
                    del jobs[lane]
            if len(state["history"])>=5 and previous+time.monotonic()-start-state["last_gain_elapsed"]>1200:
                status="diversification plateau"
                break
            for lane in range(args.workers):
                if lane in jobs:
                    continue
                if time.monotonic()+5.>=stop_at:
                    break
                number=state["next"]
                state["next"]+=1
                old=pending.pop(lane,None)
                from_checkpoint=latest(Path(old["directory"])) if old else None
                if old and old.get("pid"):
                    try:
                        cmdline=Path(f"/proc/{old['pid']}/cmdline").read_bytes().split(b"\0")
                        if os.fsencode(old["directory"]) in cmdline and any(x.endswith(b"/small_v2.py") for x in cmdline):
                            os.kill(old["pid"],signal.SIGTERM)
                    except OSError:
                        pass
                directory=folder/f"episode_{number:04d}"
                directory.mkdir(exist_ok=True)
                cmd=[sys.executable,"-B",str(Path(__file__).with_name("small_v2.py").resolve()),
                     "--output",str(directory),"--seconds",str(min(args.episode,stop_at-time.monotonic())),
                     "--seed",str(state["seed"]+104729*number),"--mode","exchange","--joint"]
                libraries=list(state["checkpoint_by_lane"].values())
                if from_checkpoint:
                    cmd+=["--resume",str(from_checkpoint)]
                elif number%6<4 and libraries:
                    cmd+=["--library",*libraries]
                with (directory/"worker.log").open("a") as stream:
                    stream.write(json.dumps(cmd)+"\n")
                    stream.flush()
                    worker=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,
                      env={**os.environ,"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1","MPLCONFIGDIR":"/tmp"},
                      start_new_session=True)
                row={"lane":lane,"number":number,"directory":str(directory),"pid":worker.pid}
                jobs[lane]=(row,worker)
                print("launched",worker.pid,directory,flush=True)
            save("running")
            time.sleep(4)
    finally:
        if stopped:
            status="interrupted"
        if state["best"]<=101.:
            status="target met"
        for lane,(row,p) in list(jobs.items()):
            if p.poll() is None:
                p.terminate()
            try:
                code=p.wait(timeout=14)
            except subprocess.TimeoutExpired:
                p.kill()
                code=p.wait()
            harvest(row,code)
            del jobs[lane]
            if stopped:
                pending[lane]=row
        save(status)
        print(status,state["best"],len(orders),len(novel),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seconds",type=float,default=5400)
    p.add_argument("--episode",type=float,default=240)
    p.add_argument("--workers",type=int,choices=(1,2,3),default=2)
    p.add_argument("--seed",type=int,default=82401)
    args=p.parse_args()
    if os.nice(0)<12:
        os.nice(12-os.nice(0))
    run(args)

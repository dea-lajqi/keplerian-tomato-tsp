"""Run and checkpoint a bounded sequence of Medium route and timing searches."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medium_coupled.run import ROOT
from medium_overnight.validation import official_udp, verify
from medium_v4.driver import unpack
from medium_coupled.checkpoint import atomic_json


def run(args):
    folder=args.directory.resolve()
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/"lock").open("a") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Campaign is already running")
        campaign(folder,args)


def campaign(folder,args):
    udp=official_udp(ROOT)
    stop=time.monotonic()+args.seconds
    file=folder/"campaign.json"
    seed=args.seed_record.resolve()
    state=json.loads(file.read_text()) if file.exists() else {
        "best_record":str(seed),"best":verify(udp,unpack(json.loads(seed.read_text())))[0],
        "completed":[],"pending":None,"round":0,"status":"starting"}
    modes=(
        ("endgame","endgame.py",120,("--starts","1","--minimum-gain",".001","--trial-seconds","14")),
        ("reinsert_one","reinsert.py",105,("--tail-size","1","--trial-seconds","5")),
        ("reinsert_two","reinsert.py",100,("--tail-size","2","--trial-seconds","5")),
        ("crossing","crossing.py",160,("--windows","24","--span","3","--trial-seconds","9")),
    )
    state.update(pid=os.getpid(),status="running")
    atomic_json(file,state)
    print("RUNNING",state["pid"],state["best"],flush=True)

    def execute(name,script,budget,options):
        if time.monotonic()>=stop:
            return False
        key=f'{state["round"]:02d}_{name}'
        if any(item["key"]==key for item in state["completed"]):
            return True
        pending=state["pending"]
        if pending is None:
            pending={"key":key,"source":state["best_record"]}
            state["pending"]=pending
            atomic_json(file,state)
        if pending["key"]!=key:
            raise RuntimeError("Checkpoint does not match next campaign stage")
        out=folder/key
        out.mkdir(exist_ok=True)
        duration=min(budget,max(1.,stop-time.monotonic()))
        command=[sys.executable,"-B",str(ROOT/"medium_coupled"/script),
                 "--seconds",str(duration),"--directory",str(out),
                 "--seed-record",pending["source"],*options]
        with (out/"worker.log").open("a") as log:
            log.write(f"\nStarting {key}, budget {duration:.1f}s\n")
            log.flush()
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        checkpoint=out/"checkpoint.json"
        if result.returncode or not checkpoint.exists():
            state["status"]="worker_error"
            atomic_json(file,state)
            raise RuntimeError(f"Worker failed, see {out/'worker.log'}")
        data=json.loads(checkpoint.read_text())
        record=data.get("record") or data.get("best_record")
        previous=state["best"]
        if record:
            record=Path(record)
            score=verify(udp,unpack(json.loads(record.read_text())))[0]
            if score<state["best"]:
                state.update(best=score,best_record=str(record.resolve()))
                print("RECORD",previous,score,record,flush=True)
        state["completed"].append({"key":key,"source":pending["source"],
                                   "best":state["best"],"directory":str(out)})
        state["pending"]=None
        atomic_json(file,state)
        print("COMPLETED",key,state["best"],flush=True)
        return True

    for _ in range(state["round"],args.rounds):
        initial=state.setdefault("round_start",state["best"])
        for mode in modes:
            if not execute(*mode):
                state["status"]="budget_finished"
                atomic_json(file,state)
                return
        if state["best"]>=initial:
            state["status"]="no_gain_across_round"
            atomic_json(file,state)
            print("FINISHED",state["status"],state["best"],flush=True)
            return
        if not execute("coupled","run.py",85,()):
            break
        state["round"]+=1
        del state["round_start"]
        atomic_json(file,state)
    state["status"]="finished"
    atomic_json(file,state)
    print("FINISHED",state["best"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--seed-record",type=Path,required=True)
    p.add_argument("--seconds",type=float,default=1080)
    p.add_argument("--rounds",type=int,default=3)
    run(p.parse_args())

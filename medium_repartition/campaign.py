"""Checkpointed Medium-only portfolio for partitioning and strict near-tour repair."""
import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import random
import subprocess
import sys
import time

from medium_coupled.checkpoint import atomic_json
from medium_overnight.validation import official_udp, to_solution, verify

RESULTS=Path("results/esa")
PREVIOUS=RESULTS/"medium_deadline180"
OLD=RESULTS/"medium_frontier"
BASE=OLD/"record_enriched/records/182.254267328103_f5531dbe.json"
ORIGINAL=PREVIOUS/"bridge_retiming/near/182.15_131_573fa0e5f2ad3cab.json"
ATLAS=PREVIOUS/"archive_phase/combined_events.json"


def run(args):
    folder=args.directory
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/"lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        portfolio(args,folder)


def portfolio(args,folder):
    base=RESULTS/"medium_repartition"
    data=folder/"campaign.json"
    if data.exists():
        state=json.loads(data.read_text())
    else:
        u=official_udp(Path.cwd())
        score=verify(u,to_solution(json.loads(BASE.read_text())))[0]
        state={"best":score,"best_path":str(BASE),"elapsed":0.,"jobs":[],"round":0}
    start=time.monotonic()
    prior=state["elapsed"]
    stop=start+max(0,args.seconds-prior)
    active={}
    verified_files=set(state.get("verified_records",[]))
    rng=random.Random(args.seed)
    stages=(
        (("move","balance","cluster"),False,.047,(182.2542,183.1,180.,181.25)),
        (("move","balance","cluster"),True,.043,(182.2542,181.25,180.,183.1)),
        (("cluster","balance"),True,.053,(184.5,183.5,185.2,182.2542)),
        (("move","seam"),True,.032,(182.2542,181.25,180.,184.5)),
        (("balance","move","cluster"),True,.037,(180.,181.25,182.2542,185.2)),
    )

    def save():
        state["elapsed"]=prior+time.monotonic()-start
        state["verified_records"]=sorted(verified_files)
        state["active"]={k:{"pid":p.pid,"directory":str(d)} for k,(p,_,d,_) in active.items()}
        atomic_json(data,state)

    def launch(key, module, directory, duration, options):
        seconds=min(duration,stop-time.monotonic())
        if seconds<15:
            return
        directory.mkdir(parents=True,exist_ok=True)
        logfile=(directory/"worker.log").open("a")
        command=[sys.executable,"-B","-m",module,"--directory",str(directory),
                 "--seed-record",state["best_path"],"--seconds",str(seconds),*map(str,options)]
        process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=logfile,stderr=subprocess.STDOUT)
        active[key]=(process,logfile,directory,time.monotonic())
        state["jobs"].append({"key":key,"directory":str(directory),"command":command,"pid":process.pid})
        print("START",key,process.pid,directory,flush=True)
        save()

    def records(directory):
        for f in sorted(directory.glob("records/*.json")):
            if str(f) in verified_files:
                continue
            raw=json.loads(f.read_text())
            try:
                udp=official_udp(Path.cwd())
                sol=to_solution(raw)
                fitness=verify(udp,sol)
                if any(float(udp.compute_transfer(*row[:4]))>udp.dv_threshold_exception
                       for row in raw["transfers"]):
                    continue
            except ValueError:
                continue
            verified_files.add(str(f))
            if fitness[0]<state["best"]-.0001:
                state.update(best=fitness[0],best_path=str(f))
                print("INDEPENDENT RECORD",fitness,f,flush=True)

    def work_partition():
        index=state["round"]
        operators,exact,step,days=stages[index%len(stages)]
        archives=[*sorted((PREVIOUS/"bidirectional_dense/feasible").glob("*.json"))]
        for d in sorted(folder.glob("partition_*"))[-5:]:
            choices=list(d.glob("feasible/*.json"))
            archives.extend(sorted(choices)[:10])
        banks=[PREVIOUS/name/"learned_events.json"
               for name in ("bidirectional_dense","adaptive","bridge_retiming")]
        banks += [p for p in (base/"pilot/events.json",base/"pilot_exact/events.json") if p.exists()]
        banks += [p for d in sorted(folder.glob("partition_*"))[-4:] if (p:=d/"events.json").exists()]
        options=["--trial-seconds",12 if exact else 9,"--random-seed",args.seed+index*16411,
                 "--operators",*operators,"--days",*days,
                 "--steps",step,step+.0083,step+.0147,"--horizon",185.5,
                 "--archive-seeds",*archives,"--events",*banks,
                 "--terminal-extra",folder/"future"]
        if exact:
            options.append("--exact-windows")
        state["round"]=index+1
        launch("partition","medium_repartition.search",folder/f"partition_{index:03d}",
               args.epoch,options)

    def work_auxiliary():
        tried={j["key"] for j in state["jobs"]}
        if "future" not in tried:
            launch("future","medium_repartition.future",folder/"future",340,
                   ["--atlas-seconds",260,"--shared",ATLAS])
            return
        ready=base/"prefix/gate_pool.json"
        if "phase_terminal" not in tried and ready.is_file():
            launch("phase_terminal","medium_repartition.terminal",folder/"phase_terminal",260,
                   ["--profiles",ready,"--atlas",ATLAS,"--enrich-seconds",155,
                    "--epoch-bin",.11,"--goal",182.25416])
            return
        if "near_branches" not in tried and (base/"near/replayed_near.json").is_file():
            launch("near_branches","medium_repartition.prefix",folder/"near_branches",310,
                   ["--near",base/"near/replayed_near.json","--original",ORIGINAL,"--trial-seconds",10])
            return
        gates=folder/"near_branches/gate_pool.json"
        if "coupled_terminal" not in tried and gates.is_file():
            launch("coupled_terminal","medium_repartition.terminal",folder/"coupled_terminal",250,
                   ["--profiles",gates,ready,"--atlas",ATLAS,"--enrich-seconds",160,
                    "--epoch-bin",.09,"--goal",state["best"]-.00011])
            return
        # Switch to a different structural model after the normalized prefix/terminal
        # stages. Give this worker different proposal operators and deadlines.
        index=state.get("alternate",0)
        key=f"alternate_{index:03d}"
        state["alternate"]=index+1
        if index%3==0:
            launch(key,"medium_frontier.topology",folder/key,170,
                   ["--normalize-passage","--wide-retime","--layouts",10+index,
                    "--goal",225+index,"--trial-seconds",16])
        else:
            banks=[p for d in sorted(folder.glob("partition_*"))[-5:] if (p:=d/"events.json").exists()]
            options=["--exact-windows","--operators","cluster","balance","--random-seed",
                     rng.randrange(100_000_000),"--steps",.036+.001*(index%7),.052,.041,
                     "--days",185.2,182.2542,184.5,180.,"--horizon",185.5,
                     "--terminal-extra",folder/"future","--events",*banks,
                     "--archive-seeds",*sorted((PREVIOUS/"bidirectional_dense/feasible").glob("*.json"))]
            launch(key,"medium_repartition.search",folder/key,args.epoch,options)

    try:
        print("PORTFOLIO",args.seconds,state["best"],flush=True)
        while time.monotonic()<stop or active:
            for key,(process,log,directory,began) in list(active.items()):
                records(directory)
                code=process.poll()
                if code is None and time.monotonic()<stop+25:
                    continue
                if code is None:
                    process.terminate()
                    code=process.wait(timeout=15)
                active.pop(key)
                log.close()
                cp=directory/"checkpoint.json"
                summary=json.loads(cp.read_text()) if cp.exists() else {}
                state["jobs"][-1].setdefault("observed",True)
                for item in state["jobs"]:
                    if item["pid"]==process.pid:
                        item.update(code=code,wall_seconds=time.monotonic()-began,
                                    counts=summary.get("counts",{}),
                                    statuses=dict(Counter(r["status"] for r in summary.get("proposals",[]))))
                        break
                print("FINISH",key,code,state["best"],flush=True)
                save()
            if time.monotonic()<stop-15:
                if "partition" not in active:
                    work_partition()
                if not any(k!="partition" for k in active):
                    work_auxiliary()
            if not active and time.monotonic()>=stop-15:
                break
            time.sleep(4)
    finally:
        for process,log,directory,_ in active.values():
            if process.poll() is None:
                process.terminate()
            log.close()
            records(directory)
        state["finished"]=time.monotonic()>=stop-15
        save()
        print("BEST",state["best"],state["best_path"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--seconds",type=float,default=3900.)
    p.add_argument("--epoch",type=float,default=355.)
    p.add_argument("--seed",type=int,default=37173518)
    run(p.parse_args())

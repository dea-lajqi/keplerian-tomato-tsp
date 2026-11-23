"""Run a deadline descent portfolio from the best strictly validated new itinerary."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from medium_coupled.checkpoint import atomic_json
from medium_repartition.sync import rank, share

BASE=Path("results/esa/medium_repartition")
PREVIOUS=Path("results/esa/medium_deadline180")


def portfolio(args):
    directory=args.directory
    directory.mkdir(parents=True,exist_ok=True)
    statefile=directory/"campaign.json"
    state=json.loads(statefile.read_text()) if statefile.exists() else {"jobs":[],"elapsed":0.}
    start=time.monotonic()
    elapsed=state["elapsed"]
    stop=start+max(0,args.seconds-elapsed)
    updated=set()
    while time.monotonic()<stop-20.:
        scores=rank(BASE,183.)
        if not scores:
            time.sleep(5)
            continue
        best,seed,_=scores[0]
        index=len(state["jobs"])
        phase=index%4
        files=[*sorted((BASE/"early_179").glob("terminal_*.json"))]
        parsed={float(p.stem[9:]) for p in files}
        proposed=[x for x in (179.25,179.65,178.6,177.5,179.9883,180.) if x in parsed]
        if len(proposed)<2:
            time.sleep(5)
            continue
        entry=directory/f"round_{index:03d}"
        bank=[*sorted(PREVIOUS.glob("*/learned_events.json")),BASE/"endpoint_181/events.json",
              BASE/"seam_dp/events.json",BASE/"normal_core_dp/events.json"]
        bank += [p for d in sorted(directory.glob("round_*"))[-3:] if (p:=d/"events.json").is_file()]
        archive=[]
        for name in ("seam_dp","endpoint_181","nov23_campaign/partition_000",
                     "nov23_campaign/partition_001","nov23_campaign/partition_002"):
            d=BASE/name
            cp=d/"checkpoint.json"
            if cp.is_file():
                options=list(json.loads(cp.read_text()).get("complete",{}).values())
                finalists=sorted(options,key=lambda x:(x["finish"]-.38*x["changes"],x["flight"]))[:5]
                finalists += sorted(options,key=lambda x:x["flight"])[:4]
                archive.extend(row["file"] for row in finalists)
        archive=list(dict.fromkeys(archive))
        common=["--events",*[p for p in bank if p.is_file()],"--archive-seeds",*archive,
                "--terminal-extra",BASE/"early_179",BASE/"early_181",BASE/"nov23_campaign/future"]
        if phase in (0,2):
            module="medium_repartition.search"
            operators=["endpoint","move","balance","cluster"] if phase==0 else [
                "move","cluster","endpoint","seam"]
            days=sorted(proposed,key=lambda x:abs(x-best+.18))
            extra=[*common,"--exact-windows","--operators",*operators,"--days",*days,
                   "--steps",.031+index*.0009,.041,.052,"--horizon",185.4,
                   "--trial-seconds",16,"--random-seed",551843+11519*index]
        elif phase==1:
            module="medium_repartition.seam_dp"
            extra=[*common,"--windows",3,4,"--windows",4,5,"--windows",5,5,
                   "--days",*proposed,"--trial-seconds",32,"--middle-limit",27,
                   "--steps",.043,.036,.052]
        else:
            module="medium_repartition.propagate"
            extra=[*common,"--limit",18,"--trial-seconds",9]
        if phase==3:
            duration=155.
        elif phase==1:
            duration=420.
        else:
            duration=470.
        baseline=best
        command=[sys.executable,"-B","-m",module,"--seed-record",str(seed),"--directory",str(entry),
                 "--seconds",str(min(duration,stop-time.monotonic())),*map(str,extra)]
        entry.mkdir(parents=True,exist_ok=True)
        item={"command":command,"directory":str(entry),"baseline":baseline}
        state.update(best=best,best_path=str(seed))
        state["jobs"].append(item)
        with (entry/"worker.log").open("a") as logfile:
            child=subprocess.Popen(command,stdout=logfile,stderr=subprocess.STDOUT)
            item["pid"]=child.pid
            print("START",index,module,best,child.pid,flush=True)
            while child.poll() is None and time.monotonic()<stop+20:
                if any(p not in updated for p in entry.glob("records/*.json")):
                    updated.update(entry.glob("records/*.json"))
                    share(BASE)
                state["elapsed"]=elapsed+time.monotonic()-start
                atomic_json(statefile,state)
                time.sleep(12)
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=10)
            item["exit"]=child.returncode
        share(BASE)
        scores=rank(BASE,183.)
        state["best"],path,_=scores[0]
        state["best_path"]=str(path)
        state["elapsed"]=elapsed+time.monotonic()-start
        atomic_json(statefile,state)
        print("FINISH",index,item["exit"],state["best"],flush=True)
    state["finished"]=True
    atomic_json(statefile,state)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--seconds",type=float,default=2800.)
    args=p.parse_args()
    args.directory.mkdir(parents=True,exist_ok=True)
    with (args.directory/"lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        portfolio(args)

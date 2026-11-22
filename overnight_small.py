"""Supervise a bounded, independently ESA-validated Small optimization campaign."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from uuid import uuid4

from continuous_windows_easy import FILES, SEED
from global_dp_easy import evaluator
from kttsp_esa_solver import chromosome, validate
from pareto_easy import write_json


ROOT = Path(__file__).resolve().parent
ARMS = {
    "boundary": ["--lookahead", "--long-stems", "--retime-suffix"],
    "fresh-suffix": ["--lookahead", "--free-suffix", "--long-stems", "--continuous-gap"],
    "elastic": ["--lookahead", "--relax", "6", "--retime-suffix"],
    "wide-closure": ["--long-stems", "--continuous-gap", "--retime-suffix"],
}
GOALS = [
    [112, 110, 108, 105, 103, 101, 114, 111.74],
    [111.74, 112, 111.5, 110, 108, 105, 103, 101],
    [101, 103, 105, 108, 110, 111.74, 112, 115],
]


class Campaign:
    def __init__(self, args):
        self.args = args
        self.folder = args.directory.resolve()
        self.folder.mkdir(parents=True, exist_ok=True)
        self.lock = (self.folder/"campaign.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.udp = evaluator().udp
        self.state_file = self.folder/"campaign.json"
        self.stop_requested = False
        self.jobs = {}
        self.started = time.monotonic()

        if self.state_file.exists():
            self.state = json.loads(self.state_file.read_text())
            incumbent = Path(self.state["incumbent"])
            self.done_seconds = self.state["elapsed_seconds"]
        else:
            incumbent = args.incumbent.resolve()
            self.done_seconds = 0.
            self.state = {"incumbent":str(incumbent),"elapsed_seconds":0.,"next_batch":0,
                          "arms":{},"history":[],"order_keys":[],"novel_order_keys":[],
                          "active":[],"records":[],"processed":[]}
        self.state.setdefault("campaign_seed",args.seed)
        self.incumbent = incumbent
        self.best = math.inf
        self.verify_file(incumbent,False)
        self.status = "running"
        self.seen_orders = {tuple(row) for row in self.state["order_keys"]}
        self.novel_orders = {tuple(row) for row in self.state["novel_order_keys"]}
        self.processed = set(self.state["processed"])
        self.stop_at = self.started + max(0., args.hours*3600-self.done_seconds)
        for mode in ARMS:
            self.state["arms"].setdefault(mode,{"count":0,"reward":0.,"failures":0})
        self.pending = list(self.state["active"])
        self.state["active"] = []
        self.env = {**os.environ,"MPLCONFIGDIR":"/tmp","OPENBLAS_NUM_THREADS":"1",
                    "OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","PYTHONUNBUFFERED":"1"}

    def verify(self, candidate, save, source):
        path, legs = candidate["path"],candidate["transfers"]
        result = validate(self.udp,path,legs)
        if not result or not result["valid"]:
            raise ValueError(f"Original ESA evaluator rejected {source}: {result}")
        value = result["fitness"][0]
        if value>=self.best:
            return False
        self.best = value
        self.best_entry = {**candidate,"validation":result}
        if save:
            name = f"record_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}.json"
            record = self.folder/name
            write_json(record,{**self.best_entry,"chromosome":chromosome(path,legs),
                               "source":str(source),"campaign_seed":self.args.seed})
            self.incumbent = record
            self.state["records"].append(str(record))
            print(f"INDEPENDENT ESA RECORD {value:.14f} {record}",flush=True)
        return True

    def verify_file(self, file, save):
        return self.verify(json.loads(file.read_text()),save,file)

    def save(self):
        self.state.update(incumbent=str(self.incumbent),best_days=self.best,
                          target_gap_days=self.best-101.,pid=os.getpid(),status=self.status,
                          elapsed_seconds=self.done_seconds+time.monotonic()-self.started,
                          budget_seconds=self.args.hours*3600,worker_limit=self.args.workers,
                          last_update_utc=datetime.now(timezone.utc).isoformat(),
                          order_keys=sorted(self.seen_orders),novel_order_keys=sorted(self.novel_orders),
                          active=[job[0] for job in self.jobs.values()]+self.pending,
                          processed=sorted(self.processed))
        temp = self.state_file.with_name(f".campaign-{uuid4().hex}.tmp")
        with temp.open("x",encoding="utf-8") as handle:
            json.dump(self.state,handle,indent=2,allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(self.state_file)

    def choose(self):
        activity = Counter(job[0]["mode"] for job in self.jobs.values())
        arms = self.state["arms"]
        total = sum(s["count"] for s in arms.values())+2
        def priority(mode):
            row = arms[mode]
            count = row["count"]
            return (row["reward"]/max(1,count)+1.25*math.sqrt(math.log(total)/(1+count))
                    -.8*activity[mode]-.3*min(row["failures"],3))
        return max(arms,key=priority)

    def launch(self):
        remaining = self.stop_at-time.monotonic()
        if self.pending:
            batch = self.pending.pop(0)
            folder = Path(batch["folder"])
            checkpoints = sorted(folder.glob("*checkpoint.json"),key=lambda p:p.stat().st_mtime)
            resume = checkpoints[-1:]
            old_pid = batch.get("pid")
            if old_pid:
                try:
                    command = Path(f"/proc/{old_pid}/cmdline").read_bytes().split(b"\0")
                    if os.fsencode(folder) in command and any(s.endswith(b"/bidirectional_small.py") for s in command):
                        os.kill(old_pid,signal.SIGINT)
                except (OSError,ProcessLookupError):
                    pass
        else:
            number = self.state["next_batch"]
            self.state["next_batch"] += 1
            mode = self.choose()
            folder = self.folder/f"batch_{number:04d}_{mode}"
            folder.mkdir(exist_ok=True)
            batch = {"number":number,"mode":mode,"folder":str(folder),
                     "seed":self.state["campaign_seed"]+104729*number,"incumbent":str(self.incumbent)}
            resume = []
        number,mode = batch["number"],batch["mode"]
        goals = GOALS[number%len(GOALS)]
        command = [sys.executable,"-B",str(ROOT/"bidirectional_small.py"),
                   "--incumbent",batch["incumbent"],"--output",str(folder),
                   "--seconds",str(max(.1,min(remaining,self.args.batch_seconds))),
                   "--seed",str(batch["seed"]),"--beam",str((28,42,58,35)[number%4]),
                   "--fanout",str((5,7,8)[number%3]),"--iterations","1000",
                   "--hybrid","--close","--diverse",*ARMS[mode],
                   "--goals",*(str(x) for x in goals)]
        if resume:
            command.extend(("--resume",str(resume[0])))
        with (folder/"worker.log").open("a",buffering=1) as log:
            log.write(json.dumps(command)+"\n")
            process = subprocess.Popen(command,cwd=ROOT,env=self.env,stdout=log,
                                       stderr=subprocess.STDOUT,start_new_session=True)
        batch["pid"] = process.pid
        self.jobs[process.pid] = (batch,process)
        self.save()
        print(f"batch {number} {mode} pid {process.pid}",flush=True)

    def harvest(self, batch, status, final):
        folder = Path(batch["folder"])
        gain = batch.get("gain",False)
        for path in sorted(folder.glob("easy_bidir_*.json"),key=lambda p:p.stat().st_mtime_ns):
            if path.name.endswith("_checkpoint.json") or str(path) in self.processed:
                continue
            try:
                gain |= self.verify_file(path,True)
            except (ValueError,KeyError) as error:
                print(str(error),flush=True)
            self.processed.add(str(path))
        batch["gain"] = gain
        if not final:
            pid = batch["pid"]
            try:
                for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                    if line.startswith("VmHWM:"):
                        batch["peak_rss_kib"] = max(batch.get("peak_rss_kib",0),int(line.split()[1]))
                        break
            except OSError:
                pass
            return
        checkpoints = sorted(folder.glob("*checkpoint.json"),key=lambda p:p.stat().st_mtime)
        summary = json.loads(checkpoints[-1].read_text()) if checkpoints else {}
        orders = {tuple(row) for row in summary.get("orders",[])}
        novel = {tuple(row) for row in summary.get("novel_orders",[])}
        new = len(novel-self.novel_orders)
        self.seen_orders.update(orders)
        self.novel_orders.update(novel)
        elite_folder = self.folder/"elites"
        elite_folder.mkdir(exist_ok=True)
        for candidate in summary.get("elite_candidates",[]) if len(list(elite_folder.glob("easy*.json")))<32 else ():
            path = candidate.get("path",[])
            token = hashlib.sha256(bytes(path)).hexdigest()[:16]
            file = elite_folder/f"easy_elite_{token}.json"
            if file.exists():
                continue
            try:
                self.verify(candidate,True,checkpoints[-1])
                write_json(file,candidate)
            except (ValueError,KeyError) as error:
                print(str(error),flush=True)
        counts = summary.get("counts",{})
        attempts = summary.get("attempts",[])
        tight = max((row["max_visits"] for row in attempts if row["goal"]<=110),default=0)
        score = 4.*int(gain)+1.4*math.log1p(new)+.45*math.log1p(counts.get("timed_joins",0))+tight/49
        row = self.state["arms"][batch["mode"]]
        row["count"] += 1
        row["reward"] += score
        row["failures"] += int(status!=0)
        self.state["history"].append({**batch,"exit_code":status,"counts":counts,
                                      "new_orders":new,"best_tight_depth":tight,
                                      "checkpoint":str(checkpoints[-1]) if checkpoints else None,
                                      "worker_seconds":summary.get("elapsed_seconds",0.)})
        print(f"batch {batch['number']} exit {status}; new orders {new}; tight depth {tight}; best {self.best:.14f}",flush=True)

    def signal(self, *_):
        self.stop_requested = True

    def run(self):
        signal.signal(signal.SIGINT,self.signal)
        signal.signal(signal.SIGTERM,self.signal)
        self.save()
        print(f"campaign pid {os.getpid()} | ESA baseline {self.best:.14f} | {self.folder}",flush=True)
        try:
            while not self.stop_requested and self.best>101. and time.monotonic()<self.stop_at:
                for pid,(batch,process) in list(self.jobs.items()):
                    code = process.poll()
                    self.harvest(batch,code,code is not None)
                    if code is not None:
                        del self.jobs[pid]
                while len(self.jobs)<self.args.workers and self.stop_at-time.monotonic()>5. and self.best>101.:
                    self.launch()
                self.save()
                time.sleep(min(5.,max(0.,self.stop_at-time.monotonic())))
        finally:
            self.status = "target met" if self.best<=101. else "interrupted" if self.stop_requested else "budget finished"
            for batch,process in self.jobs.values():
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
            for pid,(batch,process) in list(self.jobs.items()):
                try:
                    code = process.wait(timeout=12.)
                except subprocess.TimeoutExpired:
                    process.kill()
                    code = process.wait()
                self.harvest(batch,code,True)
                del self.jobs[pid]
                if self.stop_requested:
                    self.pending.append(batch)
            self.save()
            print(f"{self.status}: {self.best:.14f} days; "
                  f"{len(self.seen_orders)} distinct feasible orders ({len(self.novel_orders)} new)",flush=True)


def report(directory):
    state = json.loads((directory/"campaign.json").read_text())
    seed = json.loads(SEED.read_text())
    udp = evaluator().udp
    baseline = validate(udp,seed["path"],seed["transfers"])
    record = json.loads(Path(state["incumbent"]).read_text())
    valid = validate(udp,record["path"],record["transfers"])
    if not baseline or not baseline["valid"] or not valid or not valid["valid"]:
        raise ValueError("ESA rejected a reported schedule")
    days = valid["fitness"][0]
    totals = {}
    for batch in state["history"]:
        totals.setdefault(batch["mode"],Counter()).update(batch["counts"])
    output = {"status":state["status"],"pid":state["pid"],"utc":state["last_update_utc"],
              "best_esa_days":days,"improvement_days":baseline["fitness"][0]-days,
              "gap_to_101_days":days-101.,"elapsed_seconds":state["elapsed_seconds"],
              "distinct_complete_orders":len(state["order_keys"]),
              "unarchived_complete_orders":len(state["novel_order_keys"]),
              "completed_batch_counts_by_mode":totals,
              "completed_batch_worker_seconds":sum(b["worker_seconds"] for b in state["history"]),
              "active_workers":state["active"],"best_file":state["incumbent"]}
    print(json.dumps(output,indent=2))


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--status",action="store_true",help="Revalidate and report the current campaign state")
    p.add_argument("--incumbent",type=Path,default=SEED)
    p.add_argument("--hours",type=float,default=8.)
    p.add_argument("--batch-seconds",type=float,default=390.)
    p.add_argument("--workers",type=int,default=3)
    p.add_argument("--seed",type=int,default=62201)
    args = p.parse_args()
    if args.status:
        report(args.directory)
        sys.exit(0)
    if not 1<=args.workers<=min(8,len(os.sched_getaffinity(0))) or not 0.<args.hours<=8. or args.batch_seconds<5.:
        p.error("Use 1–8 available workers, a positive budget of at most 8 hours and batches of at least 5 seconds.")
    Campaign(args).run()

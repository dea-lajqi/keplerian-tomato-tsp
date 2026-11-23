"""Adaptive, resumable overnight optimization, exclusively for ESA Medium."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medium_coupled.checkpoint import atomic_json
from global_event_easy import geometry
from medium_overnight.validation import (EVALUATOR_REL, as_chromosome, official_udp,
                                          solution_payload, to_solution, verify)

RESULTS = ROOT / "results/esa"
OLD = RESULTS / "medium_coupled"
HISTORICAL_ATLAS = OLD / "day192_merged_events/events.json"
HISTORICAL_GATE = OLD / "day192_high_merge/gate_pool.json"
PROTECTED = (EVALUATOR_REL, "data/spoc4/kttsp/problems/medium.kttsp", "data/medium.kttsp")
MILESTONES = (188,185,180,175,170,165,160,155,150,146)
FAMILIES = ("exact_blocks","normal_phase","structural","terminal","seam","full_topology","continuous")
STOP = False


def stop_signal(*_):
    global STOP
    STOP = True


def read(path, fallback=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return fallback


def stamp(t=None):
    return datetime.fromtimestamp(time.time() if t is None else t, timezone.utc).isoformat()


def extract(d):
    sol = d.get("solution", d)
    if "dep" not in sol:
        sol = to_solution(sol)
    return {k: sol[k] for k in ("path","dep","tof")}


def route_key(sol):
    return hashlib.sha256(bytes(sol["path"])).hexdigest()


def checked(udp,path):
    sol=extract(json.loads(Path(path).read_text()))
    return sol,verify(udp,sol)


def protected_hashes():
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in PROTECTED}


def text_file(path, content):
    descriptor,name=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
    temp=Path(name)
    try:
        with os.fdopen(descriptor,"w") as out:
            out.write(content)
            out.flush()
            os.fsync(out.fileno())
        for retry in range(8):
            try:
                os.replace(temp,path)
                break
            except PermissionError:
                if retry==7:
                    raise
                time.sleep(.04*(retry+1))
    finally:
        temp.unlink(missing_ok=True)


class Worker:
    def __init__(self,args):
        self.args=args
        self.folder=args.directory
        self.stop=time.monotonic()+args.seconds
        self.source=args.seed_record
        self.sol=extract(json.loads(self.source.read_text()))
        self.finish=self.sol["dep"][-1]+self.sol["tof"][-1]
        self.rng=random.Random(args.seed)
        self.plan=read(self.folder/"stages.json",{})
        self.udp=official_udp(ROOT)
        verify(self.udp,self.sol)
        sizes=self.udp.tomato_orbital_parameters[:,0]
        self.crossings=[i for i,(a,b) in enumerate(zip(self.sol["path"],self.sol["path"][1:]))
                        if max(sizes[a],sizes[b])>2*min(sizes[a],sizes[b])]
        self.gate=self.crossings[-1]
        self.epoch=lambda i:self.sol["dep"][i-1]+self.sol["tof"][i-1]

    def stopping(self):
        return STOP or (self.args.parent is not None and os.getppid()!=self.args.parent)

    def stage(self, name, script, seconds, *options):
        out=self.folder/name
        if self.plan.get(name,{}).get("status")=="done":
            return out
        remaining=min(seconds,self.stop-time.monotonic()-2)
        if self.stopping() or remaining<4:
            return out
        out.mkdir(exist_ok=True)
        cmd=[sys.executable,"-B",str(ROOT/"medium_coupled"/script),
             "--seed-record",str(self.source),"--directory",str(out),
             "--seconds",str(remaining),*(str(a) for a in options)]
        self.plan[name]={"status":"running","command":cmd,"started":stamp()}
        atomic_json(self.folder/"stages.json",self.plan)
        start=time.monotonic()
        with (out/"worker.log").open("a") as log:
            log.write(f"\n{stamp()} {name} budget {remaining:.1f}s\n")
            log.flush()
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            try:
                while child.poll() is None and not self.stopping() and time.monotonic()<start+remaining+2:
                    time.sleep(.25)
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait()
        self.plan[name].update(status="done" if child.returncode==0 else "failed",
                                returncode=child.returncode,seconds=time.monotonic()-start)
        atomic_json(self.folder/"stages.json",self.plan)
        return out

    def profile_floor(self,profiles):
        return min((row["arrival"] for p in profiles for row in read(p,{}).values()
                    if len(row.get("legs",()))==160 and row.get("spend",9)<=4),
                   default=self.epoch(self.gate))

    def low(self,profiles,tag="low"):
        profiles=[p for p in profiles if p.exists()]
        if not profiles:
            profiles=[HISTORICAL_GATE]
        best=self.args.baseline
        floor=max(0.,min(self.profile_floor(profiles),self.epoch(self.gate))-.08)
        # Bound the sampled span around the earliest observed prefix, even
        # when it is much earlier than the current best schedule.
        goal=min(best+.38,floor+7.5,max(floor+3.3,self.finish+.20))
        if goal<=floor+.40:
            return
        atlas=self.stage(tag+"_events","terminal_atlas.py",190,"--build-only",
             "--goal",goal,"--floor-epoch",floor,"--atlas-seconds",153,
             "--profiles",*profiles,"--seed-atlas",HISTORICAL_ATLAS)
        path=atlas/"events.json"
        if not path.exists():
            return
        deadlines=sorted({round(goal,8),*(round(x,8) for x in
                        (best-.8,best-.35,best-.08,math.floor(best)) if floor+.4<x<=goal)})
        self.stage(tag+"_exact","exact_terminal.py",160,"--prefix-limit",8,
                   "--profiles",*profiles,"--atlas",path,"--forward",*deadlines,
                   "--backward",*deadlines[:2])

    def run(self):
        name=self.args.family
        i=self.args.seed
        g=self.gate
        goal=self.args.baseline+.25
        if name=="exact_blocks":
            cut=g-self.rng.choice((10,11,12,13,14,15))
            high_clock=self.epoch(g)
            late=self.stage("late_high","high_exact_atlas.py",205,
                   "--width",self.rng.choice((17,18,19,20)),
                   "--goal",high_clock+.6,"--gates",high_clock-1.2,high_clock-.45,high_clock+.5)
            early=self.stage("middle_high","high_exact_atlas.py",205,
                   "--width",min(20,cut-self.crossings[-2]-1),
                   "--right-index",cut,"--goal",self.epoch(cut)+.55,
                   "--gates",self.epoch(cut)-1.6,self.epoch(cut)-.5,self.epoch(cut)+.5,
                   "--mission-goal",goal)
            middle=early/"mid_pool.json"
            profiles=[late/"gate_pool.json",early/"gate_pool.json"]
            if middle.exists() and (late/"events.json").exists():
                merge=self.stage("merge","high_atlas_merge.py",110,"--middle",middle,
                     "--atlas",late/"events.json","--gates",high_clock-2.5,high_clock-1.2,high_clock+.35)
                profiles.insert(0,merge/"gate_pool.json")
            self.low(profiles)
        elif name=="normal_phase":
            longest=sorted(range(1,g),key=lambda j:-self.sol["tof"][j])[:24]
            others=[j for j in range(21,g) if j not in longest]
            positions=list(dict.fromkeys(longest[:5]+self.rng.sample(longest[5:],7)+
                                          self.rng.sample(others,min(12,len(others)))))
            out=self.stage("normal_probe","continuous_spine.py",250,"--probe-long",
                           "--positions",*positions,"--goal",goal,"--trial-seconds",15)
            if (out/"gate_pool.json").exists():
                self.low([out/"gate_pool.json"])
        elif name=="structural":
            mode=self.rng.choice(("windows","exchange","crossing","final_gate"))
            out=self.stage("route","nonlocal_routes.py",230,"--mode",mode,"--goal",goal,
                           "--phase",self.rng.uniform(.003,.065),"--beam",self.rng.choice((56,80,105)),
                           "--save-prefixes","--polish")
            if (out/"gate_pool.json").exists():
                repair=self.stage("exceptions","exception_repair.py",125,"--fixed",
                          "--profiles",out/"gate_pool.json","--goal",goal)
                self.low([repair/"gate_pool.json",out/"gate_pool.json"])
        elif name=="terminal":
            self.low([HISTORICAL_GATE])
            bank=self.folder/"low_events/events.json"
            if bank.exists():
                self.stage("off_grid","atlas_enrich.py",145,"--goal",goal,"--atlas",bank,
                           "--profiles",HISTORICAL_GATE,"--epoch-bin",self.rng.uniform(.15,.4))
            self.stage("end_relocation","reinsert.py",105,"--tail-size",1+i%3)
        elif name=="seam":
            d=self.args.baseline
            self.stage("coupled","coupled_seam.py",230,"--profiles",HISTORICAL_GATE,
                       "--beam",self.rng.choice((85,125,155)),"--deadlines",d-.16,d-.45,d+.08,
                       "--steps",self.rng.uniform(.023,.041))
            self.stage("meet","meet.py",150,"--repair-near","--goal",d-self.rng.uniform(.08,.68),
                       "--beam",self.rng.choice((310,450,600)))
        elif name=="full_topology":
            out=self.stage("whole_order","shell_topologies.py",230,"--mixed","--save-prefixes",
                     "--goal",min(278,self.finish+35+i%19),"--per-topology",3+i%7,
                     "--random-seed",i)
            if (out/"gate_pool.json").exists():
                self.low([out/"gate_pool.json"])
            # Also move endpoints in the middle low shell and reconstruct both crossings.
            out=self.stage("paired_crossings","multigate.py",180,"--mode","paired",
                           "--cases",40+i%31,"--goal",goal)
            if (out/"gate_pool.json").exists():
                self.low([out/"gate_pool.json"],tag="paired_low")
        else:
            self.stage("wide","clock_basin.py",230,"--trial-seconds",10)
            self.stage("polish","run.py",105,"--seed-only")
            self.stage("relocate","relocate.py",100)


class Supervisor:
    def __init__(self,args):
        self.a=args
        self.dir=args.directory.resolve()
        self.dir.mkdir(parents=True,exist_ok=True)
        self.file=self.dir/"campaign.json"
        self.udp=official_udp(ROOT)
        self.state=read(self.file)
        self.active={}
        self.seen=set()
        if self.state is None:
            self.initialize()
        self.deadline=self.state["deadline"]
        self.hashes=self.state["protected"]
        self.seen.update(self.state.get("seen_records",()))
        self.archive={a["order"]:a for a in self.state["archive"]}
        for task in self.state["tasks"]:
            if task["status"]=="running":
                task["status"]="pending"

    def initialize(self):
        archive={}
        count=0
        for path in RESULTS.glob("medium*/**/*.json"):
            if not ("records" in path.parts or "best" in path.name or "verified" in path.name):
                continue
            try:
                sol,fitness=checked(self.udp,path)
            except (OSError,ValueError,TypeError,KeyError,AttributeError):
                continue
            count+=1
            k=route_key(sol)
            if k not in archive or fitness[0]<archive[k]["score"]:
                archive[k]={"order":k,"path":str(path.resolve()),"score":fitness[0],
                            "exceptions":int(self.udp.n_dv_exception+fitness[4])}
        if not archive:
            raise RuntimeError("No independently verified Medium incumbent")
        pool=sorted(archive.values(),key=lambda a:a["score"])
        now=time.time()
        (normal,exceptional),_=geometry(self.udp,4.9048695e12)
        low={i for i,a in enumerate(self.udp.tomato_orbital_parameters[:,0]) if a<8e6}
        high=set(range(181))-low
        unseen=low.copy()
        components=[]
        while unseen:
            frontier=[unseen.pop()]
            count_nodes=0
            while frontier:
                a=frontier.pop()
                count_nodes+=1
                neighbors={b for b in unseen if exceptional[a,b] or exceptional[b,a]}
                frontier.extend(neighbors)
                unseen.difference_update(neighbors)
            components.append(count_nodes)
        shell={"low_components":sorted(components),
               "normal_low_high_edges":sum(int(normal[a,b]) for a in low for b in high)}
        self.state={"started":now,"deadline":now+self.a.seconds,"baseline":pool[0],
                    "best":pool[0],"archive":pool[:55],"records":[],"tasks":[],
                    "historical_checks_passed":count,"protected":protected_hashes(),
                    "geometry":shell,"seen_records":[],"status":"running"}
        self.log(f"Verified {count} archived schedules. Baseline {pool[0]['score']:.14f} days")

    def log(self,msg):
        line=f"{stamp()} {msg}"
        print(line,flush=True)
        with (self.dir/"progress.log").open("a") as f:
            f.write(line+"\n")

    def scan(self):
        candidates=[]
        for f in (self.dir/"runs").glob("**/records/*.json"):
            s=str(f)
            if s not in self.seen:
                candidates.append((f.stat().st_mtime_ns,f))
        for _,f in sorted(candidates):
            try:
                sol,fitness=checked(self.udp,f)
            except (OSError,ValueError,TypeError,KeyError) as exc:
                self.log(f"Rejected candidate {f.name}: {exc}")
                self.seen.add(str(f))
                continue
            self.seen.add(str(f))
            k=route_key(sol)
            score=fitness[0]
            item={"order":k,"path":str(f),"score":score,
                  "exceptions":int(self.udp.n_dv_exception+fitness[4])}
            if k not in self.archive or score<self.archive[k]["score"]:
                self.archive[k]=item
            if score<self.state["best"]["score"]:
                # Reopen the original evaluator for independent promotion.
                original=official_udp(ROOT)
                fitness=verify(original,sol)
                row=solution_payload(sol,fitness,udp=original,provenance={
                      "worker_record":str(f),"algorithm":f.relative_to(self.dir).parts[1:3],
                      "reported_details":read(f,{}).get("provenance",{})})
                now=time.time()
                waits=[sol["dep"][0]]+[b-a-d for a,b,d in
                      zip(sol["dep"],sol["dep"][1:],sol["tof"])]
                destination=self.dir/"records"/f"{len(self.state['records'])+1:05d}_{score:.14f}.json"
                atomic_json(destination,{"solution":sol,"decision_vector":as_chromosome(sol),
                    "waiting_days":waits,"exception_indices":[i for i,l in enumerate(row["transfers"])
                    if l[4]>original.dv_threshold],"discovered_utc":stamp(now),**row})
                prior=self.state["best"]["score"]
                item={"order":k,"path":str(destination),"score":score,"exceptions":row["exceptions"]}
                self.archive[k]=item
                self.state["best"]=item
                self.state["records"].append({**item,"time":stamp(now),"previous":prior,
                  "algorithm":"/".join(f.relative_to(self.dir).parts[1:3]),
                  "exceptions":row["exceptions"]})
                self.log(f"Record {prior:.14f} -> {score:.14f} days | {destination}")
                self.save()

    def choose(self):
        for task in self.state["tasks"]:
            if task["status"]=="pending":
                return task
        n=len(self.state["tasks"])
        stats=Counter(task["family"] for task in self.state["tasks"])
        gains=Counter()
        for rec in self.state["records"]:
            if rec["previous"]-rec["score"]>1e-5:
                gains[rec["algorithm"].split("/")[0].split("_",1)[1]]+=rec["previous"]-rec["score"]
        active={r["family"] for r,_,_ in self.active.values()}
        pool=[f for f in FAMILIES if f not in active] or list(FAMILIES)
        family=min(pool,key=lambda f:stats[f]/(1+min(3,gains[f])))
        best=self.state["best"]
        elite=sorted(self.archive.values(),key=lambda a:a["score"])
        recent=self.state["records"][-1]["time"] if self.state["records"] else stamp(self.state["started"])
        dry=time.time()-datetime.fromisoformat(recent).timestamp()
        if family=="full_topology" or n%5==4 or (dry>1800 and n%3==2):
            other=[x for x in elite if x["order"]!=best["order"] and x["score"]<best["score"]+12.]
            if other:
                best=random.Random(self.a.seed+n).choice(other[:16])
        task={"number":n,"family":family,"source":best["path"],"source_score":best["score"],
              "baseline":self.state["best"]["score"],"status":"pending","seed":self.a.seed+n*7919}
        self.state["tasks"].append(task)
        return task

    def launch(self,task):
        tag=f"{task['number']:04d}_{task['family']}"
        remaining=min(920,self.deadline-time.time()-15)
        folder=self.dir/"runs"/tag
        folder.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,"-B",str(Path(__file__).resolve()),"worker","--directory",str(folder),
             "--family",task["family"],"--seed-record",task["source"],"--baseline",str(task["baseline"]),
             "--seed",str(task["seed"]),"--seconds",str(remaining),"--parent",str(os.getpid())]
        with (folder/"worker.log").open("a") as log:
            child=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                       stderr=subprocess.STDOUT,start_new_session=True)
        task.update(status="running",pid=child.pid,stage=tag,started=stamp())
        self.active[child.pid]=(task,child,time.monotonic())
        self.log(f"Started {tag} (PID {child.pid}) from {task['source_score']:.9f}")

    def poll(self):
        for pid,(task,process,start) in list(self.active.items()):
            code=process.poll()
            if code is None:
                continue
            task.update(status="done" if code==0 else "failed",exit_code=code,
                        worker_seconds=task.get("worker_seconds",0.)+time.monotonic()-start)
            self.active.pop(pid)
            self.log(f"{task['stage']} exited with status {code}")

    def aggregate(self):
        totals=Counter()
        orders=set()
        for file in (self.dir/"runs").glob("**/checkpoint.json"):
            d=read(file,{})
            counts=d.get("counts")
            vals=[counts] if counts is not None else [x.get("stats",x.get("counts",{}))
                     for x in d.get("trials",d.get("tests",[]))]
            for c in vals:
                totals["official_checks"]+=c.get("official_checks",0)
                totals["valid_schedules"]+=c.get("valid",c.get("valid_schedules",0))
            orders.update(d.get("orders",d.get("valid_route_signatures",[])))
        orders.update(r["order"] for r in self.state["records"])
        totals["tracked_distinct_verified_orders"]=len(orders)
        totals["coordinator_record_replays"]=len(self.seen)
        totals["worker_seconds"]=sum(t.get("worker_seconds",0.) for t in self.state["tasks"])+sum(
                                time.monotonic()-start for _,_,start in self.active.values())
        return dict(totals)

    def save(self):
        self.state["updated"]=stamp()
        self.state["pid"]=os.getpid()
        self.state["archive"]=sorted(self.archive.values(),key=lambda a:a["score"])[:60]
        self.state["seen_records"]=sorted(self.seen)
        self.state["metrics"]=self.aggregate()
        self.state["wall_seconds"]=time.time()-self.state["started"]
        atomic_json(self.file,self.state)
        atomic_json(self.dir/"best.json",self.state["best"])
        start=self.state["baseline"]["score"]
        best=self.state["best"]["score"]
        gain=start-best
        milestones=[str(x) for x in MILESTONES if (best<=x if x==146 else best<x)]
        by_mode=Counter((t["family"],t["status"]) for t in self.state["tasks"])
        rows=["# ESA Medium overnight optimization","",
              f"Updated {self.state['updated']} UTC. Status **{self.state['status']}**.",
              f"Budget ends {stamp(self.deadline)} UTC.","",
              f"- Starting independently validated fitness **{start:.14f} days**.",
              f"- Current independently validated fitness **{best:.14f} days**.",
              f"- Gain **{gain:.14f} days**, {gain*24:.8f} hours, {gain*1440:.5f} minutes ({100*gain/start:.5f}%).",
              f"- 181 tomatoes and {self.state['best'].get('exceptions','see schedule')} exceptional maneuvers.",
              f"- Milestones reached {', '.join(milestones) or 'none yet'}.",
              f"- Best saved schedule `{self.state['best']['path']}`.",
              f"- Wall time {self.state['wall_seconds']/3600:.3f} hours.",
              "", "## Checkpointed metrics","",
              "Counts for beam and DP work include the complete schedules those workers retained and checked.",
              "Distinct orders are a lower bound where worker logs omit route signatures.",""]
        rows.extend(f"- {key.replace('_',' ')} **{value:,.2f}**."
                    for key,value in self.state["metrics"].items())
        rows.extend(["","## Validated records","","| UTC | Previous days | New days | Exceptions | Method | Saved schedule |",
                     "|---|---:|---:|---:|---|---|"])
        rows.extend(f"| {r['time']} | {r['previous']:.14f} | {r['score']:.14f} | {r['exceptions']} | {r['algorithm']} | `{r['path']}` |"
                    for r in self.state["records"])
        rows.extend(["","## Strategy outcomes",""])
        rows.extend(f"- {mode}, {status} **{n}**." for (mode,status),n in sorted(by_mode.items()))
        contributions=Counter()
        for record in self.state["records"]:
            mode=record["algorithm"].split("/")[0].split("_",1)[1]
            contributions[mode]+=record["previous"]-record["score"]
        rows.extend(f"- {mode} contributed **{gain:.10f} days** of record improvements."
                    for mode,gain in sorted(contributions.items()))
        rows.extend(["", "## Search limits and reproduction", "",
                     "Exact label-setting exhausts the sampled events for its tested node sets and starts. "
                     "Earlier unsampled orbital phases and other complete tour structures remain possible.",
                     "The four-crossing cases reserve one final exceptional maneuver. Other structural workers "
                     "test itinerary and exception alternatives separately.",
                     f"Necessary geometric screening {self.state.get('geometry', {})}. "
                     "With three disconnected low groups and no normal low/high edge, "
                     "at least four exceptional shell crossings are necessary in this graph.",
                     "To reproduce the best score, load the saved schedule's decision vector "
                     "or concatenate solution.dep, solution.tof, and solution.path in that order, "
                     "then call the original ESA Medium fitness function. Use PyKEP 2.6.x.",
                     "Worker replay commands are recorded in each run's stages.json.",
                     f"Final independent verification and tests {self.state.get('final_checks','pending')}."])
        text_file(self.dir/"MORNING_REPORT.md","\n".join(rows)+"\n")

    def finish(self):
        for pid in list(self.active):
            try:os.killpg(pid,signal.SIGTERM)
            except ProcessLookupError:pass
        until=time.monotonic()+5
        while self.active and time.monotonic()<until:
            self.poll()
            time.sleep(.15)
        for pid,(_,process,_) in list(self.active.items()):
            try:os.killpg(pid,signal.SIGKILL)
            except ProcessLookupError:pass
            process.wait()
        self.poll()
        self.scan()
        _,fit=checked(official_udp(ROOT),self.state["best"]["path"])
        original_ok=protected_hashes()==self.hashes
        test_env={**os.environ,"PYTHONPATH":str(ROOT/"medium_overnight")}
        with (self.dir/"tests.log").open("w") as log:
            proc=subprocess.run([sys.executable,"-B","-m","unittest","medium_overnight.test_offline",
                     "medium_v4.test_offline","medium_v4.test_integration_offline"],
                     cwd=ROOT,env=test_env,stdout=log,stderr=subprocess.STDOUT,timeout=12)
        self.state["final_checks"]={"fitness":fit,"protected_inputs_match":original_ok,
                                    "existing_test_exit_code":proc.returncode}
        self.state["status"]="stopped" if STOP else "budget_finished"
        self.save()
        self.log(f"Finished with independently verified fitness {fit}")

    def run(self):
        self.state["status"]="running"
        self.save()
        self.log("Supervisor active")
        try:
            while not STOP and time.time()<self.deadline-12:
                self.poll()
                self.scan()
                while len(self.active)<self.a.workers and time.time()<self.deadline-35:
                    self.launch(self.choose())
                self.save()
                time.sleep(4)
        finally:
            self.finish()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("supervisor","worker"))
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--seconds",type=float,default=28800)
    p.add_argument("--workers",type=int,choices=range(1,7),default=4)
    p.add_argument("--seed",type=int,default=146188)
    p.add_argument("--family",choices=FAMILIES)
    p.add_argument("--seed-record",type=Path)
    p.add_argument("--baseline",type=float)
    p.add_argument("--parent",type=int)
    args=p.parse_args()
    signal.signal(signal.SIGTERM,stop_signal)
    signal.signal(signal.SIGINT,stop_signal)
    if args.mode=="worker":
        with (args.directory/"worker.lock").open("a") as lock:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit("A worker already owns this task checkpoint")
            Worker(args).run()
        return
    args.directory.mkdir(parents=True,exist_ok=True)
    with (args.directory/"supervisor.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        Supervisor(args).run()


if __name__=="__main__":
    main()

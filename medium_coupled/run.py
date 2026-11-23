"""Medium-only complete-route timing and elite multistart benchmark."""
import argparse
from collections import Counter
import json
from itertools import pairwise
from pathlib import Path
import random
import signal
import sys
import time
from uuid import uuid4

import numpy as np
from scipy.optimize import LinearConstraint, minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medium_overnight.validation import official_udp, solution_payload, verify
from medium_v4.driver import unpack
from medium_coupled.checkpoint import atomic_json

ARCHIVE = ROOT / "results/esa/medium_v4/run_01"
BASE = ARCHIVE / "best.json"


class Paused(Exception):
    pass


def solve(udp, sol, left, margin, shift, stop, accept, counts, extra_edge=None,
          *, cap_flags=None, start_times=None):
    dep, tof, path = sol["dep"], sol["tof"], sol["path"]
    pairs = list(pairwise(path))[left:]
    n = len(pairs)
    x0 = np.array(list(zip(dep[left:], tof[left:]))).ravel()
    prior = dep[left-1] + tof[left-1] if left else 0.
    row = np.arange(n-1)
    chronology = np.zeros((n-1,2*n))
    chronology[row,2*row] = chronology[row,2*row+1] = -1
    chronology[row,2*row+2] = 1
    cached = None
    values = None

    def dv(i, d, f):
        if time.monotonic() >= stop:
            raise Paused
        counts["transfer_evaluations"] += 1
        return float(udp.compute_transfer(*pairs[i], float(d), float(f)))

    def measured(x):
        nonlocal cached, values
        if cached is None or not np.array_equal(x, cached):
            cached = x.copy()
            values = np.array([dv(i,d,f) for i,(d,f) in enumerate(x.reshape(n,2))])
        return values

    initially_extra = (measured(x0) > udp.dv_threshold if cap_flags is None else
                       np.asarray(cap_flags[left:],dtype=bool).copy())
    if extra_edge is not None:
        all_pairs = list(pairwise(path))
        spent = sum(float(udp.compute_transfer(*edge,d,f))>udp.dv_threshold
                    for edge,d,f in zip(all_pairs[:left],dep[:left],tof[:left]))
        if spent + sum(initially_extra)<udp.n_dv_exception:
            initially_extra[extra_edge-left] = True
    caps = np.where(initially_extra,
                    udp.dv_threshold_exception,udp.dv_threshold) - 2e-7
    def constrain(x):
        return (caps - measured(x)) / 100.

    def jac(x):
        rows = np.arange(n)
        start = measured(x)
        matrix = np.zeros((n,2*n))
        h = 3e-7
        for i,(d,f) in enumerate(x.reshape(n,2)):
            matrix[i,2*i] = (start[i]-dv(i,d+h,f))/(100*h)
            matrix[i,2*i+1] = (start[i]-dv(i,d,f+h))/(100*h)
        return matrix

    lower = np.maximum(x0-np.tile([margin, margin/2], n),
                       np.tile([prior,udp.min_tof],n))
    upper = np.minimum(x0+np.tile([margin,margin/2],n),udp.max_time)
    start = (x0.copy() if start_times is None else
             np.asarray(start_times[left:],dtype=float).ravel().copy())
    start[::2] += np.linspace(0.,shift,n)
    weight = np.zeros(2*n)
    weight[-2:] = 1.
    raw = float("inf")

    def submit(x):
        nonlocal raw
        finish = float(weight@x)
        if finish>=raw or min(chronology@x,default=0.) < -1e-9:
            return
        actual = measured(x)
        exceptions = actual > udp.dv_threshold
        if (np.max(actual) > udp.dv_threshold_exception+1e-6 or
                sum(exceptions) > udp.n_dv_exception):
            return
        out = {"path":path,"dep":dep[:left]+x[::2].tolist(),
               "tof":tof[:left]+x[1::2].tolist()}
        if accept(out):
            raw = finish

    fit = minimize(lambda x: float(weight@x), np.clip(start,lower,upper), jac=lambda _:weight,
                   method="SLSQP", bounds=list(zip(lower,upper)), callback=submit,
                   constraints=(LinearConstraint(chronology,0.,np.inf),
                                {"type":"ineq","fun":constrain,"jac":jac}),
                   options={"maxiter":75,"ftol":5e-10})
    submit(fit.x)
    return {"status":int(fit.status),"iterations":int(fit.nit),"raw_finish":float(fit.fun)}


def run(args):
    start = time.monotonic()
    stop = start + args.seconds
    udp = official_udp(ROOT)
    baseline = unpack(json.loads(BASE.read_text()))
    score = verify(udp,baseline)[0]
    folder = args.directory
    folder.mkdir(parents=True,exist_ok=True)
    state_file = folder/"checkpoint.json"
    state = (json.loads(state_file.read_text()) if state_file.exists() else
             {"baseline":score,"best":score,"best_record":None,"trials":[],"counts":{}})
    count = Counter(state["counts"])
    best = baseline
    if state["best_record"]:
        best = unpack(json.loads(Path(state["best_record"]).read_text()))
        assert verify(udp,best)[0] == state["best"]
    elite = [baseline]
    seen = {tuple(baseline["path"])}
    if args.seed_record:
        seed = unpack(json.loads(args.seed_record.read_text()))
        seed_score = verify(udp,seed)[0]
        if seed_score < state["best"]:
            state.update(best=seed_score,best_record=str(args.seed_record.resolve()))
            best = seed
        if tuple(seed["path"]) not in seen:
            elite.insert(0,seed)
            seen.add(tuple(seed["path"]))
    if args.best_record:
        known = unpack(json.loads(args.best_record.read_text()))
        known_score = verify(udp,known)[0]
        if known_score < state["best"]:
            state.update(best=known_score,best_record=str(args.best_record.resolve()))
            best = known
    pool = json.loads((ARCHIVE/"checkpoint.json").read_text())["pool"]
    for entry in pool:
        sol = unpack(entry)
        key = tuple(sol["path"])
        if key not in seen:
            verify(udp,sol)
            elite.append(sol)
            seen.add(key)
    def checkpoint():
        state["counts"] = dict(count)
        atomic_json(state_file,state)

    def halt(*_):
        raise Paused
    signal.signal(signal.SIGTERM,halt)
    signal.signal(signal.SIGINT,halt)
    plans = [(-1,left,margin,shift) for left,margin,shift in
             [(176,1.8,0.),(159,2.5,0.),(125,3.,0.),(0,3.,0.),
              (149,3.5,-.35),(99,4.,-.8),(0,4.,-1.2)]]
    rng = random.Random(94216)
    plans += [(i,left,margin,shift) for left,margin,shift in
              [(165,3.,0.),(130,4.,-.25),(0,4.,-.5)]
              for i in rng.sample(range(len(elite)),len(elite))]
    if args.seed_only:
        plans=[(-1,left,margin,shift) for left,margin,shift in
               [(177,1.2,0.),(172,2.,0.),(165,2.7,0.),(157,3.,0.),
                (145,3.5,0.),(125,4.,0.),(161,2.5,-.18),(155,3.5,-.38)]]
    if args.release_exception:
        plans = [(0,min(left,i),margin,shift,i) for left,margin,shift in
                 [(176,2.5,0.),(171,3.,0.),(160,4.,0.),(160,4.,-.2)]
                 for i in range(179,179-args.exception_span,-1)]
    done = {tuple(row["plan"]) for row in state["trials"]}
    try:
        for plan in plans:
            if plan in done:
                continue
            idx,left,margin,shift,*extra_edges = plan
            source = best if idx<0 else elite[idx]
            def accept(candidate):
                nonlocal best
                count["official_checks"] += 1
                try:
                    f = verify(udp,candidate)
                except ValueError:
                    return False
                count["valid_schedules"] += 1
                if f[0] < state["best"]:
                    checked = verify(official_udp(ROOT),candidate)
                    payload = solution_payload(candidate,checked,provenance={"plan":plan},udp=udp)
                    record = folder/"records"/f"{checked[0]:.12f}_{uuid4().hex[:8]}.json"
                    atomic_json(record,{"solution":candidate,**payload})
                    best = candidate
                    state.update(best=checked[0],best_record=str(record.resolve()))
                    checkpoint()
                    print("RECORD", checked, record,flush=True)
                return True
            before = count["transfer_evaluations"]
            t = time.monotonic()
            result = solve(udp,source,left,margin,shift,stop,accept,count,*extra_edges)
            state["trials"].append({"plan":plan,**result,"seconds":time.monotonic()-t,
                                    "transfer_evaluations":count["transfer_evaluations"]-before})
            checkpoint()
            print(state["trials"][-1],flush=True)
    except Paused:
        pass
    finally:
        state["wall_seconds"] = state.get("wall_seconds",0.) + (time.monotonic()-start)
        checkpoint()
    print("FINISH", state["best"],dict(count),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float,default=60)
    parser.add_argument("--directory",type=Path,required=True)
    parser.add_argument("--seed-record",type=Path)
    parser.add_argument("--best-record",type=Path)
    parser.add_argument("--seed-only",action="store_true")
    parser.add_argument("--release-exception",action="store_true",
                        help="Try spending a spare exception at alternate terminal legs")
    parser.add_argument("--exception-span",type=int,choices=range(1,51),default=4)
    run(parser.parse_args())

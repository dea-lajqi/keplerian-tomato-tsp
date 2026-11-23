"""Feasibility-first continuous repair of a single deadline-join timing deficit."""
import argparse
from itertools import pairwise
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from medium_coupled.high_shell import Campaign
from medium_coupled.run import Paused, solve


class Repair(Campaign):
    def fit(self, seed, left, right, margin, stop, seam=None, kick=0., deadline=None, hard_flags=None, iterations=180):
        path, departures, flights = (seed[k] for k in ("path","dep","tof"))
        edges = list(pairwise(path))[left:right]
        n = len(edges)
        original = np.ravel(np.column_stack((departures[left:right], flights[left:right])))
        prior = departures[left-1]+flights[left-1] if left else 0.
        boundary = departures[right] if right < len(departures) else self.udp.max_time
        last = boundary if deadline is None else min(self.udp.max_time, deadline)

        def cost(i, dep, tof):
            if time.monotonic() >= stop:
                raise Paused
            self.count["repair_lambert"] += 1
            return float(self.udp.compute_transfer(*edges[i],float(dep),float(tof)))

        dep_index = 2*np.arange(n)
        flight_index = dep_index+1
        cache = None
        costs = None
        flags = ([cost(i, original[2*i], original[2*i+1]) > self.normal for i in range(n)]
                 if hard_flags is None else hard_flags[left:right])
        caps = np.where(flags, self.udp.dv_threshold_exception, self.normal) - 2e-7

        def measure(x):
            nonlocal cache, costs
            if cache is None or not np.array_equal(cache, x[:-1]):
                cache = x[:-1].copy()
                costs = np.array([cost(i, x[2*i], x[2*i+1]) for i in range(n)])
            return costs

        def feasible(x):
            return x[-1] + (caps-measure(x))/100.

        def gradient(x):
            jac = np.zeros((n,2*n+1))
            base = measure(x)
            for i in range(n):
                d, t = x[2*i:2*i+2]
                h = 5e-7
                jac[i,2*i] = (base[i]-cost(i,d+h,t))/(100*h)
                jac[i,2*i+1] = (base[i]-cost(i,d,t+h))/(100*h)
            jac[:,-1] = 1.
            return jac

        matrix = np.zeros((n,2*n+1))
        row = np.arange(n-1)
        matrix[row,dep_index[:-1]] = -1.
        matrix[row,flight_index[:-1]] = -1.
        matrix[row,dep_index[1:]] = 1.
        matrix[-1,-3:-1] = -1.
        lower_clock = np.r_[np.zeros(n-1),-last]
        radius = np.tile((margin,margin/2),n)
        lo = np.r_[np.maximum(original-radius,np.tile((prior,self.udp.min_tof),n)),0.]
        hi = np.r_[np.minimum(original+radius,self.udp.max_time),20.]
        start = original.copy()
        if kick:
            # Alternate initial epochs on both sides of the seam. Decision
            # bounds still refer to the original schedule.
            distance = np.arange(left,right)-seam
            start[::2] += kick*np.exp(-np.abs(distance)/7.)*np.sign(distance)
        x0 = np.clip(np.r_[start,.02],lo,hi)
        def loss(x):
            delta = x[:-1]-original
            return float(x[-1] + 1e-5*(delta@delta))

        def loss_jac(x):
            return np.r_[2e-5*(x[:-1]-original),1.]

        result = minimize(loss, x0, jac=loss_jac, method="SLSQP", bounds=Bounds(lo,hi),
                          constraints=(LinearConstraint(matrix,lower_clock,np.inf),
                                       {"type":"ineq","fun":feasible,"jac":gradient}),
                          options={"maxiter":iterations,"ftol":2e-10})
        x = result.x
        dep = departures[:left]+x[:-1:2].tolist()+departures[right:]
        tof = flights[:left]+x[1:-1:2].tolist()+flights[right:]
        candidate = {"path":path,"dep":dep,"tof":tof}
        excess = measure(x)-caps
        info = {"iterations": int(result.nit), "status": int(result.status),
                "max_dv_excess_m_s": float(np.max(excess)),
                "worst_edges": [[*edges[i],float(excess[i])] for i in np.argsort(excess)[-3:][::-1]],
                "min_chronology_slack": float(np.min(matrix@x-lower_clock)),
                "candidate_finish": dep[-1]+tof[-1]}
        return candidate, info

    def run(self):
        sources = []
        for p in self.args.near:
            data = json.loads(p.read_text())
            sources.append((data["chronology_gap"],p,data))
        tested = {tuple(t["key"]) for t in self.state["trials"]}
        for _, path, data in sorted(sources,key=lambda row:(row[0],str(row[1])))[:self.args.limit]:
            index = int(data["trial"][-2])
            plans = [(12,1.8,0.),(23,3.1,0.),(32,4.5,0.)]
            if self.args.phase_kicks:
                plans = [(31,4.5,kick) for kick in self.args.phase_kicks]
            for reach, margin, kick in plans:
                tag = (path.name,reach,margin,kick)
                if tag in tested or time.monotonic()>=self.stop:
                    continue
                start = time.monotonic()
                left, right = max(1,index-reach),min(180,index+reach)
                try:
                    candidate, info = self.fit(data["solution"],left,right,margin,
                                               min(self.stop,start+self.args.trial_seconds),
                                               seam=index,kick=kick)
                    if self.submit(candidate,"feasibility_first",tag):
                        info["valid"] = True
                        try:
                            solve(self.udp,candidate,left,margin,0.,min(self.stop,time.monotonic()+6.),
                                  lambda sol:self.submit(sol,"feasibility_first_polish",tag),self.count)
                        except Paused:
                            pass
                except Paused:
                    info = {"time_cap":True}
                self.state["trials"].append({"key":tag,"gap":data["chronology_gap"],
                                              "seconds":time.monotonic()-start,**info})
                self.save()
                print(tag,info,self.state["best"],flush=True)
        self.state["wall_seconds"] = time.monotonic()-self.started
        self.save()


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",type=Path,required=True)
    parser.add_argument("--seed-record",type=Path,required=True)
    parser.add_argument("--near",type=Path,nargs="+",required=True)
    parser.add_argument("--seconds",type=float,default=145.)
    parser.add_argument("--trial-seconds",type=float,default=6.)
    parser.add_argument("--limit",type=int,default=12)
    parser.add_argument("--phase-kicks",type=float,nargs="*",default=())
    parser.set_defaults(baseline_record=None)
    Repair(parser.parse_args()).run()

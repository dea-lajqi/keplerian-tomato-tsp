"""Reconstruct the audited near-tour and repair it with a hard final deadline."""
import argparse
from itertools import pairwise
import json
from pathlib import Path
import time

from medium_coupled.checkpoint import atomic_json
from medium_frontier.repair import Repair
from medium_coupled.run import Paused, solve


class NearRepair(Repair):
    def inspect(self, candidate, flags):
        rows = []
        costs = []
        for i, ((a,b),d,f) in enumerate(zip(pairwise(candidate["path"]),candidate["dep"],candidate["tof"])):
            dv = float(self.udp.compute_transfer(a,b,d,f))
            costs.append(dv)
            cap = self.udp.dv_threshold_exception if flags[i] else self.normal
            if dv > cap:
                rows.append([i,a,b,d,f,dv,cap,dv-cap])
        return costs, sorted(rows,key=lambda row:-row[-1])

    def accept_hard(self, candidate, method, tag):
        self.count["strict_checks"] += 1
        dv, _ = self.inspect(candidate,[True]*180)
        if max(dv)>self.udp.dv_threshold_exception or sum(v>self.normal for v in dv)>5:
            return False
        return self.submit(candidate,method,tag)

    def tighten_anchors(self, candidate, flags, stop):
        costs,_ = self.inspect(candidate,flags)
        for i,v in enumerate(costs):
            if 600.<v<600.000003 and time.monotonic()<stop:
                # A tiny local fit tightens a historical tolerance-level leg.
                child,_=self.fit(candidate,i,i+1,.0005,min(stop,time.monotonic()+2.),
                                 hard_flags=flags,iterations=60)
                candidate=child
        return candidate

    def run(self):
        raw = json.loads(self.args.near.read_text())
        source = raw["solution"]
        seam = int(raw["trial"][-2])
        flags = [v>self.normal for v in self.inspect(source,[True]*180)[0]]
        replay = self.args.directory / "replayed_near.json"
        if replay.exists():
            data=json.loads(replay.read_text())
            candidate=data["solution"]
        else:
            candidate,report = self.fit(source,seam-31,180,4.5,min(self.stop,time.monotonic()+22.),
                                        seam=seam,kick=.37)
            _, violations=self.inspect(candidate,flags)
            atomic_json(replay,{"solution":candidate,"report":report,"violations":violations})
            print("REPLAY",report,violations[:6],flush=True)
        candidate=self.tighten_anchors(candidate,flags,self.stop)
        incumbent=self.state["baseline"]
        plans=[(l,m,deadline,k) for deadline in (incumbent-.00013,182.20,182.1,181.95)
               for l,m in ((114,1.2),(122,3.5),(127,6.),(0,1.2))
               for k in (0.,.05,-.08,.22,-.23)]
        tested={tuple(t["key"]) for t in self.state["trials"]}
        sources=[candidate,source]
        for j,(left,margin,deadline,kick) in enumerate(plans):
            if time.monotonic()>=self.stop:
                break
            if j>=20 and self.state["best"]<incumbent-.0001:
                deadline=min(deadline,self.state["best"]-.00012)
            key=(j,left,margin,deadline,kick)
            if key in tested:
                continue
            now=time.monotonic()
            chosen=sources[j%len(sources)]
            try:
                sol,info=self.fit(chosen,left,180,margin,min(self.stop,now+self.args.trial_seconds),
                    seam=seam,kick=kick,deadline=deadline,hard_flags=flags,iterations=320)
                _, violations=self.inspect(sol,flags)
                info["hard_violations"]=len(violations)
                info["valid"]=valid=self.accept_hard(sol,"hard_cap_near_repair",key)
                if valid:
                    try:
                        solve(self.udp,sol,left,margin,0.,min(self.stop,time.monotonic()+8.),
                            lambda q:self.accept_hard(q,"hard_cap_near_polish",key),
                            self.count,cap_flags=flags)
                    except Paused:
                        pass
                if len(sources)<10 and info["max_dv_excess_m_s"]<1 and j%2==0:
                    sources.append(sol)
                    atomic_json(self.args.directory/f"basin_{j:03d}.json",
                                {"solution":sol,"provenance":key,"info":info})
            except Paused:
                info={"time_cap":True}
            self.state["trials"].append({"key":key,"seconds":time.monotonic()-now,**info})
            self.save()
            print(key,info,self.state["best"],flush=True)
        self.state["wall_seconds"]=time.monotonic()-self.started
        self.save()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed-record",type=Path,required=True)
    p.add_argument("--near",type=Path,required=True)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--seconds",type=float,default=240.)
    p.add_argument("--trial-seconds",type=float,default=12.)
    p.set_defaults(baseline_record=None)
    NearRepair(p.parse_args()).run()

"""Combine an early exception with a normal-only reconstruction through 41→62's band."""
import argparse
from collections import Counter
from itertools import pairwise
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medium_coupled.nonlocal_routes import Surgery
from pareto_easy import BudgetExpired, arrival


class NormalSpines(Surgery):
    def run(self):
        profiles=json.loads(self.args.profiles.read_text())
        tested={tuple(x["key"]) for x in self.state["trials"]}
        plans=[]
        for row in sorted(profiles.values(),key=lambda p:p["arrival"]):
            legs=tuple(map(tuple,row["legs"]))
            left=len(legs)
            if left not in (112,116,120,121) or row["spend"]!=4:
                continue
            path=tuple(self.build(legs)["path"]+row["remaining"])
            if tuple(path[121:128])!=tuple(self.path[121:128]):
                continue
            monotone=tuple(sorted(path[121:128],key=lambda b:self.inc[b]))
            contenders=[("ordered",monotone),("control",path[121:128])]
            for i in (1,2,3,4,5):
                nearby=list(monotone)
                nearby[i-1],nearby[i]=nearby[i],nearby[i-1]
                contenders.append((f"swap_{i}",tuple(nearby)))
            for name,variation in contenders:
                new=(*path[:121],*variation,*path[128:])
                if all(self.normal_edges[a,b] for a,b in pairwise(new[120:129])):
                    plans.append((row["arrival"],left,name,legs,new))
        plans.sort(key=lambda x:(x[2]!="ordered",x[0]-self.dep[x[1]],x[2],x[1]))
        for day,left,name,prefix,new in plans:
            if time.monotonic()>=self.stop:
                break
            for step in (.078125,.05078125):
                key=(left,round(day,8),name,step)
                if key in tested or time.monotonic()>=self.stop:
                    continue
                start=time.monotonic()
                observations={}
                try:
                    self.complete(new,prefix,self.args.goal,step,
                                  min(self.stop,start+self.args.trial_seconds),key,observations)
                except BudgetExpired:
                    observations["time_cap"]=True
                self.state["trials"].append({"key":key,"seconds":time.monotonic()-start,**observations})
                self.save()
                print(key,observations,flush=True)
        self.state["plans_plausible"]=len(plans)
        self.state["wall_seconds"]=self.state.get("wall_seconds",0.)+time.monotonic()-self.started
        self.save()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--seed-record",type=Path,required=True)
    p.add_argument("--baseline-record",type=Path)
    p.add_argument("--anchors",type=Path,nargs="*",default=())
    p.add_argument("--profiles",type=Path,required=True)
    p.add_argument("--seconds",type=float,default=170.)
    p.add_argument("--trial-seconds",type=float,default=12.)
    p.add_argument("--goal",type=float,default=194.)
    p.set_defaults(mode="normal_spines",beam=None,phase=None,save_prefixes=True,reserve_gate=True)
    NormalSpines(p.parse_args()).run()

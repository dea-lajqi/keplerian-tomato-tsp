"""Pay for the final crossing by removing a discretionary maneuver from early prefixes."""
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


class ExceptionRepair(Surgery):
    def generate(self):
        bank=json.loads(self.args.profiles.read_text())
        for row in sorted(bank.values(),key=lambda x:x["arrival"]):
            legs=tuple(map(tuple,row["legs"]))
            if len(legs)!=160 or row["spend"]!=5:
                continue
            path=tuple(self.build(legs)["path"]+row["remaining"])
            extras=[i for i,leg in enumerate(legs) if leg[-1]>self.normal
                    and min(self.axis(leg[0]),self.axis(leg[1]))*2>=max(self.axis(leg[0]),self.axis(leg[1]))]
            if len(extras)!=2:
                continue
            for leg in legs:
                if leg not in self.anchors[leg[:2]]:
                    self.anchors[leg[:2]].append(leg)
            for idx in extras[::-1]:
                for left,right in ((idx-4,idx+4),(idx-7,idx+5),(idx-5,idx+8),(idx-3,idx+7)):
                    if sum(l[-1]>self.normal for l in legs[:left])>4:
                        continue
                    yield path,legs,row["plan"],idx,left,right

    def run(self):
        tested={tuple(t["key"]) for t in self.state["trials"]}
        for path,legs,plan,idx,left,right in self.generate():
            if time.monotonic()>=self.stop:
                break
            prefix=legs[:left]
            pending=path[left+1:right]
            extras=tuple(leg[:2] for i,leg in enumerate(legs[right:],right)
                         if leg[-1]>self.normal and i!=idx)
            if self.args.fixed:
                ordered=sorted(pending,key=lambda a:self.inc[a])
                candidates=[ordered]
                for i in range(len(ordered)-1):
                    copy=ordered[:]
                    copy[i],copy[i+1]=copy[i+1],copy[i]
                    candidates.append(copy)
                for version,order in enumerate(candidates):
                    key=(idx,left,right,version,"fixed")
                    if key in tested or time.monotonic()>=self.stop:
                        continue
                    new=(*path[:left+1],*order,*path[right:])
                    window=(path[left],*order,path[right])
                    if not all(self.normal_edges[a,b] for a,b in pairwise(window)):
                        continue
                    start=time.monotonic()
                    info={}
                    try:
                        self.complete(new,prefix,self.args.goal,.0546875,
                                      min(self.stop,start+self.args.trial_seconds),key,info,
                                      permitted_extra=extras)
                    except BudgetExpired:
                        info["time_cap"]=True
                    self.state["trials"].append({"key":key,"seconds":time.monotonic()-start,**info})
                    self.save()
                    print(key,info,self.state["best"],flush=True)
                continue
            for gap in (1.5,3.2):
                key=(idx,left,right,gap,",".join(map(str,plan)))
                if key in tested or time.monotonic()>=self.stop:
                    continue
                start=time.monotonic()
                stop=min(self.stop,start+self.args.trial_seconds)
                stats=Counter()
                info={}
                try:
                    # The other discretionary maneuver and the three
                    # prior shell changes stay in the provided prefix.
                    # The remaining shell crossing is protected here.
                    reserve=1+sum(leg[-1]>self.normal for i,leg in enumerate(legs[right:],right)
                                    if i!=idx)
                    finish=arrival(legs[right-1])+gap
                    options=self.block(prefix,pending,path[right],finish,reserve,
                                      min(stop,start+10.),stats)
                    info["block_arrivals"]=[arrival(x[-1]) for x in options[:5]]
                    for updated in options[:2]:
                        new=tuple(self.build(updated)["path"]+list(path[right+1:]))
                        if time.monotonic()>=stop:
                            break
                        self.complete(new,updated,self.args.goal,.0546875,stop,key,info,
                                      permitted_extra=extras)
                except BudgetExpired:
                    info["time_cap"]=True
                self.count.update({k:v for k,v in stats.items() if k!="block_blocked_depth"})
                self.state["trials"].append({"key":key,"seconds":time.monotonic()-start,
                                             "stats":dict(stats),**info})
                self.save()
                print(key,info,self.state["best"],flush=True)
        self.state["wall_seconds"]=self.state.get("wall_seconds",0.)+time.monotonic()-self.started
        self.save()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--seed-record",type=Path,required=True)
    p.add_argument("--baseline-record",type=Path)
    p.add_argument("--anchors",type=Path,nargs="*",default=())
    p.add_argument("--profiles",type=Path,required=True)
    p.add_argument("--seconds",type=float,default=210.)
    p.add_argument("--trial-seconds",type=float,default=16.)
    p.add_argument("--beam",type=int,default=64)
    p.add_argument("--phase",type=float,default=.011)
    p.add_argument("--goal",type=float,default=192.4)
    p.add_argument("--fixed",action="store_true")
    p.set_defaults(mode="exception_repair",save_prefixes=True,reserve_gate=True)
    ExceptionRepair(p.parse_args()).run()

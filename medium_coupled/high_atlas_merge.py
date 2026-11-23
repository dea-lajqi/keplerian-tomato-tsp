"""Merge independently improved high-shell blocks by their exact remaining sets."""
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from global_event_easy import fit_transfer_window
from medium_coupled.checkpoint import atomic_json
from medium_coupled.exact_terminal import ExactTerminal, pareto
from medium_coupled.gate_prefix_polish import PrefixPolish
from medium_coupled.run import Paused, solve
from pareto_easy import BudgetExpired, arrival


class Merge(ExactTerminal):
    def run(self):
        atlas=json.loads(self.args.atlas.read_text())["rows"]
        events={tuple(map(int,k.split(","))):pareto(v) for k,v in atlas.items()}
        edges={key:(tuple(l[2] for l in values),values) for key,values in events.items()}
        groups=defaultdict(list)
        for path in self.args.middle:
            for item in json.loads(path.read_text()).values():
                prefix=tuple(map(tuple,item["legs"]))
                remaining=tuple(item["remaining"])
                if len(prefix)+len(remaining)!=180 or item["spend"]>4:
                    continue
                high=tuple(sorted(a for a in remaining if self.axis(a)>8e6))
                low=tuple(a for a in remaining if self.axis(a)<8e6)
                if high and len(low)==20:
                    groups[high,low].append(prefix)
        file=self.args.directory/"gate_pool.json"
        result=json.loads(file.read_text()) if file.exists() else {}
        visited={tuple(x["key"]) for x in self.state["trials"]}
        for (high,low),members in groups.items():
            if time.monotonic()>=self.stop:
                break
            start=time.monotonic()
            starts={}
            stats=Counter()
            try:
                for prefix in sorted(members,key=lambda legs:arrival(legs[-1]))[:12]:
                    a=prefix[-1][1]
                    r=arrival(prefix[-1])
                    for b in high:
                        if not self.normal_edges[a,b]:
                            continue
                        outgoing=list(self.anchors[a,b])
                        if (a,b) in edges:
                            dep,values=edges[a,b]
                            j=bisect_left(dep,r-1e-9)
                            if j<len(values):
                                outgoing.append(values[j])
                        if time.monotonic()-start<19.:
                            local=min(self.stop,start+19.)
                            options,_=fit_transfer_window(self.udp,a,b,r,r+5.2,self.normal,local,
                                                          random.Random(a*181+b))
                            outgoing.extend(options)
                            stats["middle_bridge_queries"]+=1
                        possible=[l for l in outgoing if l[2]>=r-1e-9 and l[-1]<=self.normal]
                        if not possible:
                            continue
                        leg=min(possible,key=arrival)
                        if b not in starts or arrival(leg)<arrival(starts[b][-1]):
                            starts[b]=(*prefix,leg)
            except BudgetExpired:
                pass
            self.count.update(stats)
            for due in self.args.gates:
                key=high,due
                serial="-".join(map(str,high)),due
                if serial in visited or not starts or time.monotonic()>=self.stop:
                    continue
                begin=time.monotonic()
                stats=Counter()
                detail={}
                try:
                    meters,labels=self.execute(high,{b:arrival(p[-1]) for b,p in starts.items()},
                                               events,due,False)
                    detail.update(meters)
                    detail["gates"]={str(order[-1]):clock for clock,order in labels}
                    for _,order in sorted(labels)[:12]:
                        sol=self.reconstruct(order,starts,edges)
                        first=len(starts[order[0]])-1
                        b=order[-1]
                        def accept(candidate):
                            actual,spent=PrefixPolish.measure(self,candidate,stats)
                            if not actual or len(actual)!=160:
                                return False
                            clock=arrival(actual[-1])
                            name=f"{b}-{serial[0]}"
                            if name not in result or clock<result[name]["arrival"]-1e-9:
                                result[name]={"legs":actual,"remaining":low,"arrival":clock,
                                               "spend":spent,"plan":serial}
                                atomic_json(file,result)
                            return True
                        accept(sol)
                        try:
                            solve(self.udp,sol,max(128,first-8),4.2,0.,min(self.stop,time.monotonic()+4.),
                                  accept,stats)
                        except Paused:
                            pass
                except TimeoutError:
                    detail["time_cap"]=True
                self.count.update(stats)
                self.state["trials"].append({"key":serial,"seconds":time.monotonic()-begin,**detail})
                self.save()
                print(serial,detail,flush=True)
        self.state["wall_seconds"]=self.state.get("wall_seconds",0.)+time.monotonic()-self.started
        self.save()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--seed-record",type=Path,required=True)
    p.add_argument("--baseline-record",type=Path)
    p.add_argument("--anchors",type=Path,nargs="*",default=())
    p.add_argument("--atlas",type=Path,required=True)
    p.add_argument("--middle",type=Path,nargs="+",required=True)
    p.add_argument("--seconds",type=float,default=170.)
    p.add_argument("--gates",type=float,nargs="+",default=(186.2,186.8,187.4,188.2))
    p.set_defaults(mode="high_merge",phase=None,beam=None,goal=None)
    Merge(p.parse_args()).run()

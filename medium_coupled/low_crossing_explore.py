"""Sample all final crossing targets and reconstruct their terminal schedules."""
import argparse
from collections import Counter
import random
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from global_event_easy import fit_transfer_window
from medium_coupled.gate_profile_join import ProfileJoin
from medium_coupled.meet import search
from pareto_easy import BudgetExpired, arrival


class CrossingExplore(ProfileJoin):
    def run(self):
        prefixes=self.profiles()[:self.args.prefix_limit]
        tested={tuple(t["key"]) for t in self.state["trials"]}
        ran=[]
        for k,(prefix,nodes,_) in enumerate(prefixes):
            release=arrival(prefix[-1])
            here=prefix[-1][1]
            key=str(k)
            known=self.state.setdefault("crossings",{})
            if key not in known:
                branches=[]
                for b in nodes:
                    if not self.allowed[here,b] or time.monotonic()>=self.stop:
                        continue
                    try:
                        rows,cost=fit_transfer_window(self.udp,here,b,release,release+1.2,
                                  self.cap,self.stop,random.Random(89101+b))
                    except BudgetExpired:
                        break
                    self.count["continuous_gates"]+=1
                    if rows:
                        leg=min(rows,key=arrival)
                        branches.append(leg)
                known[key]=sorted(branches,key=arrival)
                self.save()
            template=next(self.terminal_orders(nodes),nodes)
            for cross in known[key][:self.args.gate_limit]:
                cross=tuple(cross)
                b=cross[1]
                order=(b,*(x for x in template if x!=b))
                legs=(*prefix,cross)
                virtual={"path":[*self.build(prefix)["path"],*order],
                          "dep":[x[2] for x in legs]+self.dep[len(legs):],
                          "tof":[x[3] for x in legs]+self.tof[len(legs):]}
                ran.append((k,here,b,cross,order,virtual))
        actual_anchors=tuple(l for (a,b),rows in self.anchors.items()
                                 if self.axis(a)<8e6 and self.axis(b)<8e6 for l in rows)
        for when in self.args.deadlines:
            for k,a,b,cross,order,virtual in ran:
                tag=(when,k,a,b)
                if tag in tested or time.monotonic()>=self.stop:
                    continue
                start=time.monotonic()
                stat=Counter()
                try:
                    search(self.udp,virtual,len(order)-1,when,.0390625,3,240,
                           min(self.stop,start+self.args.trial_seconds),stat,
                           lambda q:self.submit(q,"forward_gate_bidirectional_low",tag),
                           repair_near=True,anchor_rows=actual_anchors)
                except BudgetExpired:
                    stat["time_cap"]+=1
                self.count.update({k:v for k,v in stat.items() if k!="best_slack_after_min_tof"})
                self.state["trials"].append({"key":tag,"seconds":time.monotonic()-start,
                                            "gate_arrival":arrival(cross),"stats":dict(stat)})
                self.save()
                print(tag,dict(stat),flush=True)
        self.state["wall_seconds"]=self.state.get("wall_seconds",0.)+time.monotonic()-self.started
        self.save()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--seed-record",type=Path,required=True)
    p.add_argument("--baseline-record",type=Path)
    p.add_argument("--anchors",type=Path,nargs="*",default=())
    p.add_argument("--profiles",type=Path,nargs="+",required=True)
    p.add_argument("--seconds",type=float,default=190.)
    p.add_argument("--trial-seconds",type=float,default=16.)
    p.add_argument("--prefix-limit",type=int,default=1)
    p.add_argument("--gate-limit",type=int,default=7)
    p.add_argument("--deadlines",type=float,nargs="+",default=(192.,192.4,191.75))
    p.set_defaults(mode="crossing_explore",beam=None,goal=None,phase=None)
    CrossingExplore(p.parse_args()).run()

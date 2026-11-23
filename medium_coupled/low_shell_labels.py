"""Use protected high-shell prefixes with flexible backward low-shell coverage."""
import argparse
from collections import Counter
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medium_coupled.boundary import Boundary
from medium_coupled.gate_profile_join import ProfileJoin
from medium_coupled.meet import SideGrid
from pareto_easy import BudgetExpired, arrival


class LowBeam(ProfileJoin):
    def run(self):
        profiles=self.profiles()[:2]
        tested={tuple(x["key"]) for x in self.state["trials"]}
        for day in self.args.deadlines:
            for k,(prefix,rest,plan) in enumerate(profiles):
                for step in (.0390625,.03125):
                    key=day,k,step
                    if time.monotonic()>=self.stop or key in tested:
                        continue
                    release=arrival(prefix[-1])
                    if release+self.udp.min_tof*len(rest)>=day:
                        continue
                    start=time.monotonic()
                    stop=min(self.stop,start+self.args.trial_seconds)
                    stats=Counter()
                    info={}
                    lattice=SideGrid(self.udp,max(0.,release-.65),day,step,.008,stop,True)
                    template=next(self.terminal_orders(rest),rest)
                    try:
                        labels=Boundary.phase(self,lattice,rest,"backward",stop,stats,template=template)
                        info["compatible_sides"]=len(labels)
                        for _,tail in sorted(labels.values(),key=lambda row:row[0])[:10]:
                            if time.monotonic()>=stop:
                                break
                            item=self.join(prefix,tail[::-1],day,step,stop,stats,key)
                            info.setdefault("joins",[]).append(item)
                    except BudgetExpired:
                        info["time_cap"]=True
                    self.count.update(stats)
                    self.state["trials"].append({"key":key,"seconds":time.monotonic()-start,
                                                 "stats":dict(stats),**info})
                    self.save()
                    print(key,dict(stats),info,flush=True)
        self.state["wall_seconds"]=self.state.get("wall_seconds",0.)+time.monotonic()-self.started
        self.save()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--seed-record",type=Path,required=True)
    p.add_argument("--baseline-record",type=Path)
    p.add_argument("--anchors",type=Path,nargs="*",default=())
    p.add_argument("--profiles",type=Path,nargs="+",required=True)
    p.add_argument("--seconds",type=float,default=215.)
    p.add_argument("--trial-seconds",type=float,default=43.)
    p.add_argument("--beam",type=int,default=80)
    p.add_argument("--repair-open",action="store_true")
    p.add_argument("--deadlines",type=float,nargs="+",default=(192.6,192.,190.))
    p.set_defaults(mode="low_shell_labels",goal=None,phase=None)
    LowBeam(p.parse_args()).run()

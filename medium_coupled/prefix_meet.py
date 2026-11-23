"""Meet short backward deadline profiles against saved exact early prefixes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medium_coupled.high_shell import Campaign
from medium_coupled.meet import search
from pareto_easy import BudgetExpired, arrival


class PrefixMeet(Campaign):
    def run(self):
        profiles=json.loads(self.args.profiles.read_text())["prefix_profiles"]
        options={}
        for rows in profiles.values():
            for idx,legs in rows.items():
                cut=int(idx)
                if not 168<=cut<=175:
                    continue
                for data in legs:
                    prefix=tuple(map(tuple,data))
                    count=sum(leg[-1]>self.normal for leg in prefix)
                    key=(cut,count)
                    if key not in options or arrival(prefix[-1])<arrival(options[key][-1]):
                        options[key]=prefix
        tested={tuple(t["key"]) for t in self.state["trials"]}
        for day in self.args.deadlines:
            for cut in (173,171,169,175):
                for count in range(4,6):
                    prefix=options.get((cut,count))
                    if prefix is None:
                        continue
                    if arrival(prefix[-1])+(180-cut)*self.udp.min_tof>=day:
                        continue
                    draft=self.build((*prefix,*self.legs[cut:]))
                    for step,shift in ((.03125,2),(.0234375,3)):
                        key=(cut,count,day,step,shift)
                        if key in tested or time.monotonic()>=self.stop:
                            continue
                        start=time.monotonic()
                        stop=min(self.stop,start+self.args.trial_seconds)
                        stats=Counter()
                        try:
                            search(self.udp,draft,180-cut,day,step,shift,260,stop,stats,
                                  lambda q:self.submit(q,"early_prefix_backward_join",key),
                                  repair_near=True)
                        except BudgetExpired:
                            stats["time_cap"]=1
                        self.count.update({k:v for k,v in stats.items() if k!="best_slack_after_min_tof"})
                        self.state["trials"].append({"key":key,"seconds":time.monotonic()-start,
                              "prefix_arrival":arrival(prefix[-1]),"stats":dict(stats)})
                        self.save()
                        print(key,dict(stats),flush=True)
            if self.state["best"]<day:
                break
        self.state["wall_seconds"]=self.state.get("wall_seconds",0.)+time.monotonic()-self.started
        self.save()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profiles",type=Path,required=True)
    p.add_argument("--seed-record",type=Path,required=True)
    p.add_argument("--baseline-record",type=Path,required=True)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--seconds",type=float,default=140.)
    p.add_argument("--trial-seconds",type=float,default=12.)
    p.add_argument("--deadlines",type=float,nargs="+",default=(195.,195.5,196.,196.5))
    PrefixMeet(p.parse_args()).run()

"""Keep the best early high-shell prefix and search continuous terminal windows."""
import argparse
from collections import Counter
import math
from pathlib import Path
import random
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from global_event_easy import fit_transfer_window
from medium_coupled.endgame import decode
from medium_coupled.high_shell import Campaign
from pareto_easy import BudgetExpired, arrival


class FinishEarly(Campaign):
    def build_prefixes(self):
        i=self.args.edge_index
        stop=min(self.stop,time.monotonic()+self.args.profile_seconds)
        spend_extra=not self.args.normal_pivot
        if sum(self.exception)+int(spend_extra and not self.exception[i])>self.udp.n_dv_exception:
            raise ValueError("Selected transfer requires a spare ΔV exception")
        phases=((.078125,()),(.0625,()),(.046875,()),(.09375,()),(.03125,()))
        if self.args.mixed:
            phases=((.046875,((.03125,.006),(.0234375,.009))),
                    (.03125,((.02734375,.002),(.01953125,.012))),
                    (.0234375,((.015625,.003),(.01953125,.008))))
        for n,(step,mixed) in enumerate(phases):
            key=f"{step}-{self.args.profile_deadline}-{bool(mixed)}"
            existing=self.state.get("prefix_profiles",{})
            if key in existing or (not mixed and self.args.profile_deadline==198.5
                                   and i==125 and str(step) in existing):
                continue
            if time.monotonic()>=stop:
                break
            try:
                if self.args.reuse_prefix:
                    branches=[self.legs[i]]
                else:
                    cap=self.cap if spend_extra else self.normal
                    branches,_=fit_transfer_window(self.udp,*self.edges[i],arrival(self.legs[i-1]),
                             arrival(self.legs[i]),cap,stop,random.Random(9298+i+n))
                    if spend_extra:
                        branches=sorted((l for l in branches if l[-1]>self.normal),key=arrival)
                    else:
                        branches=sorted((*branches,self.legs[i]),key=arrival)
                if not branches:
                    continue
                saved={}
                flags=self.exception.copy()
                flags[i]=int(branches[0][-1]>self.normal)
                self.sweep(self.path,(*self.legs[:i],branches[0]),flags,stop,step,
                           goal=self.args.profile_deadline,saved=saved,mixed=mixed)
            except BudgetExpired:
                break
            self.state.setdefault("prefix_profiles",{})[key]={
                str(cut):[list(legs) for _,legs in sorted(front.values(),key=lambda row:row[0])]
                for cut,front in saved.items() if cut>=self.args.min_cut}
            self.save()

    def continuous(self,prefix,targets,day,stop,seed,limit):
        spend=self.udp.n_dv_exception-sum(leg[-1]>self.normal for leg in prefix)
        frontier={(0,prefix[-1][1],0):(arrival(prefix[-1]),())}
        counts=Counter()
        for level in range(len(targets)):
            new={}
            for (mask,a,e),(release,chain) in frontier.items():
                due=day-(len(targets)-level-1)*self.udp.min_tof
                if due-release<=self.udp.min_tof:
                    continue
                for j,b in enumerate(targets):
                    if mask & (1<<j):
                        continue
                    ceilings=(self.normal,self.cap) if e<spend else (self.normal,)
                    measured=[]
                    for cap in ceilings:
                        for side in range(limit):
                            counts["window_solves"]+=1
                            self.count["terminal_continuous_queries"]+=1
                            rows,cost=fit_transfer_window(self.udp,a,b,release,due,cap,stop,seed)
                            measured.extend(rows)
                            if math.isfinite(cost):
                                edge_key=f"best_dv_{a}_{b}"
                                counts[edge_key]=min(float(cost),counts.get(edge_key,float("inf")))
                            if rows:
                                break
                    for leg in measured:
                        k=e+int(leg[-1]>self.normal)
                        end=arrival(leg)
                        key=(mask|(1<<j),b,k)
                        if k<=spend and (key not in new or end<new[key][0]):
                            new[key]=(end,(*chain,leg))
            frontier=new
            if not frontier:
                break
        for (mask,_,_),(_,chain) in frontier.items():
            if mask.bit_count()==len(targets):
                self.submit(self.build((*prefix,*chain)),"continuous_early_finish",
                            (len(prefix),day))
        return dict(counts)

    def run(self):
        self.build_prefixes()
        profiles=self.state.get("prefix_profiles",{})
        tested={tuple(x["key"]) for x in self.state["trials"]}
        for day in self.args.deadlines:
            for cut in (178,177,176,175,173,171,169):
                options={}
                for profile in profiles.values():
                    for legs in profile.get(str(cut),[]):
                        prefix=tuple(map(tuple,legs))
                        e=sum(l[-1]>self.normal for l in prefix)
                        if e not in options or arrival(prefix[-1])<arrival(options[e][-1]):
                            options[e]=prefix
                for spend,prefix in sorted(options.items()):
                    key=(day,cut,spend)
                    if key in tested or time.monotonic()>=self.stop:
                        continue
                    start=time.monotonic()
                    release=arrival(prefix[-1])
                    if release+(180-cut)*self.udp.min_tof>=day:
                        continue
                    local=min(self.stop,start+10.)
                    stats=Counter()
                    info={"release":release}
                    draft=self.build((*prefix,*self.legs[cut:]))
                    try:
                        if cut>=175:
                            info["continuous"]=self.continuous(prefix,self.path[cut+1:],day,local,
                                random.Random(3301+cut+int(day*100)),2 if cut>=177 else 1)
                        if time.monotonic()<local:
                            decode(self.udp,draft,cut,day,.013671875,.003,local,stats,
                                   lambda q:self.submit(q,"fine_early_finish",key),
                                   extra_anchors=self.extra_anchors)
                    except BudgetExpired:
                        info["time_cap"]=True
                    self.count.update({k:v for k,v in stats.items() if k!="max_depth"})
                    self.state["trials"].append({"key":key,"seconds":time.monotonic()-start,
                                                   "lattice":dict(stats),**info})
                    self.save()
                    print(key,info,flush=True)
            if self.state["best"]<day:
                break
        self.state["wall_seconds"]=self.state.get("wall_seconds",0.)+time.monotonic()-self.started
        self.save()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--seed-record",type=Path,required=True)
    p.add_argument("--baseline-record",type=Path,required=True)
    p.add_argument("--seconds",type=float,default=150.)
    p.add_argument("--mixed",action="store_true")
    p.add_argument("--profile-deadline",type=float,default=198.5)
    p.add_argument("--profile-seconds",type=float,default=20.)
    p.add_argument("--min-cut",type=int,default=173)
    p.add_argument("--edge-index",type=int,choices=range(1,178),default=125)
    p.add_argument("--reuse-prefix",action="store_true")
    p.add_argument("--normal-pivot",action="store_true")
    p.add_argument("--deadlines",type=float,nargs="+",default=(198.,198.25,198.5,199.,199.65))
    FinishEarly(p.parse_args()).run()

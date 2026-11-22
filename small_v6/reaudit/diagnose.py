"""Instrument matched Small segment schedules and test wider phase coverage."""
import argparse
import json
import time
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from small_v6.search import FILES, GOALS, Search, arrival
from pareto_easy import BudgetExpired
from global_dp_easy import Grid


class Diagnostic(Search):
    def grid_phase(self,reverse,step,phase):
        key=("coverage",reverse,step,phase)
        if key not in self.grids:
            g=Grid(self.module,step,phase,self.deadline,finish_day=112. if reverse else None,
                   directional=True)
            g.horizon=112. if reverse else 116.
            if reverse:
                g.z_speed=-g.z_speed
            self.grids[key]=g
        return self.grids[key]

    def edge_options(self,a,b,clock,reverse,allow,fine=False):
        options=super().edge_options(a,b,clock,reverse,allow,fine)
        if not self.args.coverage:
            return options
        margin=min((clock-l[2] if reverse else arrival(l)-clock for l in options),default=1000.)
        cross=(a in self.group)!=(b in self.group)
        if not cross and margin<2.2:
            return options
        rclock=max(0.,112.-clock) if reverse else clock
        cap=self.extra if allow else self.limit
        for step,phase in ((.046875,.007),(.03125,.021)):
            grid=self.grid_phase(reverse,step,phase)
            for l in grid.edges(b if reverse else a,a if reverse else b,rclock,5.2,10.5,allow):
                if l is None or l[3]<self.udp.min_tof:
                    continue
                if reverse:
                    d=112.-arrival(l)
                    if d<0 or d+l[3]>clock+8e-7:
                        continue
                    dv=self.cost(a,b,d,l[3])
                    if dv<=cap:
                        options.append((a,b,d,l[3],dv))
                elif l[2]>=clock-8e-7 and l[4]<=cap:
                    options.append(l)
        ranks={}
        for l in options:
            e=int(l[4]>self.limit)
            rank=-l[2] if reverse else arrival(l)
            if e not in ranks or rank<ranks[e][0]:
                ranks[e]=(rank,l)
        self.stats["expanded_phase_queries"]+=1
        return [l for _,l in ranks.values()]

    def chain(self,path,clock,budget,reverse=False,fine=False):
        states,depth=super().chain(path,clock,budget,reverse,fine)
        if self.trace is not None:
            if not states and depth<len(path)-1:
                pairs=list(zip(path,path[1:]))
                pair=pairs[-depth-1] if reverse else pairs[depth]
                self.failures[("reverse" if reverse else "forward",pair,budget)]+=1
            for used,(day,legs) in states.items():
                if legs:
                    self.trace[("reverse" if reverse else "forward",tuple(path),clock,budget,used)]=(day,legs)
        return states,depth

    def experiment(self,route,m,tail,goal,trial,fine=False):
        self.trace={}
        super().experiment(route,m,tail,goal,trial,fine)
        selected=trial.get("terminal_gap")
        if selected is not None and selected>self.best_slack.get(goal,-1e6):
            self.best_slack[goal]=selected
            # Recover a coherent forward chain for the final high segment.
            ending=route[m+3:]
            branches=[(day,legs,clock,budget,used) for (d,p,clock,budget,used),(day,legs)
                      in self.trace.items() if d=="forward" and p==ending and
                      abs(tail[0][2]-day-selected)<1e-7]
            if branches:
                day,legs,release,budget,used=branches[0]
                _,back=self.chain(ending,tail[0][2],budget,True,fine)
                backward=[(count,x) for (d,p,_,b,count),x in self.trace.items()
                          if d=="reverse" and p==ending and b==budget and count<=budget]
                gaps=[]
                if backward:
                    _,(_,latest)=max(backward,key=lambda row:row[1][0])
                    prev=release
                    for f,r in zip(legs,latest):
                        gaps.append({"arc":f[:2],"forward_release":prev,"forward_departure":f[2],
                                     "forward_arrival":arrival(f),"backward_latest_departure":r[2],
                                     "phase_wait":f[2]-prev,"arrive_minus_latest_release":prev-r[2]})
                        prev=arrival(f)
                self.bottlenecks[str(goal)]={"route":route,"gate":tail[0][2],
                   "terminal_gap":selected,"segment_start":release,"suffix_exceptions":used,
                   "available_exceptions":budget,"forward_suffix":legs,"latest_backward_depth":back,
                   "arc_trace":gaps}
        if self.args.reallocate:
            self.reallocate(route,m,tail,trial,fine)
        self.trace=None

    def reallocate(self,route,m,tail,trial,fine):
        # Reserve one exception for the final high-shell segment, with the
        # other three and the terminal crossing accounted for explicitly.
        initial,left,middle,right=route[:4],route[3:m],route[m-1:m+4],route[m+3:]
        starts,_=self.chain(initial,0.,1,False,fine)
        for used,(day,pre) in starts.items():
            lefts,_=self.chain(left,day,3-used-2,False,fine)
            for lu,(release,before) in lefts.items():
                mids,_=self.chain(middle,release,3-used-lu,False,fine)
                for mu,(at,passage) in mids.items():
                    remaining=4-used-lu-mu
                    ends,_=self.chain(right,at,remaining,False,fine)
                    for count,(finish,after) in ends.items():
                        gap=tail[0][2]-finish
                        trial["reserved_late_gap"]=max(gap,trial.get("reserved_late_gap",-1e6))
                        self.stats["reserved_late_profiles"]+=1
                        if gap>=-8e-7:
                            self.submit((*pre,*before,*passage,*after,*tail),trial)
        profile=self.bottlenecks.get(str(trial["goal"]))
        if profile and profile["route"]==route and profile["available_exceptions"]==0:
            # Diagnostic only: does finding one spare exception upstream
            # have enough value to close the observed timing deficit?
            options,_=self.chain(right,profile["segment_start"],1,False,fine)
            if 1 in options:
                finish,legs=options[1]
                profile["one_spare_counterfactual"]={"finish":finish,"gate":profile["gate"],
                    "used_on":[l[:2] for l in legs if l[4]>self.limit]}
                self.stats["spare_counterfactuals"]+=1

    def run_matched(self,checkpoint):
        source=json.loads(checkpoint.read_text())
        measured={ (l[0][0],round(l[0][2],6),l[-1][1]):l
                   for tails in self.tails.values() for l in tails}
        attempted=source["attempted"]
        ref=set(zip(self.seed_path,self.seed_path[1:]))
        curated=[]
        for goal in GOALS:
            leaders=sorted((r for r in source["history"] if r["goal"]==goal and "terminal_gap" in r),
                           key=lambda r:-r["terminal_gap"])[:14]
            count=0
            for trial in leaders:
                for item in attempted:
                    route=tuple(item[:-3])
                    g,gate,end=item[-3:]
                    if (g!=goal or gate!=round(trial["gate"],6) or route[-1]!=trial["crossing"][0]
                            or sum((a,b) not in ref for a,b in zip(route,route[1:]))!=trial["changed_prefix_arcs"]):
                        continue
                    tail=measured.get((route[-1],gate,end))
                    middle=trial["left_size"]+3
                    if not tail or tail[0][1]!=trial["crossing"][1] or arrival(tail[-1])>goal:
                        continue
                    if route[middle] not in self.group or route[middle-1] in self.group:
                        continue
                    key=(route,gate,end,goal)
                    if any(row[0]==key for row in curated):
                        continue
                    curated.append((key,route,middle,tail))
                    count+=1
                    if count>=self.args.cases:
                        break
                if count>=self.args.cases:
                    break
        try:
            for key,route,m,tail in sorted(curated,key=lambda row:(GOALS.index(row[0][-1]),row[1])):
                goal=key[-1]
                self.check()
                trial={"goal":goal,"gate":tail[0][2],"crossing":tail[0][:2],
                       "ordered_middle":route[m:m+3],"terminal_finish":arrival(tail[-1])}
                self.hist.append(trial)
                self.experiment(route,m,tail,goal,trial)
                if self.args.coverage and -3.0<trial.get("terminal_gap",-1e6)<0.:
                    self.experiment(route,m,tail,goal,trial,True)
                if self.args.joint_subset:
                    alternative={"goal":goal,"gate":tail[0][2],"crossing":tail[0][:2],
                                 "method":"joint membership and timing"}
                    self.hist.append(alternative)
                    self.subset_bridge(route,m,tail,alternative)
                if self.args.alternate:
                    original_order=tuple(l[1] for l in tail)
                    another=[t for (a,order),tails in self.tails.items() if a==tail[0][0]
                             and order!=original_order and set(order)==set(original_order)
                             for t in tails if arrival(t[-1])<=goal]
                    another.sort(key=lambda t:-t[0][2])
                    variants=list(self.partitions(route))[1:3]
                    if not variants and another:
                        variants=[(route,m)]
                    for (alt,mid) in variants:
                        measured=another[0] if another else tail
                        alt_trial={"goal":goal,"gate":measured[0][2],"crossing":measured[0][:2],
                                   "ordered_middle":alt[mid:mid+3],"terminal_finish":arrival(measured[-1]),
                                   "alternate_terminal_order":tuple(l[1] for l in measured)!=original_order,
                                   "alternate_membership":set(alt[3:mid])!=set(route[3:m])}
                        self.hist.append(alt_trial)
                        self.experiment(alt,mid,measured,goal,alt_trial)
                if len(self.hist)%5==0:
                    self.save()
        except (BudgetExpired,KeyboardInterrupt):
            pass
        finally:
            self.save()
            print("matched",len(self.hist),dict(self.stats),self.best,flush=True)

    def save(self):
        super().save()
        from small_v2 import atomic_json
        atomic_json(self.out/"bottlenecks.json",{"coverage":self.args.coverage,
            "best_terminal_profiles":self.bottlenecks,
            "most_frequent_stalls":[(d,a,b,budget,n)
                for (d,(a,b),budget),n in self.failures.most_common(40)]})



if __name__=="__main__":
    from collections import Counter
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seconds",type=float,default=115.)
    p.add_argument("--matched",type=Path)
    p.add_argument("--cases",type=int,default=4)
    p.add_argument("--joint-subset",action="store_true")
    p.add_argument("--reallocate",action="store_true")
    p.add_argument("--alternate",action="store_true")
    p.add_argument("--coverage",action="store_true")
    p.add_argument("--phase",type=float,default=.039)
    p.add_argument("--seed",type=int,default=86102)
    p.add_argument("--variants",type=int,default=4)
    args=p.parse_args()
    args.atlas=FILES/"small_v4/diagnostic/window_e3e82fc984_checkpoint.json"
    args.resume=None
    args.rebuild=True
    args.subset=False
    args.radius=5
    args.max_window=17
    args.beam=48
    worker=Diagnostic(args)
    worker.trace=None
    worker.best_slack={}
    worker.bottlenecks={}
    worker.failures=Counter()
    if args.matched:
        worker.run_matched(args.matched)
    else:
        worker.run()

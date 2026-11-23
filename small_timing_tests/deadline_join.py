"""Small deadline-first forward/backward join using continuous Lambert queries."""
import argparse
from collections import Counter, defaultdict
from itertools import permutations
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from small_timing_tests.controlled import Results, corpus
from continuous_windows_easy import FILES
from global_dp_easy import Grid, evaluator
from global_event_easy import fit_transfer_window, geometry
from focus_easy import retime
from pareto_easy import BudgetExpired, arrival
from small_v2 import atomic_json


def exceptions(legs,threshold):
    return sum(l[4]>threshold for l in legs)


def path_of(legs):
    return (legs[0][0],*(l[1] for l in legs))


class Join:
    def __init__(self,args):
        self.args=args
        self.start=time.monotonic()
        self.stop=self.start+args.seconds
        self.module=evaluator()
        self.udp=self.module.udp
        self.rng=random.Random(32008)
        self.seed,self.orders,_=corpus(self.udp)
        self.normal,self.allowed=geometry(self.udp,self.module.MU_MOON)[0]
        self.atlas=json.loads((FILES/"small_v4/diagnostic/window_e3e82fc984_checkpoint.json").read_text())["windows"]
        self.results={day:Results(self.udp) for day in (112.,111.)}
        self.count=Counter()
        self.trials=[]
        self.grids={}

    def check(self):
        if time.monotonic()>=self.stop:
            raise BudgetExpired

    def reverse_label(self,path,gate,budget,parent):
        base=112.
        if base not in self.grids:
            grid=Grid(self.module,.0625,.019,self.stop,finish_day=base,directional=True)
            grid.z_speed=-grid.z_speed
            self.grids[base]=grid
        reverse=self.grids[base]
        anchors=defaultdict(list)
        # retime needs an anchored release. Seed the first reverse leg
        # independently at the requested epoch, then use it as its prefix.
        for l in parent:
            a,b,d,f,dv=l
            dep=base-d-f
            if dep>=0.:
                anchors[b,a].append((b,a,dep,f,dv))
        a,b=path[-2:]
        first=list(reverse.edges(b,a,base-gate,4.5,9.5,budget>0))
        first+=anchors.get((b,a),[])
        selected={}
        for l in first:
            if l is None or l[2]+8e-7<base-gate:
                continue
            e=int(l[4]>self.udp.dv_threshold)
            if e<=budget and (e not in selected or arrival(l)<arrival(selected[e])):
                selected[e]=l
        for first_leg in selected.values():
            timed,_=retime(reverse,path[::-1],anchors,prefix=(first_leg,),
                           max_exceptions=budget,wait=4.5,flight=9.5)
            for candidate in timed:
                legs=[]
                for b,a,d,f,_ in reversed(candidate):
                    dep=base-d-f
                    if dep<0.:
                        break
                    dv=float(self.udp.compute_transfer(a,b,dep,f))
                    if dv>self.udp.dv_threshold_exception+1e-6:
                        break
                    legs.append((a,b,dep,f,dv))
                else:
                    if arrival(legs[-1])<=gate+8e-7 and exceptions(legs,self.udp.dv_threshold)<=budget:
                        yield tuple(legs)

    def bridges(self,route,release,due,budget,trial,anchor=()):
        """Search the actual triangular intervals under each exception budget."""
        pairs=list(zip(route,route[1:]))
        states={0:(release,())}
        for i,(a,b) in enumerate(pairs):
            nxt={}
            for used,(day,legs) in states.items():
                room=due-(len(pairs)-i-1)*self.udp.min_tof
                if room<=day+self.udp.min_tof:
                    continue
                caps=[self.udp.dv_threshold] if self.normal[a,b] else []
                if used<budget:
                    caps.append(self.udp.dv_threshold_exception)
                for cap in caps:
                    self.count["continuous_lambert_queries"]+=1
                    found,cost=fit_transfer_window(self.udp,a,b,day,room,cap,self.stop,self.rng)
                    found=list(found)
                    for leg in anchor:
                        if (leg[:2]==(a,b) and day<=leg[2]+8e-7
                                and arrival(leg)<=room+8e-7 and leg[4]<=cap):
                            found.append(leg)
                    if cost<float("inf"):
                        trial.setdefault("oracle_minima_m_s",[]).append(round(cost,6))
                    for l in found:
                        e=used+int(l[4]>self.udp.dv_threshold)
                        if e<=budget and (e not in nxt or arrival(l)<nxt[e][0]):
                            nxt[e]=(arrival(l),(*legs,l))
            states=nxt
            if not states:
                break
        return [row[1] for row in states.values()] if states else []

    def submit(self,goal,legs):
        path=path_of(legs)
        self.results[goal].add(path,legs)
        if arrival(legs[-1])<=goal+1e-7:
            self.count[f"candidates_at_{goal}"]+=1

    def search(self,goal):
        proposals=[]
        for row in self.orders[:5]:
            order=tuple(row["path"])
            original=tuple(map(tuple,row["transfers"]))
            group=set(order[-3:])
            seed_tail=original[-3:]
            tails=[tuple(map(tuple,t)) for t in self.atlas
                   if t[0][0]==order[-4] and set(l[1] for l in t)==group
                   and arrival(t[-1])<=goal and exceptions(t,self.udp.dv_threshold)==1]
            if arrival(seed_tail[-1])<=goal:
                tails.append(seed_tail)
            tails.sort(key=lambda t:-t[0][2])
            for tail in tails[:self.args.tails]:
                suffix_start=len(order)-7
                suffix_order=order[suffix_start:-3]
                for suffix in self.reverse_label(suffix_order,tail[0][2],1,original):
                    post=(*suffix,*tail)
                    used=exceptions(post,self.udp.dv_threshold)
                    for cut in (suffix_start-1,suffix_start-2,suffix_start-3):
                        pre=original[:cut]
                        unvisited=set(range(49))-set(path_of(pre))-set(path_of(post))
                        if set(path_of(pre))&set(path_of(post)) or len(unvisited)!=suffix_start-cut-1:
                            continue
                        budget=self.udp.n_dv_exception-used-exceptions(pre,self.udp.dv_threshold)
                        if budget>=0:
                            for order_middle in permutations(sorted(unvisited)):
                                route=(pre[-1][1],*order_middle,post[0][0])
                                edges=tuple(zip(route,route[1:]))
                                if (all(self.allowed[a,b] for a,b in edges) and
                                       sum(not self.normal[a,b] for a,b in edges)<=budget):
                                    gap=post[0][2]-arrival(pre[-1])
                                    proposals.append((gap,route,pre,post,budget))
        proposals.sort(key=lambda row:(-row[0],len(row[1])))
        self.count[f"compatible_set_budget_{goal}"]=len(proposals)
        checked=set()
        for gap,route,pre,post,budget in proposals:
            self.check()
            key=(route,round(arrival(pre[-1]),6),round(post[0][2],6),budget)
            if key in checked:
                continue
            checked.add(key)
            if len(checked)>self.args.joins:
                break
            trial={"goal":goal,"bridge_order":route,"release":arrival(pre[-1]),
                   "due":post[0][2],"time_interval":gap,"remaining_exceptions":budget}
            self.trials.append(trial)
            for bridge in self.bridges(route,arrival(pre[-1]),post[0][2],budget,trial):
                self.submit(goal,(*pre,*bridge,*post))

    def run(self):
        control=Results(self.udp)
        try:
            original=tuple(map(tuple,self.seed["transfers"]))
            pre,bridge,post=original[:-7],original[-7:-6],original[-6:]
            budget=self.udp.n_dv_exception-exceptions((*pre,*post),self.udp.dv_threshold)
            for legs in self.bridges(path_of(bridge),arrival(pre[-1]),post[0][2],
                                     budget,{},anchor=bridge):
                complete=(*pre,*legs,*post)
                control.add(path_of(complete),complete)
            for goal in (112.,111.):
                self.search(goal)
        except BudgetExpired:
            self.count["time_limit"]=1
        elapsed=time.monotonic()-self.start
        output={"elapsed":elapsed,"counts":dict(self.count),"trials":self.trials,
                "anchor_control":control.report(),
                "deadlines":{str(g):out.report() for g,out in self.results.items()}}
        atomic_json(self.args.output,output)
        print(dict(self.count),elapsed,{g:v["independent"] for g,v in output["deadlines"].items()},flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seconds",type=float,default=75.)
    p.add_argument("--tails",type=int,default=2)
    p.add_argument("--joins",type=int,default=14)
    Join(p.parse_args()).run()

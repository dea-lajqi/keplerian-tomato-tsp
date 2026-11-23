"""Deadline-first reconstruction of both sides of the final Medium shell crossing."""
import argparse
from collections import Counter
import itertools
import random
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from global_event_easy import fit_transfer_window
from medium_coupled.meet import SideGrid
from medium_coupled.nonlocal_routes import Surgery
from medium_coupled.run import Paused, solve
from pareto_easy import BudgetExpired, arrival


class Boundary(Surgery):
    def phase(self,grid,targets,side,stop,stats,start_node=None,template=None):
        full=(1<<len(targets))-1
        if side=="forward":
            states={(0,start_node):(0.,())}
            depth=len(targets)
        else:
            states={(1<<i,b):(0.,()) for i,b in enumerate(targets)}
            depth=len(targets)-1
        reached=0
        try:
            for i in range(depth):
                update={}
                for (mask,a),(clock,chain) in states.items():
                    grid.check()
                    remaining=full^mask
                    while remaining:
                        bit=remaining&-remaining
                        remaining^=bit
                        b=targets[bit.bit_length()-1]
                        x,y=(a,b) if side=="forward" else (b,a)
                        if not self.normal_edges[x,y]:
                            continue
                        choices=list(self.anchors[x,y])
                        for l in grid.edges(a,b,clock,2.0 if side=="forward" else 1.4,
                                            6.2 if side=="forward" else 2.1,False):
                            if l is None:
                                continue
                            d,f=(grid.origin-arrival(l),l[3]) if grid.reverse else (grid.origin+l[2],l[3])
                            if d<0:
                                continue
                            v=float(self.udp.compute_transfer(x,y,d,f))
                            choices.append((x,y,d,f,v))
                            stats["exact_replays"]+=1
                        for leg in choices:
                            release=grid.origin-arrival(leg) if grid.reverse else leg[2]-grid.origin
                            finish=release+leg[3]
                            if release<clock-1e-9 or leg[-1]>self.normal or finish>grid.horizon+1e-9:
                                continue
                            key=(mask|bit,b)
                            if key not in update or finish<update[key][0]:
                                update[key]=(finish,(*chain,leg))
                stats["labels_"+side]+=len(update)
                if not update:
                    break
                reached=i+1
                states={}
                masks=Counter()
                guide=None
                if template:
                    packed=template[:i+1] if side=="forward" else template[-(i+2):]
                    guide=sum(1<<targets.index(node) for node in packed)
                ranked=sorted(update.items(),key=lambda p:p[1][0])
                if guide is not None:
                    guided=sorted(update.items(),
                                  key=lambda p:((p[0][0]^guide).bit_count(),p[1][0]))
                    ranked=guided[:max(4,self.args.beam//3)]+ranked
                for k,v in ranked:
                    if k in states:
                        continue
                    if masks[k[0]]>=6:
                        continue
                    masks[k[0]]+=1
                    states[k]=v
                    if len(states)>=self.args.beam:
                        break
        finally:
            stats.update(grid.stats)
            stats["depth_"+side]=reached
        return {a:(clock,chain) for (mask,a),(clock,chain) in states.items() if mask==full}

    def run_case(self,width,deadline,step,stop,stats):
        gate=self.shells[-1]
        l=gate-width
        high=self.path[l+1:gate+1]
        low=self.path[gate+1:]
        prefix=tuple(self.legs[:l])
        spent=sum(leg[-1]>self.normal for leg in prefix)
        if spent>=5:
            return
        origin=arrival(prefix[-1])
        if deadline<=origin:
            return
        # The high and low parts use normal transfers. The final crossing
        # gets the reserved exception; no timing/resource allocation is inferred afterward.
        forward_grid=SideGrid(self.udp,origin,deadline,step,.017,stop,False)
        reverse_floor=max(origin,self.legs[gate][2]-6.)
        backward_grid=SideGrid(self.udp,reverse_floor,deadline,max(.0234375,step/2),.006,stop,True)
        back=self.phase(backward_grid,low,"backward",stop,stats)
        if not back:
            return
        if time.monotonic()>=stop:
            return
        front=self.phase(forward_grid,high,"forward",stop,stats,start_node=self.path[l])
        stats["forward_ends"]=len(front)
        stats["backward_starts"]=len(back)
        opts=[]
        maximum_slack=None
        for a,(time_f,pre) in front.items():
            for b,(time_b,post) in back.items():
                if not self.allowed[a,b]:
                    continue
                release,due=origin+time_f,deadline-time_b
                slack=due-release-self.udp.min_tof
                maximum_slack=slack if maximum_slack is None else max(maximum_slack,slack)
                if slack>-.40:
                    opts.append((slack,a,b,release,due,pre,post))
        if maximum_slack is not None:
            stats["max_slack"]=maximum_slack
        stats["join_candidates"]=len(opts)
        for slack,a,b,release,due,pre,post in sorted(opts,key=lambda r:-r[0])[:12]:
            if time.monotonic()>=stop:
                break
            padding=0 if slack>0 else .35
            rows,_=fit_transfer_window(self.udp,a,b,max(origin,release-padding),due+padding,
                        self.cap,stop,random.Random(a*181+b+1691))
            stats["join_queries"]+=1
            for bridge in rows[:2]:
                candidate=self.build((*prefix,*pre,bridge,*post[::-1]))
                if bridge[2]>=release-1e-9 and arrival(bridge)<=due+1e-9:
                    self.submit(candidate,"shell_deadline_join",(width,deadline,step,a,b))
                if time.monotonic()>=stop:
                    break
                flags=[int(leg[-1]>self.normal) for leg in (*prefix,*pre,bridge,*post[::-1])]
                try:
                    solve(self.udp,candidate,l,1.9,0.,min(stop,time.monotonic()+4.),
                          lambda q:self.submit(q,"shell_deadline_join_polish",(width,deadline,step,a,b)),
                          self.count,cap_flags=flags)
                except Paused:
                    break

    def run(self):
        tested={tuple(row["key"]) for row in self.state["trials"]}
        for deadline in self.args.deadlines:
            for width,phase in ((6,.078125),(8,.0625),(10,.078125),(7,.046875)):
                if time.monotonic()>=self.stop:
                    break
                key=width,deadline,phase
                if key in tested:
                    continue
                start=time.monotonic()
                stats=Counter()
                try:
                    self.run_case(width,deadline,phase,min(self.stop,start+self.args.trial_seconds),stats)
                except BudgetExpired:
                    stats["time_cap"]+=1
                self.count.update({k:v for k,v in stats.items() if k!="max_slack"})
                self.state["trials"].append({"key":key,"seconds":time.monotonic()-start,"stats":dict(stats)})
                self.save()
                print(key,dict(stats),flush=True)
        self.state["wall_seconds"]=self.state.get("wall_seconds",0.)+time.monotonic()-self.started
        self.save()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",required=True,type=Path)
    p.add_argument("--seed-record",required=True,type=Path)
    p.add_argument("--baseline-record",type=Path)
    p.add_argument("--anchors",type=Path,nargs="*",default=())
    p.add_argument("--seconds",type=float,default=170.)
    p.add_argument("--trial-seconds",type=float,default=16.)
    p.add_argument("--beam",type=int,default=72)
    p.add_argument("--deadlines",type=float,nargs="+",default=(194.,192.,190.,185.))
    p.set_defaults(mode="boundary",goal=None,phase=None)
    Boundary(p.parse_args()).run()

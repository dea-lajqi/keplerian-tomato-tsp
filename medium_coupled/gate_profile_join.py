"""Connect saved high-shell prefixes to diverse reverse low-shell deadline profiles."""
import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from global_event_easy import fit_transfer_window
from medium_coupled.meet import SideGrid
from medium_coupled.nonlocal_routes import Surgery
from medium_coupled.run import solve, Paused
from medium_overnight.validation import verify
from medium_v4.driver import unpack
from pareto_easy import BudgetExpired, arrival


class ProfileJoin(Surgery):
    def profiles(self):
        best={}
        for file in self.args.profiles:
            if not file.exists():
                continue
            for value in json.loads(file.read_text()).values():
                legs=tuple(map(tuple,value["legs"]))
                rest=tuple(value["remaining"])
                if len(legs)!=160 or value["spend"]>4:
                    continue
                left=tuple(self.build(legs)["path"])
                if len(set((*left,*rest)))!=181 or any(self.axis(v)>8e6 for v in rest):
                    continue
                key=left[-1],frozenset(rest),value["spend"]
                if key not in best or arrival(legs[-1])<arrival(best[key][0][-1]):
                    best[key]=legs,rest,value["plan"]
        fallback=tuple(self.legs[:160])
        rest=tuple(self.path[161:])
        if sum(l[-1]>self.normal for l in fallback[:160])==4:
            key=fallback[-1][1],frozenset(rest),4
            if key not in best or arrival(fallback[-1])<arrival(best[key][0][-1]):
                best[key]=fallback,rest,"control"
        return sorted(best.values(),key=lambda row:arrival(row[0][-1]))

    def terminal_orders(self,rest):
        seen=set()
        for p in (self.args.seed_record,*self.args.anchors):
            s=unpack(json.loads(p.read_text()))
            t=tuple(s["path"][-len(rest):])
            if set(t)==set(rest) and t not in seen:
                seen.add(t)
                yield t
        for t in (rest,rest[::-1]):
            for i in range(5):
                lst=list(t)
                lst[0],lst[i]=lst[i],lst[0]
                p=tuple(lst)
                if p not in seen:
                    seen.add(p)
                    yield p

    def reverse_tail(self,order,floor,due,step,stop,stats):
        if floor>=due-self.udp.min_tof:
            return (),0
        grid=SideGrid(self.udp,floor,due,step,.012,stop,True)
        clock=0.
        tail=[]
        try:
            for a,b in reversed(tuple(itertools.pairwise(order))):
                options=list(self.anchors[a,b])
                for row in grid.edges(b,a,clock,1.5,2.5,False):
                    if row is None:
                        continue
                    d,f=due-arrival(row),row[3]
                    dv=float(self.udp.compute_transfer(a,b,d,f))
                    stats["replays"]+=1
                    options.append((a,b,d,f,dv))
                valid=[l for l in options if l[2]>=floor-1e-9 and l[-1]<=self.normal
                             and arrival(l)<=due-clock+1e-9]
                if not valid:
                    return tuple(tail[::-1]),len(tail)
                leg=max(valid,key=lambda x:x[2])
                tail.append(leg)
                clock=due-leg[2]
            return (*tail[::-1],),len(tail)
        finally:
            stats.update(grid.stats)

    def join(self,prefix,tail,day,step,stop,stats,key):
        a,b=prefix[-1][1],tail[0][0]
        r,d=arrival(prefix[-1]),tail[0][2]
        if not self.allowed[a,b]:
            return {}
        info={"slack_after_min_tof":d-r-self.udp.min_tof,"edge":[a,b],
              "forward":r,"backward":d}
        if info["slack_after_min_tof"]<-.70:
            return info
        opening=max(0.,r-(.35 if d-r<self.udp.min_tof else 0.))
        closing=d+(.35 if d-r<self.udp.min_tof else 0.)
        rows,cost=fit_transfer_window(self.udp,a,b,opening,closing,self.cap,
                       stop,random.Random(a+b*181+int(day*100)))
        stats["continuous_queries"]+=1
        if cost<float("inf"):
            info["sampled_dv"]=cost
        if not rows and getattr(self.args,"repair_open",False) and d>r-.35:
            rows,relaxed=fit_transfer_window(self.udp,a,b,max(0.,r-.32),d+.32,
                         self.cap,stop,random.Random(a*181+b+int(day*113)))
            stats["relaxed_gate_queries"]+=1
            if relaxed<float("inf"):
                info["relaxed_dv"]=relaxed
        for l in self.anchors[a,b]:
            if l[2]>=r-1e-9 and arrival(l)<=d+1e-9:
                rows=(*rows,l)
        for leg in rows[:4]:
            union=(*prefix,leg,*tail)
            candidate=self.build(union)
            if leg[2]>=r-1e-9 and arrival(leg)<=d+1e-9:
                self.submit(candidate,"reverse_profile_gate_join",key)
            if time.monotonic()<stop and self.state["best"]>day-.40:
                cap_flags=[int(l[-1]>self.normal) for l in union]
                try:
                    solve(self.udp,candidate,156,2.,0.,min(stop,time.monotonic()+3.),
                         lambda s:self.submit(s,"reverse_profile_gate_polish",key),self.count,
                         cap_flags=cap_flags)
                except Paused:
                    break
        return info

    def partial_join(self,prefix,rest,tail,day,stop,stats,key):
        missing=sorted(set(rest)-{tail[0][0],*(leg[1] for leg in tail)})
        if not 1<=len(missing)<=3:
            return {"missing":missing}
        r,d=arrival(prefix[-1]),tail[0][2]
        value={"missing":missing,"incoming":tail[0][0],"release":r,"due":d,
               "slack":d-r-(len(missing)+1)*self.udp.min_tof}
        if value["slack"]<-.65:
            return value
        rng=random.Random(int(day*100)+sum(missing)*17)
        for order in itertools.permutations(missing):
            if time.monotonic()>=stop:
                break
            path=(prefix[-1][1],*order,tail[0][0])
            if not self.all_allowed(path):
                continue
            chain=[]
            release=max(0.,r-.24)
            for i,(a,b) in enumerate(itertools.pairwise(path)):
                due=d+.24-(len(path)-2-i)*self.udp.min_tof
                rows,_=fit_transfer_window(self.udp,a,b,release,due,
                              self.cap if i==0 else self.normal,stop,rng)
                stats["partial_bridge_queries"]+=1
                if not rows:
                    break
                leg=min(rows,key=arrival)
                chain.append(leg)
                release=arrival(leg)
            if len(chain)!=len(path)-1:
                continue
            joined=(*prefix,*chain,*tail)
            candidate=self.build(joined)
            if self.submit(candidate,"partial_low_profile_join",key):
                stats["partial_bridge_valid"]+=1
            caps=[int(x[-1]>self.normal) for x in joined]
            try:
                solve(self.udp,candidate,156,2.2,0.,min(stop,time.monotonic()+3.),
                      lambda s:self.submit(s,"partial_low_profile_polish",key),
                      self.count,cap_flags=caps)
            except Paused:
                break
        return value

    def run(self):
        tested={tuple(t["key"]) for t in self.state["trials"]}
        prefixes=self.profiles()
        self.state["prefixes"]=[{"arrival":arrival(p[-1]),"last":p[-1][1],"plan":plan}
                                for p,_,plan in prefixes]
        for day in self.args.deadlines:
            for k,(prefix,rest,plan) in enumerate(prefixes[:self.args.prefix_limit]):
                if time.monotonic()>=self.stop:
                    break
                release=arrival(prefix[-1])
                if release+self.udp.min_tof*len(rest)>=day:
                    continue
                for j,order in enumerate(self.terminal_orders(rest)):
                    for step in (.046875,.02734375):
                        key=day,k,j,step
                        if time.monotonic()>=self.stop or key in tested:
                            continue
                        start=time.monotonic()
                        stop=min(self.stop,start+self.args.trial_seconds)
                        stats=Counter()
                        value={}
                        try:
                            if self.args.forward:
                                a,b=prefix[-1][1],order[0]
                                due=min(day-len(rest)*self.udp.min_tof,release+1.60)
                                if self.allowed[a,b]:
                                    rows,cost=fit_transfer_window(self.udp,a,b,release,due,self.cap,stop,
                                                      random.Random(j*113+int(day*100)))
                                    stats["forward_gate_queries"]+=1
                                    if cost<float("inf"):
                                        value["sampled_dv"]=cost
                                    for bridge in sorted(rows,key=arrival)[:1]:
                                        full=(*self.build(prefix)["path"],*order)
                                        self.complete(full,(*prefix,bridge),day,step,stop,key,value)
                            else:
                                tail,depth=self.reverse_tail(order,max(0.,release-.75),day,step,stop,stats)
                                value["backward_depth"]=depth
                                if depth==len(rest)-1:
                                    value["join"]=self.join(prefix,tail,day,step,stop,stats,key)
                                elif self.args.partial and tail:
                                    value["partial"]=self.partial_join(prefix,rest,tail,day,stop,stats,key)
                        except BudgetExpired:
                            value["time_cap"]=True
                        self.count.update(stats)
                        self.state["trials"].append({"key":key,"seconds":time.monotonic()-start,
                                                     "stats":dict(stats),**value})
                        self.save()
                        print(key,value,flush=True)
        self.state["wall_seconds"]=self.state.get("wall_seconds",0.)+time.monotonic()-self.started
        self.save()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--seed-record",type=Path,required=True)
    p.add_argument("--baseline-record",type=Path)
    p.add_argument("--anchors",type=Path,nargs="*",default=())
    p.add_argument("--profiles",type=Path,nargs="+",required=True)
    p.add_argument("--seconds",type=float,default=150.)
    p.add_argument("--trial-seconds",type=float,default=8.)
    p.add_argument("--deadlines",type=float,nargs="+",default=(194.,192.,190.,185.,193.5))
    p.add_argument("--partial",action="store_true")
    p.add_argument("--forward",action="store_true")
    p.add_argument("--prefix-limit",type=int,default=6)
    p.add_argument("--repair-open",action="store_true")
    p.set_defaults(mode="gate_profile_join",beam=None,goal=None,phase=None)
    ProfileJoin(p.parse_args()).run()

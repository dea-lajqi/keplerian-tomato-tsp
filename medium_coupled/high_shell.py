"""Probe long Medium flights, repair their neighborhoods and propagate the clock."""
import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import pairwise
import json
from pathlib import Path
import random
import sys
import time
from uuid import uuid4

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from global_event_easy import fit_transfer_window
from medium_coupled.checkpoint import atomic_json
from medium_coupled.endgame import decode
from medium_coupled.meet import SideGrid
from medium_coupled.run import ROOT, Paused, solve
from medium_overnight.validation import official_udp, solution_payload, verify
from medium_v4.driver import unpack
from pareto_easy import BudgetExpired, arrival

RECORD_IMPROVEMENT_DAYS = 0.0001


class Campaign:
    def __init__(self,args):
        self.args=args
        self.started=time.monotonic()
        self.stop=self.started+args.seconds
        self.udp=official_udp(ROOT)
        self.seed=unpack(json.loads(args.seed_record.read_text()))
        fitness=verify(self.udp,self.seed)
        benchmark=fitness[0]
        self.extra_anchors=[]
        if args.baseline_record:
            baseline=unpack(json.loads(args.baseline_record.read_text()))
            benchmark=min(benchmark,verify(self.udp,baseline)[0])
            for (a,b),d,f in zip(pairwise(baseline["path"]),baseline["dep"],baseline["tof"]):
                self.extra_anchors.append((a,b,d,f,float(self.udp.compute_transfer(a,b,d,f))))
        self.path,self.dep,self.tof=(self.seed[k] for k in ("path","dep","tof"))
        self.edges=list(pairwise(self.path))
        self.normal,self.cap=self.udp.dv_threshold,self.udp.dv_threshold_exception+1e-6
        self.legs=[(*edge,d,t,float(self.udp.compute_transfer(*edge,d,t)))
                   for edge,d,t in zip(self.edges,self.dep,self.tof)]
        self.exception=[int(l[-1]>self.normal) for l in self.legs]
        self.long=sorted(range(180),key=lambda i:-self.tof[i])[:6]
        self.file=args.directory/"checkpoint.json"
        self.state=json.loads(self.file.read_text()) if self.file.exists() else {
            "source":str(args.seed_record.resolve()),"baseline":fitness[0],
            "best":benchmark,"record":None,"trials":[],"counts":{},"orders":[],
            "long_flights":[[i,*self.legs[i]] for i in self.long]}
        if self.state["source"]!=str(args.seed_record.resolve()):
            raise ValueError("Checkpoint seed mismatch")
        self.count=Counter(self.state["counts"])
        self.orders=set(self.state["orders"])
        self.anchors=defaultdict(list)
        for leg in self.legs:
            self.anchors[leg[:2]].append(leg)
        for leg in self.extra_anchors:
            if leg not in self.anchors[leg[:2]]:
                self.anchors[leg[:2]].append(leg)
        self.best_source=self.seed
        if self.state["record"]:
            self.best_source=unpack(json.loads(Path(self.state["record"]).read_text()))
            verify(self.udp,self.best_source)
        self.shells=[i for i,(a,b) in enumerate(self.edges)
                     if min(self.axis(a),self.axis(b))*2<max(self.axis(a),self.axis(b))]

    def axis(self,i):
        return self.udp.tomato_orbital_parameters[i,0]

    def save(self):
        self.state["counts"]=dict(self.count)
        self.state["orders"]=sorted(self.orders)
        atomic_json(self.file,self.state)

    def submit(self,sol,method,plan):
        self.count["official_checks"]+=1
        try:
            f=verify(self.udp,sol)
        except ValueError:
            return False
        self.count["valid"]+=1
        self.orders.add(hashlib.sha256(bytes(sol["path"])).hexdigest())
        if f[0]<self.state["best"]-RECORD_IMPROVEMENT_DAYS:
            f=verify(official_udp(ROOT),sol)
            data=solution_payload(sol,f,udp=self.udp,provenance={"method":method,"plan":plan})
            record=self.args.directory/"records"/f"{f[0]:.12f}_{uuid4().hex[:8]}.json"
            atomic_json(record,{"solution":sol,**data})
            self.state.update(best=f[0],record=str(record.resolve()))
            self.best_source=sol
            self.save()
            print("RECORD",f,record,flush=True)
        return True

    def scan(self,i,deadline):
        release=arrival(self.legs[i-1])
        due=arrival(self.legs[i])
        a,b=self.edges[i]
        info={}
        for label,cap in (("normal",self.normal),("exception",self.cap)):
            rows,m=fit_transfer_window(self.udp,a,b,release,due,cap,deadline,
                                      random.Random(6809+i))
            latest=min(rows,key=arrival) if rows else None
            info[label]={"min_dv":m,"fastest_measured_arrival":arrival(latest) if latest else None,
                         "gain":due-arrival(latest) if latest else None,
                         "legs":rows}
        return info

    def sweep(self, path, prefix, flags, stop, step, reverse=False, goal=None,
              saved=None, mixed=()):
        """Earliest/latest labels with exact clock, per exception count and waiting."""
        goal=self.state["best"] if goal is None else goal
        release=arrival(prefix[-1])
        grid=SideGrid(self.udp,release,goal,step,.017,stop,reverse)
        grids=[grid]+[SideGrid(self.udp,release,goal,s,p,stop,reverse) for s,p in mixed]
        pairs=list(pairwise(path))
        route=pairs[::-1] if reverse else pairs[len(prefix):]
        starting=sum(l[-1]>self.normal for l in prefix)
        frontier={starting:(0.,prefix)} if not reverse else {0:(0.,())}
        stats=Counter()
        max_exc=self.udp.n_dv_exception
        try:
            for depth,(a,b) in enumerate(route):
                if reverse and depth>=len(pairs)-len(prefix):
                    break
                if time.monotonic()>=stop:
                    raise BudgetExpired
                next_labels={}
                for used,(clock,chain) in frontier.items():
                    if reverse and used+starting>max_exc:
                        continue
                    choices=list(self.anchors[a,b])
                    if flags is not None and not flags[len(pairs)-1-depth if reverse else len(prefix)+depth]:
                        allow=False
                    else:
                        allow=used+int(reverse)*starting<max_exc
                    sampled=(leg for lattice in grids for leg in
                             lattice.edges(b if reverse else a,a if reverse else b,
                                           clock,5.5,8.,allow))
                    for leg in sampled:
                        if not leg:
                            continue
                        d=goal-arrival(leg) if reverse else release+leg[2]
                        if d<0 or d+leg[3]>goal+1e-9:
                            continue
                        dv=float(self.udp.compute_transfer(a,b,d,leg[3]))
                        stats["exact_replays"]+=1
                        choices.append((a,b,d,leg[3],dv))
                    for leg in choices:
                        cur=goal-arrival(leg) if reverse else leg[2]-release
                        e=int(leg[-1]>self.normal)
                        count=used+e
                        if cur+1e-9<clock or count+int(reverse)*starting>max_exc or (e and not allow) or leg[-1]>self.cap:
                            continue
                        day=cur+leg[3]
                        if day>goal-release+1e-9:
                            continue
                        if count not in next_labels or day<next_labels[count][0]:
                            next_labels[count]=(day,(*chain,leg))
                if not next_labels:
                    stats["blocked_edge"]=[a,b]
                    return frontier,depth,{**grid.stats,**stats}
                frontier=next_labels
                if saved is not None and not reverse:
                    saved[len(prefix)+depth+1]=frontier
            return frontier,len(pairs)-len(prefix),{**grid.stats,**stats}
        finally:
            for lattice in grids:
                self.count.update(lattice.stats)
            self.count["sweep_replays"]+=stats["exact_replays"]

    def build(self,legs):
        return {"path":[legs[0][0],*(l[1] for l in legs)],
                "dep":[l[2] for l in legs],"tof":[l[3] for l in legs]}

    def window(self,i,width,step,deadline,free):
        left=max(1,i-width//2)
        right=min(179,left+width)
        found=[]
        reserve=sum(self.exception[right:])+free
        stats=Counter()

        def submit(part):
            full={"path":part["path"]+self.path[right+1:],
                  "dep":part["dep"]+self.dep[right:],"tof":part["tof"]+self.tof[right:]}
            if self.submit(full,"window",(i,width,step,free)):
                d=part["dep"][-1]+part["tof"][-1]
                legs=[(*edge,x,t,float(self.udp.compute_transfer(*edge,x,t)))
                      for edge,x,t in zip(pairwise(part["path"]),part["dep"],part["tof"])]
                found.append((d,legs,full))
        decode(self.udp,self.seed,left,self.dep[right],step,.013,deadline,stats,submit,
               reserve=reserve,endpoint=self.path[right],targets_override=self.path[left+1:right+1])
        self.count.update({k:v for k,v in stats.items() if k!="max_depth"})
        return right,sorted(found,key=lambda row:row[0]),dict(stats)

    def refine(self,full,left,offset,deadline,method,key,extra=None):
        return solve(self.udp,full,left,max(2.5,abs(offset)+1.5),offset,deadline,
                     lambda q:self.submit(q,method,key),self.count,extra)

    def propagate(self,full,prefix,key,deadline,step,goal):
        results={}
        path=full["path"]
        frontier,dep,stats=self.sweep(path,tuple(prefix),None,deadline,step,goal=goal)
        results["forward"]={"depth":dep,"stats":stats}
        expected=180-len(prefix)
        if dep==expected:
            for _,legs in frontier.values():
                self.submit(self.build(legs),"forward_clock",key)
        # Try a backward end-time label only for partial forward failures.
        if dep<expected and time.monotonic()<deadline:
            fronts,depth,rs=self.sweep(path,tuple(prefix),None,deadline,step,True,goal)
            results["backward"]={"depth":depth,"stats":rs}
            if depth==expected:
                for clock,tail in fronts.values():
                    if arrival(prefix[-1])<=goal-clock+1e-9:
                        self.submit(self.build((*prefix,*tail[::-1])),"backward_clock",key)
        return results

    def run(self):
        tested={tuple(x["key"]) for x in self.state["trials"]}
        plans=[("scan",i) for i in self.long]
        plans+= [("window",i,width,step,0) for width,step in ((4,.125),(6,.15625),(7,.09375))
                 for i in (*self.long,*self.shells)]
        plans+= [("window",i,width,step,1) for width,step in ((4,.09375),(6,.125))
                 for i in (169,*self.shells)]
        if self.args.reallocate:
            if sum(self.exception)>=self.udp.n_dv_exception:
                raise ValueError("Reallocation requires an ESA-valid source with a spare exception")
            plans=[("allocate",i,step,goal) for goal,step in
                   ((199.68,.109375),(198.,.078125),(195.,.09375),(199.68,.0625))
                   for i in self.long]
        for key in plans:
            if key in tested or time.monotonic()>=self.stop:
                continue
            start=time.monotonic()
            deadline=min(self.stop,start+self.args.trial_seconds)
            info={}
            try:
                if key[0]=="scan":
                    info=self.scan(key[1],deadline)
                elif key[0]=="allocate":
                    _,i,step,goal=key
                    info=self.allocate(i,step,goal,deadline,key)
                else:
                    _,i,width,step,free=key
                    right,found,stats=self.window(i,width,step,deadline,free)
                    info={"local_valid":len(found),"decode":stats,"propagations":[]}
                    seen=set()
                    for day,legs,full in found:
                        order=tuple(full["path"])
                        gained=arrival(self.legs[right-1])-day
                        if order in seen or gained<.035:
                            continue
                        seen.add(order)
                        info.setdefault("local_gain",[]).append(gained)
                        if len(seen)>2 or time.monotonic()>=deadline:
                            break
                        info["propagations"].append(self.propagate(full,legs,key,deadline,step,
                                                                    self.state["best"]-.005))
                    if found and time.monotonic()<deadline:
                        gain=arrival(self.legs[right-1])-found[0][0]
                        if gain>.06:
                            info["refined"]=self.refine(found[0][2],max(right,130),
                                                         -min(1.2,gain),deadline,"coupled",key)
            except (BudgetExpired,Paused):
                info["time_cap"]=True
            self.state["trials"].append({"key":key,"seconds":time.monotonic()-start,**info})
            self.save()
            print(key,{k:v for k,v in info.items() if k not in ("normal","exception","decode")},flush=True)
        self.state["wall_seconds"]=self.state.get("wall_seconds",0.)+time.monotonic()-self.started
        self.save()

    def allocate(self,i,step,goal,stop,key):
        """Spend the free maneuver before decoding, with mandatory downstream caps."""
        rows,_=fit_transfer_window(self.udp,*self.edges[i],arrival(self.legs[i-1]),
                    arrival(self.legs[i]),self.cap,stop,random.Random(9298+i))
        rows=sorted((l for l in rows if l[-1]>self.normal),key=arrival)
        if not rows:
            return {"no_exception_window":True}
        bridge=rows[0]
        prefix=(*self.legs[:i],bridge)
        flags=self.exception.copy()
        flags[i]=1
        info={"crossing_gain":arrival(self.legs[i])-arrival(bridge),"deadline":goal}
        saved={}
        forward,depth,stats=self.sweep(self.path,prefix,flags,stop,step,goal=goal,saved=saved)
        info["forward"]={"depth":depth,"stats":stats}
        info["milestones"]={str(i):[min(leg[-1][2]+leg[-1][3] for _,leg in saved[i].values()),
                                     arrival(self.legs[i-1])] for i in (145,159,161,170,175,178) if i in saved}
        if depth==180-len(prefix):
            for _,legs in forward.values():
                self.submit(self.build(legs),"forward_reserved_exception",key)
        elif self.args.endgame and time.monotonic()<stop:
            info["endgames"]=[]
            for cut,mesh in ((175,.0234375),(172,.03125),(169,.046875),(166,.0625)):
                if cut not in saved:
                    continue
                _,legs=min(saved[cut].values(),key=lambda x:x[0])
                draft=self.build((*legs,*self.legs[cut:]))
                s=Counter()
                t=time.monotonic()
                try:
                    decode(self.udp,draft,cut,goal,mesh,.006,min(stop,t+3.5),s,
                           lambda q:self.submit(q,"reserved_exception_endgame",key),
                           extra_anchors=self.extra_anchors)
                except BudgetExpired:
                    s["local_time_cap"]+=1
                self.count.update({k:v for k,v in s.items() if k!="max_depth"})
                info["endgames"].append({"cut":cut,**s})
                if time.monotonic()>=stop:
                    break
        elif time.monotonic()<stop:
            # Match a backward deadline profile against the furthest prefix
            # by checking the same next tomato and the remaining exceptions.
            _,advanced=min(forward.values(),key=lambda item:item[0])
            backward,reached,rs=self.sweep(self.path,advanced,flags,stop,step,True,goal)
            info["backward"]={"depth":reached,"stats":rs}
            if reached==180-len(advanced):
                for clock,tail in backward.values():
                    if arrival(advanced[-1])<=goal-clock+1e-9:
                        self.submit(self.build((*advanced,*tail[::-1])),"joined_reserved_exception",key)
        return info


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",required=True,type=Path)
    p.add_argument("--seed-record",required=True,type=Path)
    p.add_argument("--seconds",type=float,default=140.)
    p.add_argument("--trial-seconds",type=float,default=8.)
    p.add_argument("--reallocate",action="store_true")
    p.add_argument("--endgame",action="store_true")
    p.add_argument("--baseline-record",type=Path)
    Campaign(p.parse_args()).run()

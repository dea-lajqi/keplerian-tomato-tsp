"""Reusable normal-transfer event atlas and broad subset search for the final shell."""
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import itertools
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from global_event_easy import fit_transfer_window
from medium_coupled.checkpoint import atomic_json
from medium_coupled.gate_profile_join import ProfileJoin
from medium_coupled.meet import SideGrid
from medium_coupled.run import Paused, solve
from pareto_easy import BudgetExpired, arrival


def pareto(events):
    """An event is dominated only by a later departure and an earlier arrival."""
    first=float("inf")
    kept=[]
    for leg in sorted(set(map(tuple,events)),key=lambda l:(-l[2],arrival(l))):
        end=arrival(leg)
        if end<first-1e-11:
            kept.append(leg)
            first=end
    return tuple(reversed(kept))


class Atlas(ProfileJoin):
    def starts(self,profiles,group,stop):
        best={}
        for prefix,rest,_ in profiles:
            if set(rest)!=set(group):
                continue
            a=prefix[-1][1]
            release=arrival(prefix[-1])
            for b in group:
                if not self.allowed[a,b] or time.monotonic()>=stop:
                    continue
                samples,_=fit_transfer_window(self.udp,a,b,release,min(self.args.goal,release+1.5),
                                self.cap,stop,random.Random(6301+a*181+b))
                self.count["gateway_queries"]+=1
                options=[*samples,*self.anchors[a,b]]
                for l in options:
                    if l[2]>=release-1e-9 and l[-1]<=self.cap and arrival(l)<self.args.goal:
                        if b not in best or arrival(l)<arrival(best[b][-1]):
                            best[b]=(*prefix,l)
        seed=tuple(self.legs[:161])
        b=seed[-1][1]
        if b not in best or arrival(seed[-1])<arrival(best[b][-1]):
            best[b]=seed
        return best

    def cells(self,grid,a,b):
        """Enumerate one safe Pareto row per departure before running any beam."""
        grid.check()
        if not self.normal_edges[a,b]:
            return ()
        n=math.floor((grid.horizon-grid.phase)/grid.step)+1
        if n<=1:
            return ()
        maximum=math.floor(self.args.max_flight/grid.step)
        minimum=math.ceil(self.udp.min_tof/grid.step-1e-10)
        states_a=grid.states(a)[:n]
        states_b=grid.states(b)[:n]
        r1,v1=states_a.transpose(1,0,2)
        r2,v2=states_b.transpose(1,0,2)
        normals=np.cross(r1[:,None,:],r2[None,:,:])
        length=np.linalg.norm(normals,axis=2)
        cost=np.abs(np.einsum("dek,dk->de",normals,v1))+np.abs(np.einsum("dek,ek->de",normals,v2))
        can=(cost<=(self.normal+1e-5)*length)|(length<1e7)
        za,zb=grid.z_speed[a],grid.z_speed[b]
        directions=tuple(cw for cw,bound in ((False,max(0.,-za)+max(0.,-zb)),
                              (True,max(0.,za)+max(0.,zb))) if bound<=self.normal+1e-4)
        events=[]
        earliest=n
        last=n-minimum-1
        cutoff=getattr(self.args,"departure_before",None)
        if cutoff is not None:
            last=min(last,math.ceil((cutoff-grid.origin-grid.phase)/grid.step)-1)
        for d in range(last,-1,-1):
            grid.check()
            start,stop=d+minimum,min(earliest,d+maximum+1)
            if not directions or start>=stop:
                continue
            for e in np.flatnonzero(can[d,start:stop])+start:
                tof=float((e-d)*grid.step)
                cost=grid.lambert(r1[d],v1[d],r2[e],v2[e],tof,directions)
                if cost<=self.normal+1e-6:
                    dep=grid.origin+grid.epoch(d)
                    real=float(self.udp.compute_transfer(a,b,dep,tof))
                    self.count["atlas_exact_replays"]+=1
                    if real<=self.normal:
                        events.append((a,b,dep,tof,real))
                        earliest=e
                        break
        return events

    def build_atlas(self,group,floor,stop):
        path=self.args.directory/"events.json"
        bank=json.loads(path.read_text()) if path.exists() else {"rows":{},"done":[]}
        if self.args.seed_atlas and not bank["done"]:
            previous=json.loads(self.args.seed_atlas.read_text())
            bank["rows"]={k:[l for l in rows if l[2]>=floor and arrival(l)<=self.args.goal]
                           for k,rows in previous["rows"].items()}
        done=set(bank["done"])
        phases=getattr(self.args,"meshes",((.01875,.001),(.0146484375,.008)))
        def save():
            bank["done"]=sorted(done)
            atomic_json(path,bank)
        try:
            for j,(step,phase) in enumerate(phases):
                grid=SideGrid(self.udp,floor,self.args.goal,step,phase,stop,False)
                try:
                    for a,b in itertools.permutations(group,2):
                        grid.check()
                        key=f"{a},{b}"
                        task=f"{j}:{key}"
                        if task in done:
                            continue
                        rows=self.cells(grid,a,b)
                        archive=[l for l in self.anchors[a,b] if l[-1]<=self.normal
                                and l[2]>=floor and arrival(l)<=self.args.goal]
                        bank["rows"][key]=pareto([*bank["rows"].get(key,()),*rows,*archive])
                        done.add(task)
                        if len(done)%32==0:
                            save()
                finally:
                    self.count.update(grid.stats)
        except BudgetExpired:
            pass
        finally:
            save()
        return {(int(k.split(",")[0]),int(k.split(",")[1])):pareto(rows)
                for k,rows in bank["rows"].items()},len(done)

    def route(self,group,starts,events,width,mode,stop,tag):
        bit={a:1<<i for i,a in enumerate(group)}
        n=len(group)
        entries={}
        for (a,b),rows in events.items():
            if rows:
                entries[a,b]=tuple(l[2] for l in rows),rows
        minima={b:min((l[3] for a in group if (a,b) in entries
                       for l in entries[a,b][1]),default=2.) for b in group}
        initial={(bit[b],b):(arrival(p[-1]),(),b) for b,p in starts.items()}
        frontier=initial
        stats=Counter()
        template=tuple(self.path[-n:])
        for depth in range(1,n):
            if time.monotonic()>=stop:
                raise BudgetExpired
            update={}
            for (visited,a),(clock,chain,source) in frontier.items():
                for b in group:
                    if visited&bit[b] or (a,b) not in entries:
                        continue
                    ds,rows=entries[a,b]
                    i=bisect_left(ds,clock-1e-9)
                    if i==len(ds):
                        continue
                    leg=rows[i]
                    end=arrival(leg)
                    key=visited|bit[b],b
                    if key not in update or end<update[key][0]:
                        update[key]=(end,(*chain,leg),source)
            stats["labels"]+=len(update)
            stats["depth"]=depth
            if not update:
                break
            guide=sum(bit[a] for a in template[:depth+1])
            heuristic={}
            def rank(row):
                (mask,_),(clock,_,_)=row
                if mask not in heuristic:
                    heuristic[mask]=sum(minima[a] for a in group if not mask&bit[a])
                h=heuristic[mask]
                return clock+(h if mode!=2 else .25*h)+(.045*(mask^guide).bit_count() if mode==1 else 0.)
            # Leave room for labels with different visited sets through the beam.
            frontier={}
            occupancy=Counter()
            for key,row in sorted(update.items(),key=rank):
                if occupancy[key[0]]>=10:
                    continue
                occupancy[key[0]]+=1
                frontier[key]=row
                if len(frontier)>=width:
                    break
        completed=sorted((row for row in frontier.values() if len(row[1])==n-1),
                         key=lambda row:row[0])
        stats["earliest_before_polish"]=completed[0][0] if completed else None
        for _,tail,b in completed[:30]:
            proposal=self.build((*starts[b],*tail))
            if not self.submit(proposal,"terminal_event_atlas",tag):
                continue
            if (time.monotonic()<stop and stats["polishes"]<6 and
                    proposal["dep"][-1]+proposal["tof"][-1]<self.state["best"]+self.args.polish_slack):
                try:
                    fit=solve(self.udp,proposal,148,4.5,0.,min(stop,time.monotonic()+5.),
                         lambda q:self.submit(q,"terminal_atlas_polish",tag),stats)
                    stats["polishes"]+=1
                except Paused:
                    break
        stats["complete_labels"]=len(completed)
        return dict(stats)

    def run(self):
        profiles=self.profiles()
        group=tuple(self.path[-20:])
        starts=self.starts(profiles[:self.args.prefix_limit],group,min(self.stop,time.monotonic()+14.))
        self.state["crossing_arrivals"]={str(k):arrival(p[-1]) for k,p in starts.items()}
        floor=(self.args.floor_epoch if self.args.floor_epoch is not None
               else min(arrival(x[-1]) for x in starts.values())-.011)
        stop=min(self.stop-20.,time.monotonic()+self.args.atlas_seconds)
        events,n=self.build_atlas(group,floor,stop)
        self.state["atlas"]={"pair_meshes":n,"events":sum(map(len,events.values()))}
        self.save()
        if self.args.build_only:
            self.state["wall_seconds"]=self.state.get("wall_seconds",0.)+time.monotonic()-self.started
            self.save()
            return
        tested={tuple(t["key"]) for t in self.state["trials"]}
        for width,mode in itertools.product((700,2500,8000,18000),range(3)):
            key=width,mode
            if key in tested or time.monotonic()>=self.stop:
                continue
            start=time.monotonic()
            try:
                stats=self.route(group,starts,events,width,mode,self.stop,key)
            except BudgetExpired:
                stats={"time_cap":True}
            self.state["trials"].append({"key":key,"seconds":time.monotonic()-start,**stats})
            self.save()
            print(key,stats,self.state["best"],flush=True)
        self.state["wall_seconds"]=self.state.get("wall_seconds",0.)+time.monotonic()-self.started
        self.save()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--seed-record",type=Path,required=True)
    p.add_argument("--baseline-record",type=Path)
    p.add_argument("--anchors",type=Path,nargs="*",default=())
    p.add_argument("--profiles",type=Path,nargs="+",required=True)
    p.add_argument("--seconds",type=float,default=245.)
    p.add_argument("--atlas-seconds",type=float,default=120.)
    p.add_argument("--prefix-limit",type=int,default=2)
    p.add_argument("--goal",type=float,default=192.36)
    p.add_argument("--max-flight",type=float,default=1.8)
    p.add_argument("--seed-atlas",type=Path)
    p.add_argument("--polish-slack",type=float,default=.12)
    p.add_argument("--floor-epoch",type=float)
    p.add_argument("--build-only",action="store_true")
    p.add_argument("--departure-before",type=float)
    p.set_defaults(mode="terminal_atlas",phase=None,beam=None)
    Atlas(p.parse_args()).run()

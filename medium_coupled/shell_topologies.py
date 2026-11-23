"""Generate full-route alternatives by repartitioning the high shell between low groups."""
import argparse
from collections import Counter
from itertools import pairwise, permutations
import random
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medium_coupled.meet import SideGrid
from medium_coupled.nonlocal_routes import Surgery
from medium_overnight.validation import verify
from pareto_easy import BudgetExpired, arrival


class Topologies(Surgery):
    def low_options(self,group,head,tail):
        options=[]
        for original in (group,group[::-1]):
            for i in range(len(group)):
                route=original[i:]+original[:i]
                gates=([(head,route[0])] if head is not None else [])
                gates+=([(route[-1],tail)] if tail is not None else [])
                if all(self.allowed[a,b] for a,b in gates) and self.route_floor(route)==0:
                    rank=sum(self.geometry_floor[a,b] for a,b in gates)
                    options.append((rank,i,route))
        return sorted(options,key=lambda row:(row[0],row[1]))

    def candidates(self):
        a,b,c,d=self.shells
        low=(tuple(self.path[:a+1]),tuple(self.path[b+1:c+1]),tuple(self.path[d+1:]))
        high=sorted((*self.path[a+1:b+1],*self.path[c+1:d+1]),key=lambda x:self.inc[x])
        centers=[sum(self.inc[x] for x in group)/len(group) for group in low]
        seen=set()
        control=tuple(self.path[::-1])
        if self.all_allowed(control) and self.route_floor(control)<=5:
            seen.add(control)
            yield ("reverse",),control
        if self.args.reverse_only:
            return
        families={}
        for topology in permutations(range(3)):
            left,middle,right=topology
            eligible=[]
            for u,v in ((left,middle),(middle,right)):
                lo,hi=sorted((centers[u],centers[v]))
                eligible.append((lo-.10,hi+.10))
            rng=random.Random(self.args.random_seed+left*9+middle*3+right)
            ranking=[]
            for k in range(75):
                one=[];two=[]
                overlap=0
                offset=k%5-2
                for x in high:
                    inc=self.inc[x]
                    first=eligible[0][0]<=inc<=eligible[0][1]
                    later=eligible[1][0]<=inc<=eligible[1][1]
                    if first and later:
                        if abs(inc-centers[middle])<.24 and left!=1 and right!=1:
                            send=inc<centers[middle]+offset*.025
                        else:
                            send=bool((overlap+k)%2) if k<10 else rng.random()<.50
                            overlap+=1
                    elif first!=later:
                        send=first
                    else:
                        f=min(abs(inc-eligible[0][0]),abs(inc-eligible[0][1]))
                        t=min(abs(inc-eligible[1][0]),abs(inc-eligible[1][1]))
                        send=f<t
                    (one if send else two).append(x)
                if centers[left]>centers[middle]:
                    one.reverse()
                if centers[middle]>centers[right]:
                    two.reverse()
                if not one or not two:
                    continue
                if self.route_floor(one)+self.route_floor(two)>1:
                    continue
                if not self.all_allowed(one) or not self.all_allowed(two):
                    continue
                options=[self.low_options(low[left],None,one[0]),
                         self.low_options(low[middle],one[-1],two[0]),
                         self.low_options(low[right],two[-1],None)]
                if not all(options):
                    continue
                for i in range(min(3,*(len(x) for x in options))):
                    L,M,R=(group[i][2] for group in options)
                    full=(*L,*one,*M,*two,*R)
                    if len(set(full))!=181 or full in seen:
                        continue
                    seen.add(full)
                    ranking.append((sum(group[i][0] for group in options),k,i,full))
            families[topology]=sorted(ranking,key=lambda row:row[:3])[:self.args.per_topology]
            self.state.setdefault("geometric_candidates",{})[str(topology)]=len(ranking)
        for i in range(self.args.per_topology):
            for topology,ranking in families.items():
                if i<len(ranking):
                    _,k,orientation,full=ranking[i]
                    yield ("order",*topology,k,orientation),full

    def timed(self,path,key,step,stop):
        if time.monotonic()>=stop:
            raise BudgetExpired
        goal=min(self.udp.max_time,self.args.goal)
        grid=SideGrid(self.udp,0.,goal,.0390625,.011,stop,False)
        first=grid.edges(path[0],path[1],0.,3.0,3.0,False)
        candidates=[]
        for leg in first:
            if leg is None:
                continue
            d,f=leg[2:4]
            dv=float(self.udp.compute_transfer(path[0],path[1],d,f))
            if dv<=self.normal:
                candidates.append((*path[:2],d,f,dv))
        self.count.update(grid.stats)
        if not candidates:
            return {"furthest":0,"blocked_edge":path[:2]}
        prefix=(min(candidates,key=arrival),)
        saved={}
        try:
            f,depth,stats=self.sweep(path,prefix,None,stop,step,goal=goal,saved=saved,
                        mixed=((.0625,.008),) if self.args.mixed else ())
        finally:
            self.save_gate_prefixes(path,saved,key)
        ret={"furthest":depth+1,"blocked_edge":stats.get("blocked_edge")}
        if depth==179:
            for _,legs in f.values():
                s=self.build(legs)
                if self.submit(s,"whole_shell_topology",key):
                    fit=verify(self.udp,s)
                    ret["complete_finish"]=min(ret.get("complete_finish",float("inf")),fit[0])
        return ret

    def run(self):
        tested={tuple(t["key"]) for t in self.state["trials"]}
        for key,path in self.candidates():
            if time.monotonic()>=self.stop:
                break
            for step in (.09375,.0625):
                tag=(*key,step)
                if tag in tested:
                    continue
                start=time.monotonic()
                if start>=self.stop:
                    break
                try:
                    out=self.timed(path,tag,step,min(self.stop,start+self.args.trial_seconds))
                except BudgetExpired:
                    out={"time_cap":True}
                self.state["trials"].append({"key":tag,"seconds":time.monotonic()-start,**out})
                self.save()
                print(tag,out,flush=True)
        self.state["wall_seconds"]=self.state.get("wall_seconds",0.)+time.monotonic()-self.started
        self.save()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--seed-record",type=Path,required=True)
    p.add_argument("--baseline-record",type=Path)
    p.add_argument("--anchors",type=Path,nargs="*",default=())
    p.add_argument("--seconds",type=float,default=165.)
    p.add_argument("--trial-seconds",type=float,default=14.)
    p.add_argument("--goal",type=float,default=280.)
    p.add_argument("--per-topology",type=int,default=5)
    p.add_argument("--mixed",action="store_true")
    p.add_argument("--random-seed",type=int,default=873)
    p.add_argument("--reverse-only",action="store_true")
    p.add_argument("--save-prefixes",action="store_true")
    p.set_defaults(mode="whole_topology",beam=None,phase=None)
    Topologies(p.parse_args()).run()

"""Exhaust a finite terminal event graph and validate the reconstructed schedules."""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
import itertools
import json
from pathlib import Path
import numpy as np
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medium_coupled.run import Paused, solve
from medium_coupled.coupled_seam import CoupledSeam
from medium_coupled.terminal_atlas import Atlas, pareto
from pareto_easy import BudgetExpired, arrival


class ExactTerminal(Atlas):
    def execute(self,group,starts,rows,due,reverse,*,carry_boundary=False):
        count=len(group)
        full=(1<<count)-1
        clocks=np.full((count,full+1),np.inf)
        parent=np.full((count,full+1),-1,dtype=np.int8)
        edges={}
        for i,a in enumerate(group):
            if starts.get(a,np.inf)<=due:
                clocks[i,1<<i]=starts[a]
            for j,b in enumerate(group):
                if i==j:
                    continue
                arcs=rows.get((b,a) if reverse else (a,b),())
                if reverse:
                    edges[i,j]=(np.array([-arrival(l) for l in arcs[::-1]]),
                                np.array([-l[2] for l in arcs[::-1]]))
                else:
                    edges[i,j]=(np.array([l[2] for l in arcs]),
                                np.array([arrival(l) for l in arcs]))
        states=int(np.count_nonzero(np.isfinite(clocks).any(axis=1)))
        queries=0
        depth=int(states>0)
        for size in range(2,count+1):
            if time.monotonic()>=self.stop:
                raise TimeoutError
            layer=np.fromiter((sum(1<<node for node in nodes)
                               for nodes in itertools.combinations(range(count),size)),
                              dtype=np.int32)
            reachable=0
            for j in range(count):
                block=layer[(layer&(1<<j))!=0]
                previous=block^(1<<j)
                best=np.full(len(block),np.inf)
                source=np.full(len(block),-1,dtype=np.int8)
                for i in range(count):
                    if i==j:
                        continue
                    dep,arr=edges[i,j]
                    if not len(dep):
                        continue
                    clocks_at_node=clocks[i,previous]
                    possible=np.flatnonzero(clocks_at_node-1e-9<=dep[-1])
                    if not len(possible):
                        continue
                    idx=np.searchsorted(dep,clocks_at_node[possible]-1e-9)
                    queries+=len(idx)
                    finish=arr[idx]
                    wins=(finish<=due)&(finish<best[possible])
                    best[possible[wins]]=finish[wins]
                    source[possible[wins]]=i
                reachable+=np.count_nonzero(np.isfinite(best))
                clocks[j,block]=best
                parent[j,block]=source
            if not reachable:
                break
            depth=size
            states+=reachable
        def trace(j,mask):
            path=[j]
            node=j
            while mask&(mask-1):
                upstream=int(parent[node,mask])
                if upstream<0:
                    raise RuntimeError("Missing predecessor in exact event DP")
                mask^=1<<node
                node=upstream
                path.append(node)
            return tuple(group[v] for v in path[::-1])
        labels=[(float(t),trace(j,full)) for j,t in enumerate(clocks[:,full]) if np.isfinite(t)]
        metrics={"states":int(states),"queries":int(queries),"depth":depth}
        if not labels and depth==count-1:
            penultimate=sorted(
                (float(clocks[i,full^(1<<j)]),i,j)
                for j in range(count) for i in range(count) if i!=j
                and np.isfinite(clocks[i,full^(1<<j)]))
            metrics["one_missing"]=[(t,group[i],group[j]) for t,i,j in penultimate]
            if carry_boundary:
                metrics["carried_paths"]=[(t,group[j],trace(i,full^(1<<j)))
                                           for t,i,j in penultimate[:40]]
        return metrics,labels

    def reconstruct(self,route,starts,edges):
        chain=list(starts[route[0]])
        for a,b in zip(route,route[1:]):
            times,rows=edges[a,b]
            pos=bisect_left(times,arrival(chain[-1])-1e-9)
            if pos==len(rows):
                raise RuntimeError("Forward atlas label has no usable event")
            chain.append(rows[pos])
        return self.build(chain)

    def reconstruct_reverse(self,route,rows,due):
        tail=[]
        for b,a in zip(route,route[1:]):
            legs=rows[a,b]
            i=bisect_right(tuple(map(arrival,legs)),due+1e-9)-1
            if i<0:
                raise RuntimeError("Reverse atlas label has no usable event")
            leg=legs[i]
            due=leg[2]
            tail.append(leg)
        return tuple(reversed(tail))

    def run(self):
        group=tuple(self.path[-20:])
        profiles=self.profiles()[:self.args.prefix_limit]
        starts=self.starts(profiles,group,min(self.stop,time.monotonic()+35.))
        self.state["crossing_arrivals"]={str(b):arrival(p[-1]) for b,p in starts.items()}
        rows={tuple(map(int,k.split(","))):pareto(v)
              for k,v in json.loads(self.args.atlas.read_text())["rows"].items()}
        self.state["atlas"]={"edges":len(rows),"events":sum(map(len,rows.values())),
                             "starts":len(starts)}
        edges={key:(tuple(l[2] for l in values),values) for key,values in rows.items()}
        visited={tuple(t["key"]) for t in self.state["trials"]}
        for sense,days in (("forward",self.args.forward),("backward",self.args.backward)):
            reverse=sense=="backward"
            for day in days:
                key=sense,day
                if key in visited or time.monotonic()>=self.stop:
                    continue
                now=time.monotonic()
                stats=Counter()
                info={}
                try:
                    if reverse:
                        metrics,labels=self.execute(group,dict.fromkeys(group,-day),rows,
                                                 0.,True)
                        info["latest_low_starts"]={str(route[-1]):-time for time,route in labels}
                        if self.args.join_reverse:
                            info["connections"]=[]
                            for _,route in sorted(labels)[:8]:
                                tail=self.reconstruct_reverse(route,rows,day)
                                for prefix,rem,plan in profiles:
                                    if set(rem)!=set(route) or time.monotonic()>=self.stop:
                                        continue
                                    try:
                                        item=CoupledSeam.closure(self,prefix,tail,day,
                                               min(self.stop,time.monotonic()+12.),
                                               stats,(*key,route[-1],prefix[-1][1]))
                                        info["connections"].append(item)
                                    except BudgetExpired:
                                        stats["closure_time_cap"]+=1
                    else:
                        metrics,labels=self.execute(group,
                                   {b:arrival(p[-1]) for b,p in starts.items()},rows,day,False)
                        info["earliest_by_endpoint"]={str(route[-1]):t for t,route in labels}
                        for _,route in sorted(labels)[:6]:
                            candidate=self.reconstruct(route,starts,edges)
                            if self.submit(candidate,"exact_sampled_terminal",key):
                                try:
                                    solve(self.udp,candidate,145,4.4,0.,min(self.stop,time.monotonic()+6.),
                                         lambda q:self.submit(q,"exact_sampled_terminal_polish",key),stats)
                                except Paused:
                                    pass
                    info.update(metrics)
                except TimeoutError:
                    info["time_cap"]=True
                self.count.update(stats)
                self.state["trials"].append({"key":key,"seconds":time.monotonic()-now,**info})
                self.save()
                print(key,info,self.state["best"],flush=True)
        self.state["wall_seconds"]=self.state.get("wall_seconds",0.)+time.monotonic()-self.started
        self.save()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--seed-record",type=Path,required=True)
    p.add_argument("--baseline-record",type=Path)
    p.add_argument("--anchors",type=Path,nargs="*",default=())
    p.add_argument("--profiles",type=Path,nargs="+",required=True)
    p.add_argument("--atlas",type=Path,required=True)
    p.add_argument("--seconds",type=float,default=195.)
    p.add_argument("--forward",type=float,nargs="+",default=(192.,192.35,192.78))
    p.add_argument("--backward",type=float,nargs="+",default=(192.,192.2,191.85,190.))
    p.add_argument("--join-reverse",action="store_true")
    p.add_argument("--prefix-limit",type=int,default=2)
    p.set_defaults(mode="exact_terminal",phase=None,beam=None,goal=192.78)
    ExactTerminal(p.parse_args()).run()

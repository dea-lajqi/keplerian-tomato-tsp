"""Exact reordering around a deadline seam using nodal transfer-event enrichment."""
import argparse
from bisect import bisect_left
from collections import defaultdict
from itertools import pairwise
import json
from pathlib import Path
import time

from medium_coupled.nonlocal_routes import Surgery
from medium_coupled.run import Paused, solve
from medium_coupled.terminal_atlas import pareto
from medium_frontier.nodal import discover, orbit
from pareto_easy import arrival


class NodalJoin(Surgery):
    def events(self, bank, nodes, left, right, release, due, budget, stop):
        ranking = [(a,b) for a in (left,*nodes) for b in (*nodes,right)
                   if a!=b and self.allowed[a,b] and (budget or self.normal_edges[a,b])]
        ranking.sort(key=lambda edge:(edge in bank, self.geometry_floor[edge]))
        for a,b in ranking:
            if time.monotonic()>=stop:
                break
            if self.normal_edges[a,b]:
                found, _, calls = discover(self.udp,self.orbits,a,b,release,due,self.normal,
                                           min(stop,time.monotonic()+.7))
                self.count["normal_nodal"] += calls
                bank[a,b].update(found)
            if budget and not self.normal_edges[a,b]:
                found, _, calls = discover(self.udp,self.orbits,a,b,release,due,self.cap,
                                           min(stop,time.monotonic()+.7))
                self.count["exception_nodal"] += calls
                bank[a,b].update(found)

    def search(self, nodes, left, right, release, due, budget, bank):
        masks = {node:1<<i for i,node in enumerate(nodes)}
        indexes={}
        for key, options in bank.items():
            rows = [pareto(l for l in options if int(l[-1]>self.normal)==kind)
                    for kind in range(budget+1)]
            indexes[key] = [(tuple(l[2] for l in kind),kind) for kind in rows]
        labels = {(0,left,0):(release,())}
        counts = []
        for i in range(len(nodes)+1):
            next_labels={}
            for (mask,a,used),(ready,chain) in labels.items():
                for b in ((right,) if i==len(nodes) else nodes):
                    bit=masks.get(b,0)
                    if bit&mask:
                        continue
                    for kind,(deps,rows) in enumerate(indexes.get((a,b),())):
                        if kind+used>budget:
                            break
                        j=bisect_left(deps,ready-1e-9)
                        if j==len(rows):
                            continue
                        leg=rows[j]
                        finish=arrival(leg)
                        if finish>due+1e-9:
                            continue
                        key=mask|bit,b,used+kind
                        if key not in next_labels or finish<next_labels[key][0]:
                            next_labels[key]=finish,(*chain,leg)
            counts.append(len(next_labels))
            if not next_labels:
                return counts,[]
            labels=next_labels
        return counts,sorted(labels.values())

    def attempt(self, sched, seam, radius, stop, tag):
        edges = list(pairwise(sched["path"]))
        dep, tof = sched["dep"], sched["tof"]
        legs = [(*edge,d,t,float(self.udp.compute_transfer(*edge,d,t)))
                for edge,d,t in zip(edges,dep,tof)]
        left, right = max(1,seam-radius),min(179,seam+radius+1)
        a,b = sched["path"][left],sched["path"][right]
        nodes = tuple(sched["path"][left+1:right])
        release,due = arrival(legs[left-1]),dep[right]
        budget = self.udp.n_dv_exception-self.args.reserve-sum(
            l[-1]>self.normal for l in (*legs[:left],*legs[right:]))
        if budget < 0:
            return {"budget":budget,"exact_paths":0,"depths":[]}
        selected = set(nodes) | {a,b}
        relevant = list(legs[left:right])
        for bank in self.input_events:
            for x,y in bank:
                if x in selected and y in selected:
                    relevant.extend(bank[x,y])
        bank=defaultdict(set)
        for leg in relevant:
            if leg[2]>=release-1e-9 and arrival(leg)<=due+1e-9 and leg[-1]<=self.cap:
                bank[leg[:2]].add(leg)
        self.events(bank,nodes,a,b,release,due,budget,stop)
        counts,paths=self.search(nodes,a,b,release,due,budget,bank)
        for finish,chain in paths:
            sol=self.build((*legs[:left],*chain,*legs[right:]))
            if self.submit(sol,"nodal_exact_join",tag):
                try:
                    solve(self.udp,sol,left,2.5,0.,min(self.stop,time.monotonic()+5.),
                          lambda candidate:self.submit(candidate,"nodal_exact_join_polish",tag),self.count)
                except Paused:
                    pass
        return {"events":sum(map(len,bank.values())),"budget":budget,"depths":counts,
                "exact_paths":len(paths),"window":[release,due]}

    def run(self):
        self.orbits=[orbit(self.udp,n) for n in self.path]
        self.orbits=dict(zip(self.path,self.orbits))
        self.input_events=[{tuple(map(int,k.split(","))):list(map(tuple,rows))
                            for k,rows in json.loads(path.read_text())["rows"].items()}
                           for path in self.args.events]
        tested={tuple(trial["key"]) for trial in self.state["trials"]}
        sources=[(p,json.loads(p.read_text())) for p in self.args.near]
        for radius in (3,4,5):
            for p,data in sorted(sources,key=lambda row:row[1]["chronology_gap"])[:6]:
                if time.monotonic()>=self.stop:
                    break
                key=p.name,radius
                if key in tested:
                    continue
                start=time.monotonic()
                result=self.attempt(data["solution"],data["trial"][-2],radius,
                                    min(self.stop,start+self.args.trial_seconds),key)
                self.state["trials"].append({"key":key,"gap":data["chronology_gap"],
                                              "seconds":time.monotonic()-start,**result})
                self.save()
                print(key,result,flush=True)
        self.state["wall_seconds"]=time.monotonic()-self.started
        self.save()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed-record",type=Path,required=True)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--near",type=Path,nargs="+",required=True)
    p.add_argument("--events",type=Path,nargs="*",default=())
    p.add_argument("--seconds",type=float,default=125.)
    p.add_argument("--trial-seconds",type=float,default=9.)
    p.add_argument("--reserve",type=int,default=0,
                   help="Keep this many exceptions unused in the complete reconstruction.")
    p.set_defaults(mode="nodal_exact",beam=None,phase=None,goal=180.,anchors=(),baseline_record=None)
    NodalJoin(p.parse_args()).run()

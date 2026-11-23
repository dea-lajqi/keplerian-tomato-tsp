"""Propagate low-flight repartitioned seeds through new downstream timing windows."""
import argparse
from itertools import pairwise
import json
from pathlib import Path
import time

from medium_coupled.checkpoint import atomic_json
from medium_coupled.run import Paused, solve
from medium_repartition.caps import strict_schedule
from medium_repartition.search import PRIOR, Repartition
from pareto_easy import BudgetExpired, arrival


class Propagate(Repartition):
    def profile(self, chain, final, plan):
        sol=strict_schedule(self.udp,self.build(chain))
        if sol is None:
            return False
        rows=[]
        for (a,b),d,f in zip(pairwise(sol["path"]),sol["dep"],sol["tof"]):
            if d<0 or f<self.udp.min_tof-1e-9 or (rows and d<arrival(rows[-1])-1e-9):
                return False
            dv=float(self.udp.compute_transfer(a,b,d,f))
            rows.append((a,b,d,f,dv))
        if len(rows)!=160 or len(set(sol["path"]))!=161 or set(final)&set(sol["path"]):
            return False
        use=sum(l[-1]>self.normal for l in rows)
        if use>4:
            return False
        done=arrival(rows[-1])
        key=f'{rows[-1][1]}:{use}:{sol["path"][70]}:{sol["path"][91]}'
        current=self.profiles.get(key)
        if current is None or done<current["arrival"]-1e-7:
            self.profiles[key]={"legs":rows,"spend":use,"arrival":done,"remaining":final,"plan":plan}
            atomic_json(self.args.directory/"gate_pool.json",self.profiles)
            print("PROFILE",done,key,plan,flush=True)
        return True

    def run(self):
        out=self.args.directory/"gate_pool.json"
        self.profiles=json.loads(out.read_text()) if out.exists() else {}
        ranked=sorted(self.sources,key=lambda p:sum(l[3] for l in p[1]))
        tried={tuple(t["key"]) for t in self.state["trials"]}
        for step in (.039,.050,.061,.031):
            for index,(pieces,legs) in enumerate(ranked[:self.args.limit]):
                terminal=pieces[-1]
                route=sum(pieces[:4],())
                mid=len(pieces[0])+len(pieces[1])+len(pieces[2])
                reserve=[sum(not self.normal_edges[a,b] for a,b in pairwise(route[i:]))+1
                         for i in range(1,len(route))]
                for left in (mid-6,mid,mid+10,118,135,148):
                    key=(index,left,step)
                    if key in tried or time.monotonic()>=self.stop:
                        continue
                    start=time.monotonic()
                    prefix=tuple(legs[:left])
                    spend=sum(l[-1]>self.normal for l in prefix)
                    front={spend:(arrival(prefix[-1]),sum(l[3] for l in prefix),prefix)}
                    due=min(self.args.horizon,182.19)
                    try:
                        result,depth,block=self.pass_edges(route,left,len(route)-1,front,
                          False,reserve,step,due,min(self.stop,start+self.args.trial_seconds),1)
                        details={"depth":depth,"block":block,"ends":{}}
                        for use,(done,flight,full) in result.items():
                            if self.profile(full,terminal,key):
                                details["ends"][str(use)]=done
                                if done<179.4:
                                    try:
                                        solve(self.udp,self.build(full),left,3.5,0.,
                                              min(self.stop,time.monotonic()+4.),
                                              lambda sol:self.profile_solution(sol,terminal,key),
                                              self.count)
                                    except Paused:
                                        pass
                    except BudgetExpired:
                        details={"time_cap":True}
                    self.state["trials"].append({"key":key,"seconds":time.monotonic()-start,
                                                 **details})
                    self.save()
        for grid in self.grids.values():
            self.count.update(grid.stats)
        self.state["wall_seconds"]=time.monotonic()-self.started
        self.save()

    def profile_solution(self, sol, final, plan):
        rows=[(*edge,d,f,float(self.udp.compute_transfer(*edge,d,f))) for edge,d,f in
              zip(pairwise(sol["path"]),sol["dep"],sol["tof"])]
        return self.profile(rows,final,plan)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed-record",type=Path,required=True)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--archive-seeds",type=Path,nargs="*",default=())
    p.add_argument("--events",type=Path,nargs="*",default=())
    p.add_argument("--anchors",type=Path,nargs="*",default=())
    p.add_argument("--terminals",type=Path,default=PRIOR/"bidirectional_dense")
    p.add_argument("--terminal-extra",type=Path,nargs="*",default=())
    p.add_argument("--seconds",type=float,default=250.)
    p.add_argument("--trial-seconds",type=float,default=11.)
    p.add_argument("--horizon",type=float,default=183.5)
    p.add_argument("--limit",type=int,default=12)
    p.set_defaults(mode="propagate_partition",beam=None,goal=180.,phase=None,
                   random_seed=713741,baseline_record=None)
    Propagate(p.parse_args()).run()

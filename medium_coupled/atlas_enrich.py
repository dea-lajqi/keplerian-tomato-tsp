"""Refine sampled event fronts continuously and search the enlarged terminal graph."""
import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medium_coupled.checkpoint import atomic_json
from medium_coupled.terminal_atlas import Atlas, pareto
from pareto_easy import BudgetExpired, arrival


class Enrich(Atlas):
    def improve(self,a,b,event,stop,stats):
        d,f=event[2:4]
        t=d+f
        bins=[]
        floor=self.udp.min_tof
        def cost(x):
            if time.monotonic()>=stop:
                raise BudgetExpired
            stats["enrichment_queries"]+=1
            return float(self.udp.compute_transfer(a,b,float(x[0]),float(x[1])))
        common={"type":"ineq","fun":lambda x:self.normal-3e-7-cost(x)}
        for mode in ("earlier","later"):
            if mode=="earlier":
                obj=lambda x:float(x[0]+x[1])
                jac=lambda x:(1.,1.)
                bounds=((d,d+.085),(max(floor,f-.12),f+.06))
                constraints=[common]
            else:
                obj=lambda x:float(-x[0])
                jac=lambda x:(-1.,0.)
                # Slightly varying the arrival cap preserves useful
                # departures beyond the exact lattice event.
                close=t+.0025
                bounds=((max(0.,d-.03),d+.14),(max(floor,f-.16),f+.04))
                constraints=[common,{"type":"ineq","fun":lambda x:close-x[0]-x[1],
                                     "jac":lambda x:(-1.,-1.)}]
            fit=minimize(obj,np.clip([d,f],[a for a,_ in bounds],[b for _,b in bounds]),
                         jac=jac,bounds=bounds,constraints=constraints,method="SLSQP",
                         options={"maxiter":15,"ftol":4e-10})
            new_d,new_f=map(float,fit.x)
            measured=cost(fit.x)
            if (measured<=self.normal and new_d+new_f<=self.args.goal
                    and (mode=="earlier" or new_d+new_f<=close+1e-9)):
                bins.append((a,b,new_d,new_f,measured))
        return bins

    def run(self):
        source=json.loads(self.args.atlas.read_text())
        dest=self.args.directory/"events.json"
        output=json.loads(dest.read_text()) if dest.exists() else source
        completed=set(output.get("enriched",()))
        group=tuple(self.path[-20:])
        starts=self.starts(self.profiles()[:2],group,min(self.stop,time.monotonic()+18.))
        self.state["crossing_options"]=len(starts)
        stats=Counter()
        stop=min(self.stop-30.,time.monotonic()+self.args.enrich_seconds)
        def save_bank():
            output["enriched"]=sorted(completed)
            atomic_json(dest,output)
        try:
            original=list(tuple(map(int,k.split(","))) for k in source["rows"])
            active={l[:2] for l in self.legs[-19:]}
            original.sort(key=lambda e:(e not in active, e))
            for a,b in original:
                if time.monotonic()>=stop:
                    break
                key=f"{a},{b}"
                if key in completed:
                    continue
                all_rows=pareto([l for l in itertools.chain(output["rows"][key],self.anchors[a,b])
                                 if arrival(l)<=self.args.goal and l[-1]<=self.normal])
                buckets={}
                for leg in all_rows:
                    k=int(leg[2]/self.args.epoch_bin)
                    if k not in buckets or leg[3]<buckets[k][3]:
                        buckets[k]=leg
                added=[]
                for event in buckets.values():
                    added.extend(self.improve(a,b,event,stop,stats))
                events=pareto([*all_rows,*added])
                stats["new_events"]+=sum(x not in all_rows for x in events)
                output["rows"][key]=events
                completed.add(key)
                if len(completed)%30==0:
                    save_bank()
        except BudgetExpired:
            stats["enrich_time_cap"]+=1
        finally:
            save_bank()
        self.count.update(stats)
        events={tuple(map(int,k.split(","))):pareto(v) for k,v in output["rows"].items()}
        self.state["atlas"]={"pairs_enriched":len(completed),"events":sum(map(len,events.values())),
                              "pair_meshes":len(output["done"])}
        self.save()
        tested={tuple(t["key"]) for t in self.state["trials"]}
        for width,mode in itertools.product((2500,8500,17000),range(3)):
            key=(width,mode)
            if key in tested or time.monotonic()>=self.stop:
                continue
            start=time.monotonic()
            try:
                info=self.route(group,starts,events,width,mode,self.stop,key)
            except BudgetExpired:
                info={"time_cap":True}
            self.state["trials"].append({"key":key,"seconds":time.monotonic()-start,**info})
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
    p.add_argument("--seconds",type=float,default=215.)
    p.add_argument("--enrich-seconds",type=float,default=135.)
    p.add_argument("--epoch-bin",type=float,default=.42)
    p.add_argument("--polish-slack",type=float,default=.38)
    p.add_argument("--goal",type=float,default=192.78)
    p.set_defaults(mode="atlas_enrich",phase=None,beam=None)
    Enrich(p.parse_args()).run()

"""Bridge optimized first-high prefixes through flexible polar-shell orders."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from global_event_easy import fit_transfer_window
from medium_coupled.exact_terminal import ExactTerminal, pareto
from pareto_easy import BudgetExpired, arrival


class Polar(ExactTerminal):
    def run(self):
        entry,exit=self.shells[1:3]
        group=tuple(self.path[entry+1:exit+1])
        old=arrival(self.legs[exit-1])
        bank=json.loads(self.args.profiles.read_text())
        profiles=[tuple(map(tuple,x["legs"])) for x in bank.values()
                  if len(x["legs"])==entry and x["spend"]==1]
        if not profiles:
            raise RuntimeError("Expected an exact one-exception prefix")
        begin=time.monotonic()
        last=min(self.stop-30.,begin+self.args.atlas_seconds)
        floor=min(arrival(x[-1]) for x in profiles)
        if self.args.atlas_input:
            snapshot=json.loads(self.args.atlas_input.read_text())
            events={tuple(map(int,k.split(","))):pareto(v) for k,v in snapshot["rows"].items()}
            pairs=len(snapshot["done"])
        else:
            events,pairs=self.build_atlas(group,floor,last)
        self.state["atlas"]={"pairs":pairs,"events":sum(map(len,events.values())),
                            "incoming_clock":floor,"old_outgoing_clock":old}
        starts={}
        stop=min(self.stop,time.monotonic()+16.)
        for pre in sorted(profiles,key=lambda x:arrival(x[-1])):
            a=pre[-1][1]
            release=arrival(pre[-1])
            for b in group:
                if not self.allowed[a,b] or time.monotonic()>=stop:
                    continue
                try:
                    fit,_=fit_transfer_window(self.udp,a,b,release,release+1.8,self.cap,stop,
                                  random.Random(2891+a*181+b))
                except BudgetExpired:
                    break
                self.count["polar_crossing_queries"]+=1
                measured=[l for l in (*fit,*self.anchors[a,b])
                          if l[2]>=release-1e-9 and l[-1]<=self.cap]
                if measured:
                    leg=min(measured,key=arrival)
                    if b not in starts or arrival(leg)<arrival(starts[b][-1]):
                        starts[b]=(*pre,leg)
        self.state["incoming_options"]={b:arrival(p[-1]) for b,p in starts.items()}
        self.save()
        edges={key:(tuple(l[2] for l in v),v) for key,v in events.items()}
        done={tuple(t["key"]) for t in self.state["trials"]}
        optional={l[:2] for l in self.legs if l[-1]>self.normal and
             min(self.axis(l[0]),self.axis(l[1]))*2>=max(self.axis(l[0]),self.axis(l[1]))}
        for day in (old-1.25,old-.6,old+.18,self.args.goal):
            key=("polar",day)
            if key in done or time.monotonic()>=self.stop:
                continue
            t0=time.monotonic()
            stats={}
            try:
                stats,labels=self.execute(group,{b:arrival(p[-1]) for b,p in starts.items()},
                                          events,day,False)
                stats["outbound_clocks"]={str(order[-1]):t for t,order in labels}
                stats["through"]=[]
                for clock,order in sorted(labels)[:8]:
                    if time.monotonic()>=self.stop:
                        break
                    part=self.reconstruct(order,starts,edges)
                    legs=[(a,b,d,f,float(self.udp.compute_transfer(a,b,d,f)))
                       for a,b,d,f in zip(part["path"],part["path"][1:],part["dep"],part["tof"])]
                    for move in ((1,2,4,6) if self.args.change_exit else (0,)):
                        remainder=self.path[exit+1:]
                        remainder[0],remainder[move]=remainder[move],remainder[0]
                        path=tuple(part["path"]+remainder)
                        if not self.all_allowed(path) or self.route_floor(path)>5:
                            continue
                        detail={"polar_finish":clock,"polar_exit":order[-1],
                                "next_high":remainder[0]}
                        try:
                            self.complete(path,tuple(legs),self.args.mission_goal,.0634765625,
                                      min(self.stop,time.monotonic()+8.),(*key,move),detail,
                                      permitted_extra=optional)
                        except BudgetExpired:
                            detail["time_cap"]=True
                        stats["through"].append(detail)
                    self.save()
            except TimeoutError:
                stats["time_cap"]=True
            self.state["trials"].append({"key":key,"seconds":time.monotonic()-t0,**stats})
            self.save()
            print(key,stats,self.state["best"],flush=True)
        self.state["wall_seconds"]=time.monotonic()-self.started
        self.save()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed-record",type=Path,required=True)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--profiles",type=Path,required=True)
    p.add_argument("--baseline-record",type=Path)
    p.add_argument("--anchors",type=Path,nargs="*",default=())
    p.add_argument("--seconds",type=float,default=205)
    p.add_argument("--atlas-seconds",type=float,default=135)
    p.add_argument("--goal",type=float,default=97.3)
    p.add_argument("--mission-goal",type=float,default=183.03)
    p.add_argument("--atlas-input",type=Path)
    p.add_argument("--change-exit",action="store_true")
    p.set_defaults(mode="polar",phase=None,beam=None,seed_atlas=None,max_flight=2.8,
                   meshes=((.025390625,.007),(.03125,.016)),
                   reserve_gate=True,save_prefixes=True,polish=True)
    Polar(p.parse_args()).run()

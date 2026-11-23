"""Continuously optimize verified high-shell prefixes for earlier crossing compatibility."""
import argparse
from collections import Counter
from itertools import pairwise
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medium_coupled.checkpoint import atomic_json
from medium_coupled.meet import search
from medium_coupled.nonlocal_routes import Surgery
from medium_coupled.run import solve, Paused
from pareto_easy import BudgetExpired, arrival


class PrefixPolish(Surgery):
    def measure(self,sol,stats):
        actual=[]
        last=0.
        spent=0
        for (a,b),d,t in zip(pairwise(sol["path"]),sol["dep"],sol["tof"]):
            if d<last-1e-9 or t<self.udp.min_tof-1e-10:
                return (),0
            dv=float(self.udp.compute_transfer(a,b,d,t))
            stats["prefix_checks"]+=1
            spent+=int(dv>self.normal)
            if spent>4 or dv>self.cap:
                return (),0
            actual.append((a,b,d,t,dv))
            last=d+t
        return actual,spent

    def run(self):
        file=self.args.directory/"gate_pool.json"
        output=json.loads(file.read_text()) if file.exists() else {}
        tested={tuple(t["key"]) for t in self.state["trials"]}
        seeds={}
        for source in self.args.profiles:
            if not source.exists():
                continue
            for name,row in json.loads(source.read_text()).items():
                if len(row["legs"])==160 and row["spend"]<=4:
                    if name not in seeds or row["arrival"]<seeds[name]["arrival"]:
                        seeds[name]=row
        chosen=sorted(seeds.items(),key=lambda x:x[1]["arrival"])[:10]
        if self.args.meet:
            self.rebuild(chosen,output,file,tested)
            self.state["wall_seconds"]=self.state.get("wall_seconds",0.)+time.monotonic()-self.started
            self.save()
            return
        modes=[(147,0.,None),(137,0.,None),(125,0.,None),(116,0.,None),
               (135,-.38,None),(122,-.85,None),(135,0.,138),(115,0.,125)]
        for left,shift,compression in modes:
            for name,row in chosen:
                if time.monotonic()>=self.stop:
                    break
                tag=(name,left,shift,compression)
                if tag in tested:
                    continue
                original=output.get(name,row)
                old=original["arrival"]
                legs=tuple(map(tuple,original["legs"]))
                path=self.build(legs)
                flags=[l[-1]>self.normal for l in legs]
                start=time.monotonic()
                local=min(self.stop,start+self.args.trial_seconds)
                stats=Counter()

                def accept(sol):
                    actual,spent=self.measure(sol,stats)
                    if not actual:
                        return False
                    last=arrival(actual[-1])
                    stats["valid_prefixes"]+=1
                    if last<output.get(name,row)["arrival"]-1e-8:
                        output[name]={"legs":actual,"remaining":row["remaining"],"arrival":last,
                                      "spend":spent,"plan":tag}
                        atomic_json(file,output)
                        print("EARLIER HIGH PREFIX",old,last,tag,flush=True)
                    return True

                init=None
                if compression is not None and left<=compression<len(legs):
                    init=[(l[2],l[3]) for l in legs]
                    period=float(2*3.141592653589793*(self.axis(legs[compression][0])**3/4.9048695e12)**.5/86400)
                    reduce=min(period,max(0.,legs[compression][3]-.03))
                    if reduce>.03:
                        t,f=init[compression]
                        init[compression]=(t,f-reduce)
                        for j in range(compression+1,len(init)):
                            t,f=init[j]
                            init[j]=(t-reduce,f)
                details={}
                try:
                    details=solve(self.udp,path,left,3.8,shift,local,accept,stats,
                                  cap_flags=flags,start_times=init)
                except Paused:
                    details["time_cap"]=True
                self.count.update(stats)
                self.state["trials"].append({"key":tag,"seconds":time.monotonic()-start,
                      "before":old,"after":output.get(name,row)["arrival"],"details":details})
                self.save()
        self.state["wall_seconds"]=self.state.get("wall_seconds",0.)+time.monotonic()-self.started
        self.save()

    def rebuild(self,chosen,output,file,tested):
        for goal in (188.8,188.45,189.05):
            for name,row in chosen[:2]:
                for width,step,shift in ((12,.078125,2),(16,.109375,3),(18,.109375,3)):
                    tag=name,goal,width,step,shift
                    if tag in tested or time.monotonic()>=self.stop:
                        continue
                    begin=time.monotonic()
                    stop=min(self.stop,begin+self.args.trial_seconds)
                    stats=Counter()
                    def accept(sol):
                        actual,spend=self.measure(sol,stats)
                        if not actual:
                            return False
                        gate=actual[-1][1]
                        mark=f"{name}-{gate}"
                        clock=arrival(actual[-1])
                        if mark not in output or clock<output[mark]["arrival"]-1e-8:
                            output[mark]={"legs":actual,"remaining":row["remaining"],
                                          "arrival":clock,"spend":spend,"plan":tag}
                            atomic_json(file,output)
                            print("REBUILT HIGH PREFIX",clock,gate,tag,flush=True)
                        stats["valid_prefixes"]+=1
                        return True
                    try:
                        search(self.udp,self.build(row["legs"]),width,goal,step,shift,75,stop,
                               stats,accept,repair_near=True,reserve=1)
                    except BudgetExpired:
                        stats["time_cap"]+=1
                    self.count.update({k:v for k,v in stats.items() if k!="best_slack_after_min_tof"})
                    self.state["trials"].append({"key":tag,"seconds":time.monotonic()-begin,
                                                  "stats":dict(stats)})
                    self.save()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--seed-record",type=Path,required=True)
    p.add_argument("--baseline-record",type=Path)
    p.add_argument("--anchors",type=Path,nargs="*",default=())
    p.add_argument("--profiles",type=Path,nargs="+",required=True)
    p.add_argument("--seconds",type=float,default=145.)
    p.add_argument("--trial-seconds",type=float,default=6.)
    p.add_argument("--meet",action="store_true")
    p.set_defaults(mode="gate_prefix_polish",beam=None,goal=None,phase=None)
    PrefixPolish(p.parse_args()).run()

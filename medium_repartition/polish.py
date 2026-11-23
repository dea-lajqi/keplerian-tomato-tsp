"""Hard-cap trajectory descent from validated repartitioned starting schedules."""
import argparse
from itertools import pairwise
import json
from pathlib import Path
import time

from medium_coupled.checkpoint import atomic_json
from medium_coupled.high_shell import Campaign
from medium_coupled.run import Paused, solve
from medium_repartition.caps import strict_schedule
from medium_overnight.validation import to_solution, verify
from pareto_easy import arrival


class Polish(Campaign):
    def run(self):
        profiles={}
        out=self.args.directory/"gate_pool.json"
        if out.exists():
            profiles=json.loads(out.read_text())
        gate=self.shells[-1]
        initial=[self.seed]
        for file in self.args.archive_seeds:
            sol=to_solution(json.loads(file.read_text()))
            verify(self.udp,sol)
            initial.append(sol)
        tested={tuple(t["key"]) for t in self.state["trials"]}

        def save_prefix(trial, plan):
            q=strict_schedule(self.udp,trial)
            if q is None:
                return False
            legs=[]
            for (a,b),d,f in zip(pairwise(q["path"]),q["dep"],q["tof"]):
                if f < self.udp.min_tof-1e-10 or (legs and d<arrival(legs[-1])-1e-9):
                    return False
                dv=float(self.udp.compute_transfer(a,b,d,f))
                legs.append((a,b,d,f,dv))
            used=sum(l[-1]>self.normal for l in legs)
            if used>4:
                return False
            done=arrival(legs[-1])
            key=f'{legs[-1][1]}:{used}:{q["path"][70]}:{q["path"][91]}'
            if key not in profiles or done<profiles[key]["arrival"]-1e-8:
                profiles[key]={"arrival":done,"legs":legs,"spend":used,
                               "remaining":self.path[gate+1:],"plan":plan}
                atomic_json(out,profiles)
                print("GATE",done,key,plan,flush=True)
            return True

        modes=((151,2.8,0.),(126,4.2,0.),(108,4.8,0.),(0,2.8,0.),
               (120,4.7,-.25),(117,5.1,-.55),(103,5.8,-.85),(85,3.8,-.20),
               (135,3.8,-.20),(151,3.2,-.13),(59,5.4,-.35),(0,5.8,-.55))
        for partial in (False,True):
            for left,margin,shift in modes:
                if time.monotonic()>=self.stop:
                    break
                for index,sol in enumerate(initial):
                    if time.monotonic()>=self.stop:
                        break
                    source=self.best_source if index==0 and not partial else sol
                    tag=(partial,left,margin,shift,index)
                    if tag in tested:
                        continue
                    flags=[float(self.udp.compute_transfer(*pair,d,f))>self.normal
                           for pair,d,f in zip(pairwise(source["path"]),source["dep"],source["tof"])]
                    start=time.monotonic()
                    try:
                        if partial:
                            q={n:items[:gate+int(n=="path")] for n,items in source.items()}
                            res=solve(self.udp,q,left,margin,shift,
                                    min(self.stop,start+self.args.trial_seconds),
                                    lambda p:save_prefix(p,tag),self.count,cap_flags=flags[:gate])
                        else:
                            def accept(candidate):
                                p=strict_schedule(self.udp,candidate)
                                return p is not None and self.submit(p,"hard_cap_repartition_polish",tag)
                            res=solve(self.udp,source,left,margin,shift,
                                    min(self.stop,start+self.args.trial_seconds),
                                    accept,self.count,cap_flags=flags)
                    except Paused:
                        res={"time_cap":True}
                    self.state["trials"].append({"key":tag,"seconds":time.monotonic()-start,**res})
                    self.save()
        self.state["wall_seconds"]=time.monotonic()-self.started
        self.save()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed-record",type=Path,required=True)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--archive-seeds",type=Path,nargs="*",default=())
    p.add_argument("--seconds",type=float,default=310.)
    p.add_argument("--trial-seconds",type=float,default=12.)
    p.set_defaults(baseline_record=None)
    Polish(p.parse_args()).run()

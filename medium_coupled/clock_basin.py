"""Continuous basin moves on complete Medium trajectories."""
import argparse
from collections import Counter
from pathlib import Path
import random
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medium_coupled.clock_projection import restore
from medium_coupled.high_shell import Campaign
from medium_coupled.run import Paused, solve
from pareto_easy import arrival


class ClockBasin(Campaign):
    def plans(self):
        for left in (150,139,125,112,54,0):
            yield ("wide",left,0,0.)
        for i in sorted(self.long):
            for change in (.42,.9,1.91):
                yield ("long",max(0,i-2),i,change)
        for left in (153,146,135,118):
            for shift in (.18,.38,.68):
                yield ("gate",left,160,shift)
        rng=random.Random(98177)
        for index in range(24):
            yield ("basin",rng.choice((133,142,149,159,168)),index,rng.uniform(.12,.55))

    def run(self):
        done={tuple(t["key"]) for t in self.state["trials"]}
        for key in self.plans():
            if key in done or time.monotonic()>=self.stop:
                continue
            kind,left,pivot,amount=key
            source=self.best_source
            d,f,p=(source[k] for k in ("dep","tof","path"))
            costs=[float(self.udp.compute_transfer(a,b,x,y))
                   for a,b,x,y in zip(p,p[1:],d,f)]
            self.count["seed_evaluations"]+=len(costs)
            flags=[c>self.normal for c in costs]
            times=list(zip(d,f))
            if kind=="long":
                clip=min(amount,f[pivot]-self.udp.min_tof)
                times[pivot]=(d[pivot],f[pivot]-clip)
                for i in range(pivot+1,180):
                    times[i]=(d[i]-clip*(.35+.65*max(0.,1.-(i-pivot-1)/22)),f[i])
            elif kind=="gate":
                for i in range(left,180):
                    factor=min(1.,(i-left+1)/(160-left+1))
                    times[i]=(d[i]-amount*factor,f[i])
            elif kind=="basin":
                rng=random.Random(99017+pivot)
                for i in range(left,180):
                    a=(i-left+1)/(180-left)
                    times[i]=(d[i]-a*amount+rng.uniform(-.07,.07),f[i]+rng.uniform(-.04,.04))
            start=time.monotonic()
            stop=min(self.stop,start+self.args.trial_seconds)
            stats=Counter()
            details={}
            accept=lambda sol:self.submit(sol,"wide_clock_"+kind,key)
            try:
                details["direct"]=solve(self.udp,source,left,4.5,0.,min(stop,start+6.5),
                             accept,stats,cap_flags=flags,start_times=times)
                if kind!="wide" and time.monotonic()+1.<stop:
                    # Ask the sparse feasibility projection for a different
                    # candidate before polishing against the actual objective.
                    candidate={"path":p,
                               "dep":[max(0.,x[0]) for x in times],
                               "tof":[max(self.udp.min_tof,x[1]) for x in times]}
                    corrected,res=restore(self.udp,candidate,left,
                        min(192.,self.state["best"]-.01),min(stop,time.monotonic()+2.8),
                        stats,shift=0.,cap_flags=flags)
                    details["projection"]=res
                    if res["largest_scaled_violation"]<.22 and time.monotonic()<stop:
                        details["recovered"]=solve(self.udp,source,left,4.5,0.,stop,
                           accept,stats,cap_flags=flags,start_times=corrected)
            except Paused:
                details["time_cap"]=True
            self.count.update(stats)
            self.state["trials"].append({"key":key,"seconds":time.monotonic()-start,**details})
            self.save()
            print(key,self.state["best"],details,flush=True)
        self.state["wall_seconds"]=self.state.get("wall_seconds",0.)+time.monotonic()-self.started
        self.save()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--seed-record",type=Path,required=True)
    p.add_argument("--baseline-record",type=Path)
    p.add_argument("--seconds",type=float,default=225.)
    p.add_argument("--trial-seconds",type=float,default=10.)
    ClockBasin(p.parse_args()).run()

"""Warm-start a complete low-shell sequence one or more orbital periods earlier."""
import argparse
import math
from pathlib import Path
import random
import statistics
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from global_event_easy import fit_transfer_window
from medium_coupled.high_shell import Campaign
from medium_coupled.run import solve, Paused
from pareto_easy import BudgetExpired, arrival


class PhaseShift(Campaign):
    def run(self):
        gate=self.shells[-1]
        self.state["gate"]=gate
        period=statistics.median(
            float(2*3.141592653589793*(self.axis(node)**3/4.9048695e12)**.5/86400)
            for node in self.path[gate+1:])
        self.state["low_shell_period_days"]=period
        tested={tuple(row["key"]) for row in self.state["trials"]}
        for cycles in (1.,2.,.5,1.5,3.):
            for swap in (0,1,2,3,4):
                key=cycles,swap
                if key in tested or time.monotonic()>=self.stop:
                    continue
                start=time.monotonic()
                stop=min(self.stop,start+self.args.trial_seconds)
                shift=period*cycles
                shifted=list(zip(self.dep,self.tof))
                for i in range(gate+1,180):
                    shifted[i]=(self.dep[i]-shift,self.tof[i])
                targets=self.path[:]
                target=gate+1+swap
                targets[gate+1],targets[target]=targets[target],targets[gate+1]
                proposal={"path":targets,"dep":self.dep,"tof":self.tof}
                release=arrival(self.legs[gate-1])
                due=shifted[gate+1][0]+.06
                info={"proposed_low_departure":shifted[gate+1][0]}
                try:
                    branches,cost=fit_transfer_window(self.udp,targets[gate],targets[gate+1],
                              release-.50,due,self.cap,stop,random.Random(1410+swap+int(cycles*101)))
                    info["gate_min_dv"]=cost if math.isfinite(cost) else None
                    if branches:
                        leg=min(branches,key=lambda row:max(0.,release-row[2])+
                                max(0.,arrival(row)-shifted[gate+1][0]))
                        shifted[gate]=tuple(leg[2:4])
                        info["gate_seed"]=leg
                    info["solve"]=solve(self.udp,proposal,gate-4,2.3,0.,stop,
                             lambda q:self.submit(q,"low_shell_period_shift",key),self.count,
                             cap_flags=self.exception,start_times=shifted)
                except (BudgetExpired,Paused):
                    info["time_cap"]=True
                self.state["trials"].append({"key":key,"seconds":time.monotonic()-start,**info})
                self.save()
                print(key,info,flush=True)
        self.state["wall_seconds"]=self.state.get("wall_seconds",0.)+time.monotonic()-self.started
        self.save()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--seed-record",type=Path,required=True)
    p.add_argument("--baseline-record",type=Path)
    p.add_argument("--seconds",type=float,default=100.)
    p.add_argument("--trial-seconds",type=float,default=6.)
    PhaseShift(p.parse_args()).run()

"""Restore complete near-deadline tours with both sides of the shell seam free."""
import argparse
from collections import Counter
import itertools
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from global_event_easy import fit_transfer_window
from medium_coupled.boundary import Boundary
from medium_coupled.clock_projection import restore
from medium_coupled.gate_profile_join import ProfileJoin
from medium_coupled.meet import SideGrid
from medium_coupled.run import Paused, solve
from pareto_easy import BudgetExpired, arrival


class CoupledSeam(ProfileJoin):
    def closure(self,prefix,tail,day,stop,stats,key):
        a,b=prefix[-1][1],tail[0][0]
        release,due=arrival(prefix[-1]),tail[0][2]
        result={"edge":[a,b],"release":release,"due":due,
                "minimum_flight_slack":due-release-self.udp.min_tof,"solves":[]}
        if not self.allowed[a,b] or due-release<-.95:
            return result
        rng=random.Random(70113+181*a+b+int(day*1000))
        options=[]
        for pad in (.49,.84,1.22):
            edges,cost=fit_transfer_window(self.udp,a,b,release-pad,due+pad,
                                           self.cap,stop,rng)
            stats["crossing_queries"]+=1
            options.extend(edges)
        options.extend(self.anchors[a,b])
        options={tuple(row) for row in options
                 if row[-1]<=self.cap and abs(row[2]-release)<1.6}
        ordered=sorted(options,key=lambda x:(max(0.,release-x[2])+max(0.,arrival(x)-due),x))
        used=set()
        for bridge in ordered:
            region=round(bridge[2]*8),round(bridge[3]*8)
            if region in used:
                continue
            if len(used)>=3 or time.monotonic()>=stop:
                break
            used.add(region)
            chain=(*prefix,bridge,*tail)
            proposal=self.build(chain)
            if release<=bridge[2]+1e-9 and arrival(bridge)<=due+1e-9:
                self.submit(proposal,"seam_exact_join",key)
            flags=[x[-1]>self.normal for x in chain]
            depart_early=min(0.,bridge[2]-release)
            depart_late=max(0.,arrival(bridge)-due)
            for left in (149,138,124):
                begin=time.monotonic()
                if begin>=stop:
                    break
                times=[tuple(x[2:4]) for x in chain]
                for i in range(left,160):
                    d,f=times[i]
                    times[i]=(d+depart_early*(i-left+1)/(160-left),f)
                for i in range(161,180):
                    d,f=times[i]
                    times[i]=(d+depart_late*(180-i)/19.,f)
                trial={"left":left,"bridge":[bridge[2],bridge[3]],
                       "prefix_shift":depart_early,"suffix_shift":depart_late}
                accept=lambda q:self.submit(q,"wide_coupled_seam",key)
                try:
                    local=min(stop,begin+4.)
                    trial["fit"]=solve(self.udp,proposal,left,4.2,0.,local,accept,stats,
                                           cap_flags=flags,start_times=times)
                    if left==149 and time.monotonic()+.25<stop:
                        virtual={"path":proposal["path"],"dep":[x[0] for x in times],
                                 "tof":[x[1] for x in times]}
                        restored,info=restore(self.udp,virtual,left,day,
                                    min(stop,time.monotonic()+1.8),stats,shift=0.,cap_flags=flags)
                        trial["projection"]=info
                        if time.monotonic()<stop and info["largest_scaled_violation"]<.19:
                            trial["restored_fit"]=solve(self.udp,proposal,left,4.2,0.,
                                    min(stop,time.monotonic()+2.8),accept,stats,
                                    cap_flags=flags,start_times=restored)
                except Paused:
                    trial["time_cap"]=True
                result["solves"].append(trial)
        return result

    def run(self):
        tested={tuple(x["key"]) for x in self.state["trials"]}
        prefixes=self.profiles()[:self.args.prefix_limit]
        for day,step in itertools.product(self.args.deadlines,self.args.steps):
            for k,(prefix,order,_) in enumerate(prefixes):
                key=day,step,k
                if key in tested or time.monotonic()>=self.stop:
                    continue
                begin=time.monotonic()
                stop=min(self.stop,begin+self.args.trial_seconds)
                clock=arrival(prefix[-1])
                stats=Counter()
                info={"connections":[]}
                lattice=SideGrid(self.udp,max(0.,clock-1.25),day,step,.008,stop,True)
                try:
                    template=next(self.terminal_orders(order),order)
                    labels=Boundary.phase(self,lattice,order,"backward",stop,stats,template=template)
                    info["complete_suffixes"]=len(labels)
                    for _,tail in sorted(labels.values(),key=lambda x:x[0])[:6]:
                        if time.monotonic()>=stop:
                            break
                        info["connections"].append(self.closure(prefix,tail[::-1],day,
                                                   min(stop,time.monotonic()+15.),stats,key))
                except BudgetExpired:
                    info["time_cap"]=True
                self.count.update(stats)
                self.state["trials"].append({"key":key,"seconds":time.monotonic()-begin,
                                             "stats":dict(stats),**info})
                self.save()
                print(key,len(info["connections"]),self.state["best"],flush=True)
        self.state["wall_seconds"]=self.state.get("wall_seconds",0.)+time.monotonic()-self.started
        self.save()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",type=Path,required=True)
    parser.add_argument("--seed-record",type=Path,required=True)
    parser.add_argument("--baseline-record",type=Path)
    parser.add_argument("--anchors",type=Path,nargs="*",default=())
    parser.add_argument("--profiles",type=Path,nargs="+",required=True)
    parser.add_argument("--seconds",type=float,default=220.)
    parser.add_argument("--trial-seconds",type=float,default=66.)
    parser.add_argument("--beam",type=int,default=115)
    parser.add_argument("--prefix-limit",type=int,default=1)
    parser.add_argument("--deadlines",type=float,nargs="+",default=(192.,192.24,191.92))
    parser.add_argument("--steps",type=float,nargs="+",default=(.03125,.0390625))
    parser.set_defaults(mode="wide_seam",goal=None,phase=None)
    CoupledSeam(parser.parse_args()).run()

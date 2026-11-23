"""Retiming via absolute-epoch continuous transfer envelopes, then reconstructing the tail."""
import argparse
from collections import Counter
from pathlib import Path
import random
import sys
import time

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medium_coupled.meet import SideGrid
from global_event_easy import fit_transfer_window
from medium_coupled.nonlocal_routes import Surgery
from pareto_easy import BudgetExpired, arrival


class Spine(Surgery):
    def local(self,a,b,release,anchors,cap,stop,stats):
        """Minimize arrival from distinct Lambert timing branches."""
        if not anchors:
            return None
        options={}
        for leg in anchors:
            if leg[2]>=release-1e-9 and leg[-1]<=cap:
                options[leg[2:4]]=leg
        best=min(options.values(),key=arrival) if options else None
        seeds=sorted(anchors,key=lambda l:(max(0.,release-l[2])*3+arrival(l),l[-1]))
        used=set()
        for leg in seeds:
            dep,tof=leg[2:4]
            key=round(dep*9),round(tof*9)
            if key in used:
                continue
            if len(used)>=self.args.starts or time.monotonic()>=stop:
                break
            used.add(key)
            lo=(max(release,dep-.48),max(self.udp.min_tof,tof-.43))
            hi=(max(lo[0]+1e-7,dep+.48),tof+.48)
            if best and lo[0]+lo[1]>arrival(best):
                continue
            def cost(x):
                if time.monotonic()>=stop:
                    raise BudgetExpired
                stats["continuous_evaluations"]+=1
                return float(self.udp.compute_transfer(a,b,float(x[0]),float(x[1])))
            fit=minimize(lambda x:x[0]+x[1],np.clip((dep,tof),lo,hi),jac=lambda x:(1.,1.),
                         method="SLSQP",bounds=list(zip(lo,hi)),
                         constraints={"type":"ineq","fun":lambda x:cap-2e-7-cost(x)},
                         options={"maxiter":24,"ftol":3e-9})
            d,f=map(float,fit.x)
            v=cost(fit.x)
            if v<=cap and (best is None or d+f<arrival(best)):
                best=(a,b,d,f,v)
        return best

    def push(self,source,left,gate,step,stop,tag,stats,*,prefix=None):
        prefix=tuple(self.legs[:left]) if prefix is None else prefix
        path=list(source)
        release=arrival(prefix[-1])
        grid=SideGrid(self.udp,release,max(release+.5,self.args.goal),step,.019,stop,False)
        secondary=SideGrid(self.udp,release,max(release+.5,self.args.goal),.0390625,.006,stop,False)
        chain=list(prefix)
        extras={leg[:2] for leg in self.legs if leg[-1]>self.normal and
                min(self.axis(leg[0]),self.axis(leg[1]))*2>=max(self.axis(leg[0]),self.axis(leg[1]))}
        try:
            for i in range(left,gate):
                if time.monotonic()>=stop:
                    raise BudgetExpired
                a,b=path[i:i+2]
                cap=self.cap if (a,b) in extras or min(self.axis(a),self.axis(b))*2<max(self.axis(a),self.axis(b)) else self.normal
                refs=[l for l in self.anchors[a,b] if l[2]>=release-.5 and arrival(l)<=self.args.goal]
                primary=self.sample(grid,a,b,release,cap>self.normal,stats,secondary)
                refs.extend(l for l in primary if l[2]>=release-1e-9 and l[-1]<=cap)
                # Local phase perturbations can reveal branches lying between
                # the tessellated starts or beside incumbent event anchors.
                for arc in sorted(refs,key=arrival)[:2]:
                    if arc[3]>.18:
                        for slip in (-.12,.12):
                            q=max(release,arc[2]+slip)
                            f=max(self.udp.min_tof,arc[3]-slip*.4)
                            v=float(self.udp.compute_transfer(a,b,q,f))
                            stats["phase_probes"]+=1
                            refs.append((a,b,q,f,v))
                row=self.local(a,b,release,refs,cap,stop,stats)
                if row is None:
                    stats["last_depth"]=i
                    return {"blocking_edge":[a,b],"prefix_clock":release}
                chain.append(row)
                release=arrival(row)
            stats["last_depth"]=gate
            spent=sum(l[-1]>self.normal for l in chain)
            if spent<=4:
                self.save_gate_prefixes(source,{gate:{spent:(release,tuple(chain))}},tag)
            outcome={"arrival":release,"spent":spent,"baseline":arrival(self.legs[gate-1])}
            if spent<=4 and time.monotonic()+1.<stop:
                self.complete(path,tuple(chain),self.args.goal,.03125,stop,tag,outcome)
            return outcome
        finally:
            stats.update(grid.stats)
            stats.update(secondary.stats)

    def run(self):
        gate=self.shells[-1]
        tested={tuple(t["key"]) for t in self.state["trials"]}
        if self.args.probe_long:
            self.probe_long(gate,tested)
            self.state["wall_seconds"]=self.state.get("wall_seconds",0.)+time.monotonic()-self.started
            self.save()
            return
        for left,swap,step in ((142,0,.078125),(124,0,.0625),(148,1,.078125),
                              (148,2,.078125),(144,3,.0625),(152,0,.046875),
                              (135,0,.046875),(113,0,.078125),(142,2,.0546875),
                              (140,5,.0546875),(129,4,.0625),(148,6,.05)):
            key=left,swap,step
            if key in tested or time.monotonic()>=self.stop:
                continue
            path=self.path[:]
            if swap:
                j=gate-swap
                path[j],path[gate]=path[gate],path[j]
            if not self.all_allowed(path) or self.route_floor(path)>5:
                continue
            begin=time.monotonic()
            stats=Counter()
            try:
                out=self.push(path,left,gate,step,min(self.stop,begin+self.args.trial_seconds),key,stats)
            except BudgetExpired:
                out={"time_cap":True}
            self.count.update(stats)
            self.state["trials"].append({"key":key,"seconds":time.monotonic()-begin,
                                        "stats":dict(stats),**out})
            self.save()
            print(key,out,self.state["best"],flush=True)
        self.state["wall_seconds"]=self.state.get("wall_seconds",0.)+time.monotonic()-self.started
        self.save()

    def probe_long(self,gate,tested):
        for i in self.args.positions or self.long[:4]:
            key=("normal_probe",i)
            if key in tested or time.monotonic()>=self.stop:
                continue
            start=time.monotonic()
            stop=min(self.stop,start+self.args.trial_seconds)
            a,b=self.edges[i]
            release=arrival(self.legs[i-1])
            due=arrival(self.legs[i])
            options=[self.legs[i]]
            stats=Counter()
            info={}
            try:
                for phase in (0,1,2,3,4):
                    fit,cost=fit_transfer_window(self.udp,a,b,release,due,self.normal,stop,
                                             random.Random(9507+i*171+phase))
                    options.extend(fit)
                    stats["triangle_probes"]+=1
                candidate=self.local(a,b,release,options,self.normal,stop,stats)
                if candidate:
                    gain=due-arrival(candidate)
                    info={"measured_normal":candidate,"arrival_gain":gain}
                    if gain>.006 and time.monotonic()<stop:
                        info["propagated"]=self.push(self.path,i+1,gate,.065,stop,key,stats,
                                                   prefix=(*self.legs[:i],candidate))
            except BudgetExpired:
                info["time_cap"]=True
            self.count.update(stats)
            self.state["trials"].append({"key":key,"seconds":time.monotonic()-start,
                                        "stats":dict(stats),**info})
            self.save()
            print(key,info,self.state["best"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--seed-record",type=Path,required=True)
    p.add_argument("--baseline-record",type=Path)
    p.add_argument("--anchors",type=Path,nargs="*",default=())
    p.add_argument("--seconds",type=float,default=170.)
    p.add_argument("--trial-seconds",type=float,default=13.)
    p.add_argument("--starts",type=int,default=5)
    p.add_argument("--goal",type=float,default=192.38)
    p.add_argument("--probe-long",action="store_true")
    p.add_argument("--positions",type=int,nargs="+")
    p.set_defaults(mode="continuous_spine",phase=None,beam=None,save_prefixes=True)
    Spine(p.parse_args()).run()

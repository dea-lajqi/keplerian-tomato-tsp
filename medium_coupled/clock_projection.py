"""Sparse deadline feasibility restoration for coupled Medium transfer clocks."""
import argparse
from collections import Counter
from itertools import pairwise
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medium_coupled.checkpoint import atomic_json
from medium_coupled.gate_prefix_polish import PrefixPolish
from medium_coupled.run import solve, Paused
from pareto_easy import arrival


def restore(udp,sol,left,target,stop,stats,*,shift,cap_flags):
    """Minimize violated chronology and exact ΔV constraints against a fixed finish."""
    seq=list(pairwise(sol["path"]))[left:]
    n=len(seq)
    ref=np.array(list(zip(sol["dep"][left:],sol["tof"][left:]))).ravel()
    previous=sol["dep"][left-1]+sol["tof"][left-1] if left else 0.
    lo=np.maximum(ref-np.tile([4.,2.],n),np.tile([previous,udp.min_tof],n))
    hi=np.minimum(ref+np.tile([4.,2.],n),udp.max_time)
    caps=np.where(cap_flags[left:],udp.dv_threshold_exception,udp.dv_threshold)-1e-5
    x0=ref.copy()
    x0[::2]+=np.linspace(0.,shift,n)
    clock_weight=6.0
    cache=None
    nominal=None

    def evaluate(i,d,f):
        if time.monotonic()>=stop:
            raise Paused
        stats["transfer_evaluations"]+=1
        return float(udp.compute_transfer(*seq[i],float(d),float(f)))

    def cost(x):
        nonlocal cache,nominal
        if cache is None or not np.array_equal(x,cache):
            cache=x.copy()
            nominal=np.array([evaluate(i,d,f) for i,(d,f) in enumerate(x.reshape(-1,2))])
        return nominal

    def residual(x):
        d,f=x[::2],x[1::2]
        return np.maximum(0.,np.r_[(cost(x)-caps)/100.,clock_weight*(d[:-1]+f[:-1]-d[1:]),
                                    clock_weight*(d[-1]+f[-1]-target)])

    def jacobian(x):
        nominal=cost(x)
        out=[];col=[];row=[]
        h=3e-7
        for i,(d,f) in enumerate(x.reshape(-1,2)):
            if nominal[i]>caps[i]:
                row.extend((i,i));col.extend((2*i,2*i+1))
                out.extend(((evaluate(i,d+h,f)-nominal[i])/(100*h),
                            (evaluate(i,d,f+h)-nominal[i])/(100*h)))
            if i<n-1 and d+f>x[2*i+2]:
                row.extend((n+i,)*3);col.extend((2*i,2*i+1,2*i+2))
                out.extend((clock_weight,clock_weight,-clock_weight))
        if x[-2]+x[-1]>target:
            row.extend((2*n-1,)*2);col.extend((2*n-2,2*n-1))
            out.extend((clock_weight,clock_weight))
        return csr_matrix((out,(row,col)),shape=(2*n,2*n))

    fit=least_squares(residual,np.clip(x0,lo,hi),jac=jacobian,bounds=(lo,hi),
                      tr_solver="lsmr",max_nfev=100,ftol=1e-9,xtol=1e-9,gtol=1e-9)
    times=list(zip(sol["dep"][:left],sol["tof"][:left]))+fit.x.reshape(-1,2).tolist()
    return times,{"violation_norm":float(np.linalg.norm(fit.fun)),
                  "largest_scaled_violation":float(np.max(fit.fun)),
                  "raw_finish":float(fit.x[-2]+fit.x[-1]),"iterations":fit.nfev}


class ClockProjection(PrefixPolish):
    def run(self):
        output_path=self.args.directory/"gate_pool.json"
        out=json.loads(output_path.read_text()) if output_path.exists() else {}
        tested={tuple(t["key"]) for t in self.state["trials"]}
        pool=[]
        for p in self.args.profiles:
            pool.extend(json.loads(p.read_text()).values())
        pool.sort(key=lambda x:x["arrival"])
        for within in (False,True):
            options=pool[:2] if within else [None]
            for source in options:
                legs=tuple(map(tuple,source["legs"])) if within else tuple(self.legs)
                sol=self.build(legs)
                old=arrival(legs[-1])
                caps=[l[-1]>self.normal for l in legs]
                for gain in (.20,.45,.8):
                    for left in ((153,142,128) if within else (160,151,135)):
                        key=(within,legs[-1][1],left,gain)
                        if key in tested or time.monotonic()>=self.stop:
                            continue
                        start=time.monotonic()
                        stop=min(self.stop,start+self.args.trial_seconds)
                        stats=Counter()
                        info={}
                        def accept(candidate):
                            if not within:
                                return self.submit(candidate,"sparse_deadline_projection",key)
                            actual,spent=self.measure(candidate,stats)
                            if not actual:
                                return False
                            clock=arrival(actual[-1])
                            mark=str(actual[-1][1])
                            if clock<out.get(mark,source)["arrival"]-1e-8:
                                out[mark]={"legs":actual,"remaining":source["remaining"],
                                           "arrival":clock,"spend":spent,"plan":key}
                                atomic_json(output_path,out)
                                print("EARLIER HIGH PREFIX",old,clock,flush=True)
                            return True
                        try:
                            times,info=restore(self.udp,sol,left,old-gain,
                                               min(stop,start+6.),stats,shift=-gain,cap_flags=caps)
                            if time.monotonic()<stop and info["largest_scaled_violation"]<.30:
                                info["polish"]=solve(self.udp,sol,left,3.8,0.,stop,accept,stats,
                                                     cap_flags=caps,start_times=times)
                        except Paused:
                            info["time_cap"]=True
                        self.count.update(stats)
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
    p.add_argument("--anchors",type=Path,nargs="*",default=())
    p.add_argument("--profiles",type=Path,nargs="+",required=True)
    p.add_argument("--seconds",type=float,default=180.)
    p.add_argument("--trial-seconds",type=float,default=9.)
    p.set_defaults(mode="clock_projection",beam=None,goal=None,phase=None)
    ClockProjection(p.parse_args()).run()

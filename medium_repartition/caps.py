"""Tighten evaluator-tolerance-level transfers before accepting a schedule."""
from itertools import pairwise

from scipy.optimize import minimize


def strict_schedule(udp, sol):
    dep, tof = list(sol["dep"]), list(sol["tof"])
    path = sol["path"]
    cap = udp.dv_threshold_exception
    for i,(a,b) in enumerate(pairwise(path)):
        def dv(x):
            return float(udp.compute_transfer(a,b,float(x[0]),float(x[1])))
        measured = dv((dep[i],tof[i]))
        if measured <= cap:
            continue
        if measured > cap + 1.e-6:
            return None
        ready = dep[i-1]+tof[i-1] if i else 0.
        deadline = dep[i+1] if i+1<len(dep) else dep[i]+tof[i]+.00002
        d,f = dep[i],tof[i]
        limits = [(max(ready,d-.0001),min(deadline-udp.min_tof,d+.0001)),
                  (max(udp.min_tof,f-.0001),f+.0001)]
        if any(a0>b0 for a0,b0 in limits):
            return None
        fit=minimize(lambda x: ((x[0]-d)**2+(x[1]-f)**2)*1.e6,
                     [d,f],method="SLSQP",bounds=limits,options={"maxiter":32,"ftol":1e-12},
                     constraints=({"type":"ineq","fun":lambda x:cap-4.e-7-dv(x)},
                                  {"type":"ineq","fun":lambda x:deadline-x[0]-x[1]}))
        x,y=map(float,fit.x)
        if dv((x,y))>cap or x<ready-1e-10 or x+y>deadline+1e-10:
            return None
        dep[i],tof[i]=x,y
    return {"path":path,"dep":dep,"tof":tof}

"""Optional continuous polishing around a short route segment.

Unlike the old 1/16-day grids, these variables are continuous. SciPy SLSQP
may fail on the nonsmooth Lambert minimum; every result gets ESA verification.
"""
from __future__ import annotations
import math


def smooth_segment(udp,sol,left,right,*,maxiter=30,margin=0.03):
    """Optimize departure and flight times of edges [left,right), fixed route.

    Maintains the reference exception/nonexception classification for each edge
    to avoid 'using' an exception taken by an unchanged distant segment.
    Produces a tentative candidate only; call ESA fitness on the full result.
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        return None, 'SciPy not available; optional continuous stage skipped'
    dep=list(sol['dep']);tof=list(sol['tof']);path=sol['path'];n=len(dep)
    if not (0<=left<right<=n and right-left<=8):
        return None,'window bounds invalid'
    m=right-left
    x0=[v for i in range(left,right) for v in (dep[i],tof[i])]
    caps=[]
    for i in range(left,right):
        original=float(udp.compute_transfer(path[i],path[i+1],dep[i],tof[i]))
        caps.append(udp.dv_threshold_exception if original>udp.dv_threshold else udp.dv_threshold)
    predecessor=dep[left-1]+tof[left-1] if left else 0.
    successor=dep[right] if right<n else udp.max_time
    # Tight coordinate bounds prevent SLSQP from exploring an entirely different
    # multi-revolution Lambert branch, then failing to recover the incumbent.
    bounds=[]
    for i in range(left,right):
        bounds.append((max(predecessor,dep[i]-1.5),min(udp.max_time,dep[i]+1.5)))
        bounds.append((udp.min_tof,min(7.,max(7.,tof[i]+1.5))))
    def constraint(x):
        vals=[x[0]-predecessor,successor-(x[-2]+x[-1])]
        for j in range(m):
            d,f=x[2*j],x[2*j+1]
            vals.extend([d, f-udp.min_tof, udp.max_time-d-f])
            if j:
                vals.append(d-(x[2*(j-1)]+x[2*(j-1)+1]))
            dv=udp.compute_transfer(path[left+j],path[left+j+1],float(d),float(f))
            vals.append(caps[j]-margin-dv if math.isfinite(dv) else -1e6)
        return vals
    def objective(x):
        return x[-2]+x[-1]
    try:
        result=minimize(objective,x0,method='SLSQP',bounds=bounds,
                        constraints=[{'type':'ineq','fun':constraint}],
                        options={'maxiter':int(maxiter),'ftol':1e-9,'eps':1e-5})
    except Exception as e:
        return None,f'SLSQP {type(e).__name__}: {e}'
    x=[float(t) for t in result.x]
    vals=constraint(x)
    if not all(math.isfinite(v) and v>=-1e-7 for v in vals):
        return None,f'constraint violation; success={result.success}'
    if objective(x)>=objective(x0)-1e-8:
        return None,'no reduction in local arrival'
    for j,i in enumerate(range(left,right)):
        dep[i],tof[i]=x[2*j],x[2*j+1]
    return {'path':path[:],'dep':dep,'tof':tof},f'SLSQP arrival reduction {objective(x0)-objective(x):.9f} d'

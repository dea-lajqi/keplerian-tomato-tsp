"""Measure the recurring 62→85 bottleneck and normal two-transfer detours."""
import argparse
from pathlib import Path
import time

from medium_coupled.checkpoint import atomic_json
from medium_coupled.meet import SideGrid
from global_event_easy import geometry
from medium_frontier.nodal import discover, orbit
from medium_overnight.validation import official_udp
from pareto_easy import BudgetExpired, arrival


def diagnose(directory, seconds):
    began=time.monotonic()
    end=began+seconds
    udp=official_udp(Path.cwd())
    (normal,_),bounds=geometry(udp,4.9048695e12)
    high=[i for i in range(181) if udp.tomato_orbital_parameters[i,0]>8e6]
    nodes={i:orbit(udp,i) for i in high}
    a,b=62,85
    via=[n for n in high if n not in (a,b) and normal[a,n] and normal[n,b]]
    releases=(142.988,144.915,145.869)
    results=[]
    for r in releases:
        stop=min(end,time.monotonic()+12.)
        e,cost,calls=discover(udp,nodes,a,b,r,r+13.,udp.dv_threshold,stop)
        grid=SideGrid(udp,r,r+14.,.045,.009,stop,False)
        try:
            direct=(*e,*(l for l in options(udp,grid,a,b,r) if l is not None))
            joined=[]
            for n in via:
                first=options(udp,grid,a,n,r)
                for left in first:
                    for right in options(udp,grid,n,b,arrival(left)):
                        if right[2]>=arrival(left)-1e-9:
                            joined.append((arrival(right),n,left,right))
            direct=sorted(set(direct),key=arrival)
            joined.sort()
        except BudgetExpired:
            direct=sorted(e,key=arrival)
            joined=[]
        results.append({"release":r,"best_sampled_direct_dv":cost,
                        "direct_first":direct[:8],"via_first":joined[:12],
                        "direct_node_evaluations":calls})
        atomic_json(directory/"arc_62_85.json",{"geometry_floor":float(bounds[a,b]),
                    "geometry_via":via,"results":results,"wall_seconds":time.monotonic()-began})
        print(r,len(direct),joined[:1],flush=True)


def options(udp,grid,a,b,ready):
    choices=[]
    for row in grid.edges(a,b,ready-grid.origin,7.,8.,False):
        if row is None:
            continue
        d,f=grid.origin+row[2],row[3]
        dv=float(udp.compute_transfer(a,b,d,f))
        if d>=ready-1e-9 and dv<=udp.dv_threshold:
            choices.append((a,b,d,f,dv))
    return choices


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--seconds",type=float,default=45.)
    a=p.parse_args()
    diagnose(a.directory,a.seconds)

"""Discover missed transfer branches near mutual orbital nodes."""
import argparse
from collections import defaultdict
from itertools import pairwise
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize

from medium_coupled.checkpoint import atomic_json
from medium_frontier.crossing_bound import MU_MOON
from medium_overnight.validation import official_udp


def mean_at_direction(eccentricity, periapsis, transverse, direction):
    true = math.atan2(direction @ transverse, direction @ periapsis)
    eccentric = math.atan2(math.sqrt(1.-eccentricity**2)*math.sin(true),
                           eccentricity+math.cos(true))
    return eccentric - eccentricity*math.sin(eccentric)


def orbit(udp, node):
    r, v = map(np.asarray,udp.tomatoes[node].eph(0.))
    axis = float(udp.tomato_orbital_parameters[node,0])
    h = np.cross(r,v)
    normal = h/np.linalg.norm(h)
    evec = np.cross(v,h)/MU_MOON - r/np.linalg.norm(r)
    e = float(np.linalg.norm(evec))
    periapsis = evec/e
    transverse = np.cross(normal,periapsis)
    period = 2.*math.pi*math.sqrt(axis**3/MU_MOON)/86400.
    current = mean_at_direction(e,periapsis,transverse,r/np.linalg.norm(r))
    return normal, period, current, e, periapsis, transverse


def passages(state, line, low, high):
    _, period, current, e, p, q = state
    for sign in (-1.,1.):
        mean = mean_at_direction(e,p,q,sign*line)
        epoch = (mean-current)*period/(2.*math.pi)
        a = math.ceil((low-epoch)/period)
        b = math.floor((high-epoch)/period)
        for tick in range(a,b+1):
            yield epoch + tick*period


def discover(udp, states, a, b, release, deadline, cap, stop):
    line = np.cross(states[a][0],states[b][0])
    length = np.linalg.norm(line)
    if length < 1e-8 or deadline-release < udp.min_tof:
        return [], None, 0
    line /= length
    # Keep windows wider for nearly coplanar pairs. The nodes seed the local
    # search, while measured impulses determine whether any arc is admissible.
    radius = max(.033,min(.16,.012/length))
    starts = tuple(passages(states[a],line,release-radius,deadline+radius))
    ends = tuple(passages(states[b],line,release-radius,deadline+radius))
    centers = set()
    for a_day in starts:
        for b_day in ends:
            if udp.min_tof-radius < b_day-a_day < 8.:
                d = max(release,a_day)
                end = min(deadline,b_day)
                if end-d >= udp.min_tof:
                    centers.add((d,end))
    calls = 0
    best = math.inf

    def measure(x):
        nonlocal calls
        calls += 1
        d, end = map(float,x)
        flight = end-d
        return float(udp.compute_transfer(a,b,d,flight)) if flight > 0. else 1.e6

    raw = sorted((measure(x),x) for x in centers)
    found = set()
    for cost, center in raw[:16]:
        if time.monotonic()>=stop:
            break
        bounds = ((max(release,center[0]-radius),min(deadline-udp.min_tof,center[0]+radius)),
                  (max(release+udp.min_tof,center[1]-radius),min(deadline,center[1]+radius)))
        chronology = {"type":"ineq","fun":lambda x:x[1]-x[0]-udp.min_tof}
        fit = minimize(measure,center,method="SLSQP",bounds=bounds,
                       constraints=chronology,options={"maxiter":17,"ftol":1e-7})
        for x in (center,fit.x):
            d, end = map(float,x)
            dv = measure(x)
            best = min(best,dv)
            if udp.min_tof-1e-10 <= end-d and dv<=cap and d>=release-1e-9 and end<=deadline+1e-9:
                found.add((a,b,d,end-d,dv))
        if fit.fun+1e-5 > cap:
            continue
        constraint = {"type":"ineq","fun":lambda x:cap-measure(x)}
        for k,direction in ((0,-1),(1,1)):
            result = minimize(lambda x:direction*x[k],fit.x,method="SLSQP",bounds=bounds,
                              constraints=(chronology,constraint),
                              options={"maxiter":19,"ftol":1e-9})
            d,end = map(float,result.x)
            dv = measure(result.x)
            if udp.min_tof-1e-10 <= end-d and dv<=cap and d>=release-1e-9 and end<=deadline+1e-9:
                found.add((a,b,d,end-d,dv))
    return sorted(found), best if math.isfinite(best) else None, calls


def run(args):
    started = time.monotonic()
    stop = started + args.seconds
    udp = official_udp(Path.cwd())
    states = [orbit(udp,node) for node in range(udp.n_tomatoes)]
    bank = defaultdict(set)
    trials = []
    cases = set()
    for p in args.near:
        data = json.loads(p.read_text())
        sol = data["solution"]
        j = data["trial"][-2]
        edges = list(pairwise(sol["path"]))
        for i in range(j-2,j+3):
            start = sol["dep"][i-1]+sol["tof"][i-1]
            due = sol["dep"][i+1]
            a,b = edges[i]
            cap = udp.dv_threshold_exception if udp.compute_transfer(a,b,sol["dep"][i],sol["tof"][i]) > udp.dv_threshold else udp.dv_threshold
            if due-start >= udp.min_tof:
                cases.add((a,b,start,due,cap,"fixed_neighbors"))
            # The enlarged box can support paired changes in the next join.
            cases.add((a,b,start-.25,max(due,start+.42)+.25,cap,"coordinated"))
    for a,b,left,right,cap,label in sorted(cases,key=lambda row: (row[-1] != "fixed_neighbors",row[1])):
        if time.monotonic()>=stop:
            break
        events, best, calls = discover(udp,states,a,b,left,right,cap,
                                      min(stop,time.monotonic()+2.7))
        bank[a,b].update(events)
        trials.append({"edge":[a,b],"release":left,"deadline":right,"limit":cap,
                       "type":label,"best_dv":best,"events":len(events),"calls":calls})
        if len(trials)%10==0:
            atomic_json(args.directory/"events.json",
                        {"rows":{f"{a},{b}":sorted(rows) for (a,b),rows in bank.items() if rows}})
    atomic_json(args.directory/"events.json",
                {"rows":{f"{a},{b}":sorted(rows) for (a,b),rows in bank.items() if rows}})
    atomic_json(args.directory/"summary.json",{"trials":trials,"wall_seconds":time.monotonic()-started})
    print(len(trials),sum(map(len,bank.values())),flush=True)


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--near",type=Path,nargs="+",required=True)
    p.add_argument("--seconds",type=float,default=55.)
    run(p.parse_args())

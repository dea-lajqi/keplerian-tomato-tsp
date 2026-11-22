"""Global topology and time-expanded mixed-integer search for ESA Easy."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import time
from uuid import uuid4

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, differential_evolution, linprog, milp, minimize
from scipy.sparse import coo_matrix, vstack

from coupled_easy import optimize_block, restore_suffix
from elite_easy import anchor_bank, flight_time, scan_archive
from focus_easy import refine_joint, retime, shortest_band
from global_dp_easy import Grid, evaluator
from kttsp_esa_solver import ROOT, validate
from pareto_easy import BudgetExpired, arrival, write_json

FILES = ROOT/"results/esa"
SEED = FILES/"easy_elite_20261121T134844Z_f0fefc4b_0001.json"


class Rows:
    """Sparse constraint builder for SciPy/HiGHS."""
    def __init__(self):
        self.i, self.j, self.v, self.lower, self.upper = [], [], [], [], []

    def add(self, entries, lower=-np.inf, upper=np.inf):
        index = len(self.lower)
        for col,val in entries:
            if val:
                self.i.append(index)
                self.j.append(col)
                self.v.append(val)
        self.lower.append(lower)
        self.upper.append(upper)

    def linear(self, size):
        matrix = coo_matrix((self.v,(self.i,self.j)),shape=(len(self.lower),size)).tocsc()
        return LinearConstraint(matrix,self.lower,self.upper)

    def duals(self,c,lo,hi,seconds):
        linear=self.linear(len(c))
        A=linear.A.tocsr()
        lower=np.asarray(self.lower)
        upper=np.asarray(self.upper)
        eq=np.flatnonzero(lower==upper)
        high=np.flatnonzero(np.isfinite(upper)&(lower!=upper))
        low=np.flatnonzero(np.isfinite(lower)&(lower!=upper))
        inequalities=vstack((A[high],-A[low]),format="csr")
        fit=linprog(c,A_ub=inequalities,b_ub=np.r_[upper[high],-lower[low]],
                    A_eq=A[eq],b_eq=upper[eq],bounds=np.column_stack((lo,hi)),
                    method="highs",options={"time_limit":seconds})
        if not fit.success:
            return None,None
        dual=np.zeros(len(self.lower))
        dual[eq]=fit.eqlin.marginals
        dual[high]+=fit.ineqlin.marginals[:len(high)]
        dual[low]-=fit.ineqlin.marginals[len(high):]
        return dual,float(fit.fun)


class Pricing:
    def __init__(self,dual,exception,chrono,goals,cdfs,bias,forbid,forbid_row):
        self.dual=dual
        self.exception=exception
        self.chrono=chrono
        self.goals=goals
        self.cdfs=cdfs
        self.bias=bias
        self.forbid=forbid
        self.forbid_row=forbid_row

    def reduced_cost(self,a,b,dep,tof,extra):
        end=dep+tof
        y=self.dual
        score=(self.bias*tof-y[2*a]-y[2*b+1]-extra*y[self.exception]
               -dep*y[self.chrono[a]]+end*(y[self.chrono[b]]+y[self.goals[b]]))
        if self.forbid_row is not None and (a,b) in self.forbid:
            score-=y[self.forbid_row]
        score-=sum(y[k] for t,k in self.cdfs[a] if dep<=t)
        score+=sum(y[k] for t,k in self.cdfs[b] if end<=t+1e-6)
        return score


def geometry(udp, mu):
    """Necessary impulse bounds from conserved specific energy and angular momentum."""
    a,e,inc,raan=udp.tomato_orbital_parameters[:,:4].T
    n=np.column_stack((np.sin(inc)*np.sin(raan),-np.sin(inc)*np.cos(raan),np.cos(inc)))
    h=(np.sqrt(mu*a*(1-e**2)))[:,None]*n
    radius=a*(1+e)
    speed=np.sqrt(mu*(1+e)/(a*(1-e)))
    h_gap=np.linalg.norm(h[:,None]-h[None,:],axis=2)
    e_gap=np.abs(np.subtract.outer(-mu/(2*a),-mu/(2*a)))
    floors=h_gap/np.maximum.outer(radius,radius)
    allowed=[]
    for dv in (udp.dv_threshold,udp.dv_threshold_exception+1e-6):
        ok=np.zeros_like(floors,dtype=bool)
        for i,j in np.ndindex(ok.shape):
            if i==j or floors[i,j]>dv+1e-5:
                continue
            # For x at the larger-radius orbit, angular momentum requires
            # r_large*x + r_small*(dv-x) >= |H_i-H_j|.
            small,large=sorted((i,j),key=lambda k:radius[k])
            span=radius[large]-radius[small]
            lo=max(0.,(h_gap[i,j]-radius[small]*dv)/span) if span>0 else 0.
            if lo>dv+1e-7:
                continue
            energy=lambda x: speed[large]*x+speed[small]*(dv-x)+(x*x+(dv-x)**2)/2
            if e_gap[i,j]<=max(energy(min(lo,dv)),energy(dv))+1e-4:
                ok[i,j]=True
        allowed.append(ok)
    # In a cross-shell transfer the common angular momentum and radial
    # speed must also be sufficient to span the two endpoint radii.
    # Minimize a relaxation over rectangular cells in |h| and plane angle.
    # Each cell uses independent lower bounds, so the rejection is safe.
    for i,j in np.ndindex(floors.shape):
        if i>=j or not allowed[1][i,j] or max(a[i],a[j])<2*min(a[i],a[j]):
            continue
        low,high=sorted((i,j),key=lambda k:a[k])
        angles=math.acos(float(np.clip(n[low]@n[high],-1.,1.)))
        hl,hh=np.linalg.norm(h[low]),np.linalg.norm(h[high])
        radius_min=a*(1-e)
        max_dv=udp.dv_threshold_exception+1e-6
        bounds=np.linspace(max(0.,hl-radius[low]*max_dv),hl+radius[low]*max_dv,77)
        h0,h1=bounds[:-1,None],bounds[1:,None]
        angle_grid=np.linspace(0.,angles,61)
        cos_low=np.cos(angle_grid[:-1])[None,:]
        cos_high=np.cos(angles-angle_grid[1:])[None,:]

        def transverse(home,costheta,r):
            part=np.clip(home*costheta,h0,h1)
            return np.sqrt(np.maximum(0.,(part-home)**2+2*part*home*(1-costheta)))/r

        radial_squared=(2*mu*(1/radius[low]-1/radius_min[high])
                        -h1*h1*(1/radius_min[low]**2-1/radius[high]**2))
        nominal_radial=e[low]*math.sqrt(mu/(a[low]*(1-e[low]**2)))
        residual=np.maximum(0.,np.sqrt(np.maximum(radial_squared,0.))-nominal_radial)
        combined=np.hypot(transverse(hl,cos_low,radius[low]),residual)+transverse(
            hh,cos_high,radius[high])
        value=float(np.min(combined))
        floors[i,j]=floors[j,i]=max(floors[i,j],value)
        if value>max_dv+1e-5:
            allowed[1][i,j]=allowed[1][j,i]=False
    return tuple(allowed),floors


def add_degrees(rows, arcs, n):
    out,ins=defaultdict(list),defaultdict(list)
    for k,arc in enumerate(arcs):
        a,b=arc[:2]
        out[a].append(k)
        ins[b].append(k)
    for i in range(n+1):
        rows.add(((k,1) for k in out[i]),1,1)
        rows.add(((k,1) for k in ins[i]),1,1)
    return out,ins


def build_path(arcs, selected, dummy):
    next_arc={arcs[k][0]:arcs[k] for k in selected}
    if len(next_arc)!=dummy+1:
        return None
    a=next_arc[dummy][1]
    path=[a]
    chosen=[]
    while a!=dummy and len(path)<=dummy+1:
        leg=next_arc[a]
        a=leg[1]
        if a==dummy:
            break
        chosen.append(leg)
        path.append(a)
    if len(path)!=dummy or len(set(path))!=dummy or a!=dummy:
        return None
    return tuple(path),tuple(chosen)


def components(arcs, selected):
    unseen={arcs[k][0]:arcs[k][1] for k in selected}
    groups=[]
    while unseen:
        start=next(iter(unseen))
        group=set()
        while start in unseen:
            group.add(start)
            start=unseen.pop(start)
        groups.append(frozenset(group))
    return groups


def nondominated_events(events,threshold,cap):
    """Remove events that leave earlier and arrive later on the same arc."""
    by_pair=defaultdict(set)
    for l in events:
        if arrival(l)<=cap+1e-6:
            by_pair[l[:2]].add(l)
    filtered=[]
    for group in by_pair.values():
        best=[math.inf,math.inf]
        for l in sorted(group,key=lambda x:(-x[2],arrival(x),x[4]>threshold)):
            cost=int(l[4]>threshold)
            finish=arrival(l)
            if finish>=min(best[:cost+1]):
                continue
            filtered.append(l)
            best[cost]=finish
    return filtered


def fit_transfer_window(udp,a,b,release,latest,threshold,deadline,rng):
    """Search the triangular departure/arrival window for a bridge."""
    span=latest-release-udp.min_tof
    if span<=0:
        return (),math.inf

    def map_time(x):
        dep=release+span*float(x[0])
        tof=udp.min_tof+max(0.,latest-dep-udp.min_tof)*float(x[1])
        return dep,tof

    def cost(x):
        if time.monotonic()>=deadline:
            raise BudgetExpired
        return float(udp.compute_transfer(a,b,*map_time(x)))

    fit=differential_evolution(cost,((0.,1.),(0.,1.)),maxiter=32,popsize=7,tol=1e-6,
                               polish=False,seed=rng.randrange(2**32))
    value=float(fit.fun)
    if value>threshold:
        return (),value
    t,f=map_time(fit.x)
    actual=[(a,b,t,f,value)]
    for objective in (lambda x:sum(map_time(x)),lambda x:map_time(x)[1]):
        estimate=minimize(objective,fit.x,method="SLSQP",bounds=((0.,1.),(0.,1.)),
                          constraints=({"type":"ineq","fun":lambda x:threshold-2e-7-cost(x)},),
                          options={"maxiter":31,"ftol":1e-8,"eps":2e-7})
        dv=cost(estimate.x)
        if dv<=threshold:
            t,f=map_time(estimate.x)
            actual.append((a,b,t,f,dv))
    return tuple(actual),value


def clock_labels(udp,path,real,synthetic,cap):
    """Exact earliest arrivals by exception count and unresolved-event count."""
    options=defaultdict(list)
    for leg in real|synthetic:
        if arrival(leg)<=cap:
            options[leg[:2]].append((leg,int(leg in synthetic and leg not in real)))
    frontier={(0,0):(0.,())}
    for a,b in zip(path,path[1:]):
        updated={}
        for (used,unknown),(day,chain) in frontier.items():
            for leg,artificial in options[a,b]:
                cost=used+int(leg[4]>udp.dv_threshold)
                if cost>udp.n_dv_exception or leg[2]+1e-6<day:
                    continue
                key=(cost,unknown+artificial)
                end=arrival(leg)
                if key not in updated or end<updated[key][0]:
                    updated[key]=(end,chain+(leg,))
        if not updated:
            return None
        # Earlier arrival with no greater count of either resource dominates.
        frontier={}
        for key,(day,chain) in sorted(updated.items(),key=lambda row:(sum(row[0]),row[1][0])):
            if any(a<=key[0] and b<=key[1] and time_value<=day
                   for (a,b),(time_value,_) in frontier.items()):
                continue
            frontier[key]=(day,chain)
    (used,unknown),(_,chain)=min(frontier.items(),key=lambda row:(row[0][1],row[1][0]))
    return chain,unknown


def event_mip(udp, events, upper, seconds, weight=0., forbid=None, diversity=0,
              flight_primary=False, price_lp=False, artificial=(), forced_exception=None):
    """Choose actual transfer epochs. Positive flight times exclude subtours."""
    n=udp.n_tomatoes
    genuine=set(events)
    synthetic=set(artificial)-genuine
    kept=nondominated_events(genuine,udp.dv_threshold,upper)
    arcs=sorted(set(kept)|{leg for leg in synthetic if arrival(leg)<=upper+1e-6})
    arcs.extend((n,i,0.,0.,0.) for i in range(n))
    arcs.extend((i,n,upper,0.,0.) for i in range(n))
    count=len(arcs)
    rows=Rows()
    out,ins=add_degrees(rows,arcs,n)
    exception_row=len(rows.lower)
    rows.add(((k,1) for k,l in enumerate(arcs) if l[4]>udp.dv_threshold),upper=udp.n_dv_exception)
    if forced_exception is not None:
        rows.add(((k,1) for k,l in enumerate(arcs) if l[:2]==forced_exception and l[4]>udp.dv_threshold),
                 lower=1.)
    forbid_row=None
    if forbid and diversity:
        forbid_row=len(rows.lower)
        rows.add(((k,1) for k,l in enumerate(arcs) if l[:2] in forbid),
                 upper=len(forbid)-diversity)
    chrono,goals,cdfs={}, {}, defaultdict(list)
    for i in range(n):
        chrono[i]=len(rows.lower)
        rows.add([*((k,arcs[k][2]) for k in out[i]),*((k,-arrival(arcs[k])) for k in ins[i])],
                 lower=-1e-6)
        goals[i]=len(rows.lower)
        rows.add([(count,1),*((k,-arrival(arcs[k])) for k in ins[i])],lower=0.)
        # Cumulative compatibility cuts greatly tighten the LP when incoming
        # and outgoing event choices are fractional.
        ticks=sorted({arcs[k][2] for k in out[i] if arcs[k][1]!=n})
        if len(ticks)>28:
            ticks=[ticks[k*len(ticks)//28] for k in range(28)]
        for t in ticks:
            cdfs[i].append((t,len(rows.lower)))
            rows.add([*((k,1) for k in out[i] if arcs[k][2]<=t),
                      *((k,-1) for k in ins[i] if arrival(arcs[k])<=t+1e-6)],upper=0.)
    c=(np.array([l[3] for l in arcs]+[1e-4]) if flight_primary else
       np.array([weight*l[3] for l in arcs]+[1.]))
    if synthetic:
        c[:count]+=np.asarray([200.*(l in synthetic) for l in arcs])
    lo=np.zeros(count+1); hi=np.ones(count+1); hi[-1]=upper
    integer=np.ones(count+1,dtype=int); integer[-1]=0
    dual,lp_bound=(rows.duals(c,lo,hi,min(seconds*.25,5.))
                   if price_lp and forced_exception is None else (None,None))
    result=milp(c,integrality=integer,bounds=Bounds(lo,hi),constraints=rows.linear(count+1),
                options={"time_limit":seconds,"mip_rel_gap":1e-7})
    report={"status":int(result.status),"columns":count,"rows":len(rows.lower),
            "sample_cap_days":upper,"weight_for_flight":weight,"flight_primary":flight_primary,
            "input_real_events":len(genuine),"real_events_after_pruning":len(kept)}
    bound=getattr(result,"mip_dual_bound",None)
    if bound is not None and np.isfinite(bound):
        report["sampled_dual_bound"]=float(bound)
    if dual is not None:
        report["sampled_lp_bound"]=lp_bound
        report["_pricing"]=Pricing(dual,exception_row,chrono,goals,cdfs,1. if flight_primary else weight,
                                    forbid,forbid_row)
    if result.x is None:
        return None,report
    selected=np.flatnonzero(result.x[:count]>.5)
    if synthetic:
        report["artificial_legs"]=[arcs[k] for k in selected if arcs[k] in synthetic]
    return build_path(arcs,selected,n),report


class Topology:
    def __init__(self,udp,allowed,plane,archive):
        self.udp=udp
        self.n=udp.n_tomatoes
        self.arcs=[]
        self.price=[]
        self.quality={}
        self.cuts=set()
        self.forbidden=[]
        self.by_arc={}
        self.failed=Counter()
        for a,b in zip(*np.nonzero(allowed[1])):
            for extra in (0,1):
                if extra==0 and not allowed[0][a,b]:
                    continue
                key=(int(a),int(b),extra)
                norm=plane[a,b]
                cost=(.37+.0035*norm) if extra else (.43+.018*norm)
                self.by_arc[key]=len(self.arcs)
                self.arcs.append(key)
                self.price.append(cost)
        for i in range(self.n):
            self.arcs.extend(((self.n,i,0),(i,self.n,0)))
            self.price.extend((0.,0.))
        for routes in archive.values():
            for _,legs,_ in routes:
                self.observe(legs)
        # Initialize obvious disconnected normal components as connectivity cuts.
        unseen=set(range(self.n))
        while unseen:
            group={unseen.pop()}
            frontier=list(group)
            while frontier:
                i=frontier.pop()
                for j in np.flatnonzero(allowed[0][i]):
                    if int(j) in unseen:
                        unseen.remove(int(j))
                        group.add(int(j))
                        frontier.append(int(j))
            if len(group)>1:
                self.cuts.add(frozenset(group))

    def observe(self,legs):
        for a,b,_,tof,dv in legs:
            key=(a,b,int(dv>self.udp.dv_threshold))
            if key in self.by_arc:
                self.quality[key]=min(self.quality.get(key,math.inf),tof)

    def propose(self,rng,deadline,reference,minimum_difference,variant,maximum_difference=None,
                forced=(),forced_exception=None):
        arcs=self.arcs
        c=np.asarray(self.price,dtype=float)
        for k,(a,b,e) in enumerate(arcs):
            if a==self.n or b==self.n:
                c[k]=rng.uniform(0.,.16)
                continue
            if (a,b,e) in self.quality:
                c[k]=self.quality[a,b,e]
            else:
                c[k]+=(.65 if e else 1.0)
            c[k]+=.19*self.failed[a,b]+rng.uniform(0.,variant)
        for attempt in range(13):
            remaining=deadline-time.monotonic()
            if remaining<=0:
                break
            rows=Rows()
            add_degrees(rows,arcs,self.n)
            rows.add(((k,e) for k,(_,_,e) in enumerate(arcs)),upper=self.udp.n_dv_exception)
            for group in self.cuts:
                rows.add(((k,1) for k,(a,b,_) in enumerate(arcs) if a in group and b not in group),lower=1.)
            for known in self.forbidden:
                rows.add(((k,1) for k,(a,b,_) in enumerate(arcs) if (a,b) in known),upper=self.n-2)
            if minimum_difference or maximum_difference is not None:
                rows.add(((k,1) for k,(a,b,_) in enumerate(arcs) if (a,b) in reference),
                         lower=-np.inf if maximum_difference is None else self.n-1-maximum_difference,
                         upper=self.n-1-minimum_difference)
            for edge in forced:
                rows.add(((k,1) for k,(a,b,_) in enumerate(arcs) if (a,b)==edge),lower=1.)
            if forced_exception is not None:
                rows.add(((k,1) for k,(a,b,e) in enumerate(arcs) if (a,b)==forced_exception and e),lower=1.)
            result=milp(c,integrality=np.ones(len(arcs)),bounds=Bounds(0.,1.),
                        constraints=rows.linear(len(arcs)),
                        options={"time_limit":min(remaining,3.5),"mip_rel_gap":.025})
            if result.x is None:
                break
            selected=np.flatnonzero(result.x>.5)
            route=build_path(arcs,selected,self.n)
            if route:
                path,links=route
                self.forbidden.append(set(zip(path,path[1:])))
                return path,tuple((a,b) for a,b,e in links if e),float(result.fun)
            groups=components(arcs,selected)
            self.cuts.update(group for group in groups if len(group)<self.n+1)
        return None


class Search:
    def __init__(self,args):
        self.args=args
        self.started=time.monotonic()
        self.deadline=self.started+args.seconds
        self.module=evaluator()
        self.udp=self.module.udp
        self.allowed,self.floors=geometry(self.udp,self.module.MU_MOON)
        self.archive,self.loaded=scan_archive(self.udp)
        raw=json.loads(args.seed_file.read_text())
        self.baseline=validate(self.udp,raw["path"],raw["transfers"])
        if not self.baseline or not self.baseline["valid"]:
            raise ValueError("Incumbent was rejected by ESA")
        self.best_path=tuple(raw["path"])
        self.best=tuple((*l[:4],float(self.udp.compute_transfer(*l[:4]))) for l in raw["transfers"])
        self.best_days=self.baseline["fitness"][0]
        self.original_path=self.best_path
        self.topology=Topology(self.udp,self.allowed,self.floors,self.archive)
        self.anchors=anchor_bank(self.archive)
        self.priced_events=set()
        self.fragments=set()
        self.fragment_ticks=set()
        for p in FILES.glob("easy_focus*summary.json"):
            rows=json.loads(p.read_text()).get("flight_atlas",[])
            for row in rows:
                for leg in row["anchors"]:
                    self.import_event(leg)
        for p in args.event_bank:
            for leg in json.loads(p.read_text()).get("priced_events",[]):
                self.import_event(leg)
        self.sampled={}
        self.grids={}
        self.seen={tuple(l[:4] for l in self.best)}
        self.route_set=set()
        self.rng=random.Random(args.random_seed)
        self.stats=Counter()
        self.history=[]
        self.last_gain=self.started
        self.recent_feasible=self.started
        self.best_new=None
        self.pricing=None
        self.priced_signatures=set()
        self.pricing_failures=Counter()
        self.deadline_columns=defaultdict(set)
        self.conversion_counts={}
        self.deadline_seen=set()
        self.initial_orders=set(self.archive)
        self.archived_starts=sorted((min(es,key=lambda x:x[0])[0],p) for p,es in self.archive.items())
        self.run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid4().hex[:8]

    def import_event(self,leg):
        a,b,dep,tof,_=leg
        dv=float(self.udp.compute_transfer(a,b,dep,tof))
        if dv<=self.udp.dv_threshold_exception+1e-6:
            measured=(a,b,dep,tof,dv)
            old=self.anchors.get((a,b),())
            if measured not in old:
                self.anchors[a,b]=(*old,measured)
            self.topology.observe((measured,))

    def independent_flight(self,path):
        table={0:0.}
        for edge in zip(path,path[1:]):
            choices={}
            for leg in self.anchors.get(edge,()):
                extra=int(leg[4]>self.udp.dv_threshold)
                choices[extra]=min(choices.get(extra,math.inf),leg[3])
            new={}
            for used,value in table.items():
                for extra,duration in choices.items():
                    count=used+extra
                    if count<=self.udp.n_dv_exception:
                        new[count]=min(new.get(count,math.inf),value+duration)
            table=new
        return min(table.values(),default=None)

    def relaxed_flights(self,path):
        states={0:(0.,())}
        for a,b in zip(path,path[1:]):
            next_states={}
            for extra in (0,1):
                key=(a,b,extra)
                index=self.topology.by_arc.get(key)
                if index is None:
                    continue
                duration=self.topology.quality.get(key,self.topology.price[index]+.25)
                for used,(total,legs) in states.items():
                    count=used+extra
                    if count<=self.udp.n_dv_exception and (
                        count not in next_states or total+duration<next_states[count][0]
                    ):
                        next_states[count]=(total+duration,legs+((duration,extra),))
            states=next_states
        return min(states.values(),default=None)

    def seed_deadline(self,path,cap,phase):
        relaxed=self.relaxed_flights(path)
        if not relaxed:
            return
        flight,layout=relaxed
        minimum=self.udp.min_tof
        fixed_minimum=len(layout)*minimum
        usable=cap-.03-fixed_minimum
        scale=min(1.,usable/max(1e-6,flight-fixed_minimum))
        durations=[minimum+(f-minimum)*scale for f,_ in layout]
        slack=max(0.,cap-.03-sum(durations))
        # Several ways to spend slack supply different clocks to the
        # optimizer. These artificial columns are never ESA candidates.
        times=np.linspace(.04,1.,len(layout))
        weights=(times**2 if phase==0 else (1.04-times)**2 if phase==1 else
                 np.ones(len(layout)) if phase==2 else .15+np.sin(np.pi*times)**2)
        weights=weights/weights.sum()
        day=0.
        for (a,b),duration,(_,extra),mass in zip(zip(path,path[1:]),durations,layout,weights):
            dep=day+slack*float(mass)
            dv=(self.udp.dv_threshold_exception-.01 if extra else self.udp.dv_threshold-.01)
            self.deadline_columns[cap].add((a,b,dep,duration,dv))
            day=dep+duration

    def shifted_deadline(self,cap,first,last,variant):
        """Optimistic time-compression templates concentrated in one block."""
        original=self.best
        span=self.best_days-cap+.025
        if span<=0:
            return
        waits=np.diff([0.,*(arrival(leg) for leg in original)])
        flights=np.asarray([leg[3] for leg in original])
        waits=np.maximum(0.,waits-flights)
        release=np.zeros(len(original))
        release[first:last]=waits[first:last]
        # First consume available waiting, then distribute the remaining
        # reduction over flights. The model still treats all changed epochs
        # as synthetic until priced against the actual orbit geometry.
        amount=float(release.sum())
        if amount:
            waits-=release*min(1.,span/amount)
        left=max(0.,span-amount)
        if left:
            capacity=np.maximum(0.,flights[first:last]-self.udp.min_tof)
            if sum(capacity)<left:
                return
            if variant:
                lever=capacity*(.65+np.linspace(0.,.7,len(capacity)))
            else:
                lever=capacity
            reduction=np.minimum(capacity,lever*(left/sum(lever)))
            missing=left-sum(reduction)
            if missing>0:
                remaining=capacity-reduction
                if sum(remaining)>0:
                    reduction+=remaining*(missing/sum(remaining))
            flights[first:last]-=reduction
        day=0.
        for (a,b,_,_,dv),w,f in zip(original,waits,flights):
            dep=day+float(w)
            self.deadline_columns[cap].add((a,b,dep,float(f),
                                            self.udp.dv_threshold_exception-.01 if dv>100 else
                                            self.udp.dv_threshold-.01))
            day=dep+float(f)

    def repair_runs(self,legs,artificial,trial,i):
        target=set(artificial)
        indices=[j for j,l in enumerate(legs) if l in target]
        if not indices or len(indices)>16:
            return
        # Warm-start a whole-mission nonlinear solve from the MILP timing.
        # The artificial ΔV values set only the exception class; each
        # proposed result is measured and fully validated before saving.
        order=(legs[0][0],*(l[1] for l in legs))
        first,last=max(0,indices[0]-1),min(len(legs),indices[-1]+2)
        refine_joint(self.udp,legs,self.deadline,
                     lambda s:self.submit(order,s,trial,"deadline bridge with joint continuous repair"),
                     .01,self.rng,False,flight_weight=.035,coherent_shift=(first,last,0.))
        if i%2==0:
            refine_joint(self.udp,legs,self.deadline,
                         lambda s:self.submit(order,s,trial,"deadline bridge with wider epoch shift"),
                         0.,self.rng,False,coherent_shift=(first,last,-.28))

    def deadline_probe(self,i,trial):
        targets=(110.75,108.,105.,101.,109.7,107.,103.,101.)
        if self.args.repair_deadline:
            targets=(111.55,110.75,111.30,109.7,111.0,110.25,108.,101.)
        if self.args.label_deadline:
            targets=(111.65,111.55,111.25,110.75,111.45,110.5,109.75,101.)
        if self.args.compound:
            targets=(111.68,111.55,111.30,110.75,111.45,111.15,110.50,111.66)
        cap=targets[i%len(targets)]
        if cap>self.best_days:
            cap=math.floor(4*(self.best_days-.05))/4
        working=self.best_path
        paths=[working,self.archived_starts[(i//3)%min(5,len(self.archived_starts))][1]]
        if i%3!=0:
            source=set(zip(working,working[1:]))
            got=self.topology.propose(self.rng,min(self.deadline,time.monotonic()+4.5),source,
                                      (2,4,6)[i%3],.26,14)
            if got:
                paths.append(got[0])
        for route in paths:
            for phase in (i%4,(i+1)%4):
                self.seed_deadline(route,cap,phase)
        if self.args.repair_deadline:
            spans=((17,32),(30,48),(0,17),(24,39),(12,45),(41,48))
            first,last=spans[i%6]
            self.shifted_deadline(cap,first,last,i%2)
            self.shifted_deadline(cap,0,48,(i+1)%2)
        true={l for variants in self.anchors.values() for l in variants}
        synthetic=self.deadline_columns[cap]
        labelled=None
        if self.args.label_deadline:
            for route in paths:
                candidate=clock_labels(self.udp,route,true,synthetic,cap)
                if candidate and (labelled is None or
                                  (candidate[1],arrival(candidate[0][-1])) <
                                  (labelled[1],arrival(labelled[0][-1]))):
                    labelled=candidate
            if labelled and not labelled[1]:
                legs=labelled[0]
                self.submit((legs[0][0],*(l[1] for l in legs)),legs,trial,"all-real deadline clock labels")
            # Only exact or translated support of the compressed routes
            # is needed for this near-incumbent deadline repair.
            relevant={l[:2] for l in synthetic}
            actual={l for l in true if l[:2] in relevant}
        else:
            actual=true
        seconds=max(.05,min(self.args.mip_seconds,self.deadline-time.monotonic()))
        result,model=event_mip(self.udp,actual,cap,seconds,weight=.015,artificial=synthetic)
        if labelled and (not result or labelled[1]<len(model.get("artificial_legs",[]))):
            chain=labelled[0]
            result=((chain[0][0],*(l[1] for l in chain)),chain)
            model["artificial_legs"]=[l for l in chain if l in synthetic and l not in true]
            model["clock_label_fallback"]=True
        trial["deadline_model"]=model
        self.stats["deadline_models"]+=1
        if not result:
            return
        path,legs=result
        artificial=model.get("artificial_legs",[])
        old=self.conversion_counts.get(cap,49)
        if len(artificial)<old:
            self.conversion_counts[cap]=len(artificial)
            print(f"Deadline {cap:g} days | {len(artificial)} unresolved transfer events",flush=True)
        if not artificial:
            self.submit(path,legs,trial,"deadline-directed event MILP")
            refine_joint(self.udp,legs,self.deadline,
                         lambda s:self.submit(path,s,trial,"deadline-directed continuous refinement"),
                         0.,self.rng,False)
            return
        if self.args.repair_deadline:
            self.repair_runs(legs,artificial,trial,i)
            self.close_deadline_gaps(legs,artificial,trial,i)
            if self.args.compound and len(artificial)<=3:
                self.compound_bridge(path,legs,artificial,trial,i)
        grid=self.grid(.03125,(0.,.5,.25,.75)[(i+i//8)%4])
        for a,b,dep,tof,dv in artificial:
            key=(a,b,round(dep,3),round(tof,3),grid.phase)
            if key in self.deadline_seen:
                continue
            self.deadline_seen.add(key)
            extra=dv>self.udp.dv_threshold
            horizon=max(1.2,tof+.85) if extra else min(9.,tof+1.25)
            for offset in (0.,-.42,.31):
                self.sample(a,b,max(0.,dep+offset),grid,.65,horizon,extra)
                self.stats["deadline_event_queries"]+=1

    def grid(self,step,phase=0.,refine=False):
        key=step,phase,refine
        if key not in self.grids:
            self.grids[key]=Grid(self.module,step,phase*step,self.deadline,refine=refine,
                                 directional=self.args.directional)
            self.grids[key].horizon=self.args.coverage_day
        return self.grids[key]

    def submit(self,path,legs,trial,method):
        signature=tuple(l[:4] for l in legs)
        if signature in self.seen:
            return
        self.seen.add(signature)
        self.stats["esa_checks"]+=1
        result=validate(self.udp,path,legs)
        if not result or not result["valid"]:
            self.stats["esa_rejected"]+=1
            return
        self.stats["esa_feasible"]+=1
        if path not in self.route_set:
            self.route_set.add(path)
            self.stats["distinct_feasible_orders"]+=1
            self.recent_feasible=time.monotonic()
        if path!=self.original_path:
            self.stats["changed_order"]+=1
        days=result["fitness"][0]
        if path not in self.initial_orders and (self.best_new is None or days<self.best_new["days"]):
            self.best_new={"days":days,"path":path,"flight_days":flight_time(legs)}
        trial["best_accepted"]=min(days,trial.get("best_accepted",math.inf))
        self.topology.observe(legs)
        if days<=self.best_days+11.:
            for leg in legs:
                pair=leg[:2]
                old=self.anchors.get(pair,())
                if leg not in old:
                    self.anchors[pair]=(*old,leg)
        if days<self.best_days-1e-6:
            self.best_days=days
            self.best_path=path
            self.best=tuple(legs)
            self.last_gain=time.monotonic()
            self.stats["records"]+=1
            file=FILES/f"easy_events_{self.run_id}_{self.stats['records']:04d}.json"
            write_json(file,{"path":path,"transfers":legs,"arrival_day":days,"flight_days":flight_time(legs),
                             "validation":result,"attempt":{**trial,"method":method}})
            print(f"ESA improvement {days:.12f} days | {file.name}",flush=True)

    def sample(self,a,b,day,grid,wait,max_flight,allow):
        key=(a,b,round(day/grid.step),grid.step,grid.phase,allow,wait,max_flight)
        if key in self.sampled or day>=grid.horizon:
            return ()
        options=tuple(l for l in grid.edges(a,b,max(0.,day),wait,max_flight,allow) if l is not None)
        self.sampled[key]=options
        self.stats["sample_queries"]+=1
        self.stats["additional_events"]+=len(options)
        self.priced_events.update(options)
        for l in options:
            pair=l[:2]
            old=self.anchors.get(pair,())
            if l not in old:
                self.anchors[pair]=(*old,l)
        self.topology.observe(options)
        return options

    def collect_fragment(self,leg):
        a,b,dep,tof,dv=leg
        mark=(a,b,round(dep,4),round(tof,4),dv>self.udp.dv_threshold)
        if mark in self.fragment_ticks:
            return
        self.fragment_ticks.add(mark)
        self.fragments.add(leg)
        old=self.anchors.get((a,b),())
        if leg not in old:
            self.anchors[a,b]=(*old,leg)
            self.stats["rollout_columns"]+=1
        self.topology.observe((leg,))

    def close_deadline_gaps(self,legs,artificial,trial,i):
        missing=set(artificial)
        indices=[j for j,l in enumerate(legs) if l in missing]
        if not indices or len(indices)>14:
            return
        left=indices[0]
        groups=[]
        end=left+1
        for j in indices[1:]:
            if end!=j:
                groups.append((left,end))
                left=j
            end=j+1
        groups.append((left,end))
        notes=[]
        grid=self.grid(.015625,(0.,.5,.25)[i%3])
        for begin,stop in groups:
            start=arrival(legs[begin-1]) if begin else 0.
            limit=legs[stop][2] if stop<len(legs) else trial["deadline_model"]["sample_cap_days"]
            if stop-begin==1:
                a,b,_,_,dv=legs[begin]
                cutoff=self.udp.dv_threshold_exception+1e-6 if dv>self.udp.dv_threshold else self.udp.dv_threshold
                pad=.085 if i%3==2 else 0.
                found,cost=fit_transfer_window(self.udp,a,b,max(0.,start-pad),limit+pad,cutoff,
                                               self.deadline,self.rng)
                self.stats["constrained_bridge_solves"]+=1
                notes.append({"edge":[a,b],"earliest_departure":start,"latest_arrival":limit,
                              "best_sampled_dv":cost if math.isfinite(cost) else None,
                              "padding_days":pad})
                for leg in found:
                    self.collect_fragment(leg)
                    self.stats["constrained_bridge_events"]+=1
            else:
                current=start
                for a,b,_,_,_ in legs[begin:stop]:
                    options=self.sample(a,b,current,grid,.8,min(8.,limit-current+1.),True)
                    if not options:
                        break
                    current=min(map(arrival,options))
                    if current>limit+1.:
                        break
        trial["bridge_diagnostics"]=notes

    def compound_bridge(self,path,legs,artificial,trial,i):
        unresolved=set(artificial)
        where=[j for j,leg in enumerate(legs) if leg in unresolved]
        if not where or where[-1]-where[0]>8:
            return
        pool=[]
        for radius in (1,2,3):
            first=max(0,where[0]-radius)
            last=min(len(legs),where[-1]+radius+1)
            release=arrival(legs[first-1]) if first else 0.
            cap=legs[last][2] if last<len(legs) else trial["deadline_model"]["sample_cap_days"]
            marker=(path,first,last,round(release,6),round(cap,6),i%3)
            if marker in self.deadline_seen:
                continue
            self.deadline_seen.add(marker)
            block=legs[first:last]
            options,calls=optimize_block(self.udp,tuple(l[:2] for l in block),block,release,
                                         self.deadline,self.rng)
            self.stats["compound_objective_evaluations"]+=calls
            self.stats["compound_bridge_solves"]+=1
            for candidate in options:
                self.stats["compound_feasible_blocks"]+=1
                for l in candidate:
                    self.collect_fragment(l)
                finish=arrival(candidate[-1])
                pool.append((first,last,finish,cap))
                if finish<=cap+1e-6:
                    complete=legs[:first]+candidate+legs[last:]
                    self.submit(path,complete,trial,"physical multileg deadline bridge")
                    if arrival(complete[-1])<=self.best_days+.2:
                        refine_joint(self.udp,complete,self.deadline,
                                     lambda s:self.submit(path,s,trial,"multileg deadline bridge refinement"),
                                     0.,self.rng,False)
        trial["compound_bridge_matches"]=pool

    def schedule(self,path,trial,i,forced_exception=None):
        grids=(self.grid(.0625,(.5,.0,.25)[i%3]),self.grid(.03125,(.0,.5,.25)[i%3]))
        best=None
        for level,grid in enumerate(grids):
            if level and best is None and trial.get("maximum_depth",0)<30:
                break
            plans,depth=retime(grid,path,self.anchors,wait=6.,flight=10.,
                               observer=self.collect_fragment if self.args.rollout else None,
                               require_extra=forced_exception,extra_edge=forced_exception)
            trial["maximum_depth"]=max(depth,trial.get("maximum_depth",0))
            if not plans:
                if depth<len(path)-1:
                    self.topology.failed[path[depth],path[depth+1]]+=1
                continue
            for legs in plans:
                self.submit(path,legs,trial,"global topology with exact-timing repair")
                if best is None or arrival(legs[-1])<arrival(best[-1]):
                    best=legs
        if best and arrival(best[-1])<=self.best_days+1.2:
            fine=self.grid(.015625,(.5,0.,.25)[i%3],refine=i%3==2)
            precise,_=retime(fine,path,self.anchors,wait=6.,
                             observer=self.collect_fragment if self.args.rollout else None,
                             require_extra=forced_exception,extra_edge=forced_exception)
            for legs in precise:
                self.submit(path,legs,trial,"global topology with fine timing DP")
            if precise and arrival(precise[0][-1])<=self.best_days+.6:
                refine_joint(self.udp,precise[0],self.deadline,
                             lambda s:self.submit(path,s,trial,"global topology with joint refinement"),
                             0.,self.rng,False,flight_weight=.012)
        if self.args.rollout and i%4==2 and (best is None or arrival(best[-1])<=self.best_days+4.):
            target=min(self.args.coverage_day,self.best_days+(2. if best else 5.))
            backward=Grid(self.module,.03125,.015625,self.deadline,finish_day=target,
                          directional=self.args.directional)
            reversed_anchors={}
            for (a,b),legs in self.anchors.items():
                reversed_anchors[b,a]=[(b,a,target-arrival(l),l[3],l[4]) for l in legs if arrival(l)<=target]
            reverse_forced=forced_exception[::-1] if forced_exception else None
            plans,depth=retime(backward,path[::-1],reversed_anchors,wait=5.,
                               require_extra=reverse_forced,extra_edge=reverse_forced)
            trial["backward_depth"]=depth
            for reverse in plans:
                candidate=restore_suffix(self.udp,reverse,target)
                self.submit(path,candidate,trial,"global topology with backward timing")

    def price(self,i,path):
        # Price relevant events around distinct release and arrival profiles,
        # including times earlier than ever used by the archived tours.
        g=self.grid(.0625,(0.,.5,.25,.75)[i%4])
        if self.args.dual_price and self.pricing is not None and i%4!=0:
            self.price_duals(i,path,g)
            return
        n=len(path)
        loc={a:j for j,a in enumerate(path)}
        route_edges=list(zip(path,path[1:]))
        potential=[(int(a),int(b)) for a,b in zip(*np.nonzero(self.allowed[0]))]
        shells=self.udp.tomato_orbital_parameters[:,0]>1e7
        crossings=[(int(a),int(b)) for a,b in zip(*np.nonzero(self.allowed[1]))
                   if shells[a]!=shells[b]]
        subjects=(list(route_edges) if i%3==1 else potential if i%3==0 else
                  list(route_edges)+crossings)
        order=set(route_edges)
        self.rng.shuffle(subjects)
        current=self.best
        epochs={a:dep for a,_,dep,_,_ in current}
        trajectory={}
        cursor=0.
        for a,b in route_edges:
            trajectory[a]=cursor
            duration=min(self.topology.quality.get((a,b,0),math.inf),
                         self.topology.quality.get((a,b,1),math.inf))
            cursor+=duration+.15 if math.isfinite(duration) else 2.5
        if cursor>0:
            trajectory={a:t*self.best_days/cursor for a,t in trajectory.items()}
        queried=0
        for a,b in subjects:
            if queried>=self.args.samples:
                break
            g.check()
            if i%3==0 and (a,b) not in order and self.rng.random()<.25:
                continue
            nominal=epochs.get(a,self.best_days*loc[a]/n)
            if i%3==0:
                nominal=.65*nominal+.35*(self.best_days*loc[a]/n)
            elif self.args.rollout and a in trajectory:
                nominal=trajectory[a] if i%2 else .25*nominal+.75*trajectory[a]
            shift=(-3.25,-1.5,-.55,.0,.48,1.2,-5.)[(i+queried)%7]
            start=max(0.,nominal+shift)
            self.sample(a,b,start,g,1.7,8.5,(a,b) in order or i%4==0)
            if self.args.flight_first and (queried%3==0 or shells[a]!=shells[b]):
                shortest=self.grid(.046875)
                limit=self.udp.dv_threshold if shells[a]==shells[b] else self.udp.dv_threshold_exception
                maximum=7.5 if shells[a]==shells[b] else 2.3
                end=min(start+1.25,shortest.horizon-maximum-1e-4)
                if end>=start:
                    leg=shortest_band(shortest,a,b,start,end,maximum,limit)
                    self.stats["short_flight_queries"]+=1
                    if leg and leg not in self.anchors.get((a,b),()):
                        self.anchors[a,b]=(*self.anchors.get((a,b),()),leg)
                        self.topology.observe((leg,))
                        self.stats["short_flight_events"]+=1
                        self.priced_events.add(leg)
            queried+=1

    def price_duals(self,i,path,grid):
        nominal={a:dep for a,_,dep,_,_ in self.best}
        proposed={a:j*self.best_days/len(path) for j,a in enumerate(path)}
        candidates=[]
        for a,b,extra in self.topology.by_arc:
            source=nominal.get(a,proposed[a])
            prior=self.topology.quality.get((a,b,extra),.7 if extra else 2.9)
            for t in (source-1.2,source*.905,proposed[a],source*.96+.4):
                dep=max(0.,t)
                marker=(a,b,extra,round(dep,1),grid.phase)
                if marker in self.priced_signatures or dep+prior>self.best_days+3.:
                    continue
                estimated=self.pricing.reduced_cost(a,b,dep,prior,extra)
                estimated+=.035*self.pricing_failures[a,b,extra]+self.rng.uniform(0.,.08)
                candidates.append((estimated,a,b,extra,dep,marker))
        candidates.sort()
        for _,a,b,extra,dep,marker in candidates[:self.args.samples]:
            grid.check()
            self.priced_signatures.add(marker)
            maximum=8.5 if not extra else 4.5
            options=self.sample(a,b,dep,grid,.85,maximum,extra==1)
            if not options:
                self.pricing_failures[a,b,extra]+=1
            for l in options:
                cost=self.pricing.reduced_cost(a,b,l[2],l[3],int(l[4]>self.udp.dv_threshold))
                if cost < -1e-5:
                    self.stats["negative_reduced_cost_events"]+=1
            self.stats["dual_guided_queries"]+=1

    def events(self,trial,i):
        upper=min(self.args.coverage_day,self.best_days+(5. if i%5==2 else 1.35))
        available=set()
        for pair,legs in self.anchors.items():
            # Keep all early-to-late event choices, with only near-identical
            # epochs coalesced. Archive schedules remain usable as witnesses.
            compact={}
            for l in legs:
                finish=arrival(l)
                if finish<=upper+1e-6:
                    compact.setdefault((round(l[2],6),round(l[3],6),l[4]>self.udp.dv_threshold),l)
            available.update(compact.values())
        # Optional deadline probes must not be mistaken for a physical bound.
        if i%9==8:
            upper=min(upper,101.)
        seconds=max(.05,min(self.args.mip_seconds,self.deadline-time.monotonic()))
        forbid=set(zip(self.best_path,self.best_path[1:])) if i%6==4 else None
        flight_primary=i%5==3
        result,info=event_mip(self.udp,sorted(available),upper,seconds,(0.,.035,.1)[i%3],forbid,2,
                              flight_primary,self.args.dual_price)
        profile=info.pop("_pricing",None)
        if profile is not None:
            self.pricing=profile
        trial["event_model"]=info
        self.stats["event_models"]+=1
        if result:
            path,legs=result
            self.submit(path,legs,trial,"time-expanded global event MILP")
            if arrival(legs[-1])<=self.best_days+.65:
                refine_joint(self.udp,legs,self.deadline,
                             lambda s:self.submit(path,s,trial,"event MILP + continuous refinement"),
                             0.,self.rng,False)

    def exception_probe(self,i,trial):
        candidates=sorted((l for l in self.best if l[4]<=self.udp.dv_threshold),
                          key=lambda l:l[3],reverse=True)[:12]
        a,b,dep,tof,_=candidates[i%len(candidates)]
        edge=a,b
        trial.update(required_exception=edge,incumbent_flight_days=tof)
        grid=self.grid(.03125,(0.,.5,.25)[i%3])
        for shift in (-4.,-2.,-.7,0.,.4):
            self.sample(a,b,max(0.,dep+shift),grid,.9,3.,True)
        anchor=self.archived_starts[(i//12)%min(4,len(self.archived_starts))][1]
        reference=set(zip(anchor,anchor[1:]))
        result=self.topology.propose(self.rng,min(self.deadline,time.monotonic()+self.args.topology_seconds),
                                     reference,3,(.13,.44,.77)[i%3],18,forced_exception=edge)
        self.stats["forced_exception_topologies"]+=1
        if result:
            path,allocated,score=result
            trial.update(path=path,proposed_exceptions=allocated,surrogate_flight_days=score)
            self.stats["complete_proposed_orders"]+=1
            self.schedule(path,trial,i,edge)
            self.price(i,path)
        events={l for variants in self.anchors.values() for l in variants}
        plan,model=event_mip(self.udp,events,self.best_days+2.5,
                             max(.05,min(self.args.mip_seconds,self.deadline-time.monotonic())),
                             flight_primary=i%2==0,forced_exception=edge)
        trial["event_model"]=model
        if plan:
            path,legs=plan
            self.submit(path,legs,trial,"global reallocation of the movable exception")

    def audit(self,i,trial):
        # A final experiment over merged real event banks, with distinct
        # objectives and deadlines. A failed finite model is only a
        # statement about its event catalogue.
        spec=((self.best_days+.2,False),(101.,False),(self.best_days+.035,True),
              (self.best_days-.01,False),(self.best_days+2.,True))
        cap,flight_primary=spec[i%len(spec)]
        real={l for variants in self.anchors.values() for l in variants}
        if self.args.audit_corridor:
            bounds=[]
            near=sorted((min(rows,key=lambda r:r[0]) for rows in self.archive.values()),
                        key=lambda row:row[0])[:5]
            exact=set()
            for _,schedule,_ in near:
                depart={a:d for a,_,d,_,_ in schedule}
                reach={l[1]:arrival(l) for l in schedule}
                bounds.append((depart,reach))
                exact.update(schedule)
            window=self.args.audit_corridor
            real=exact|{l for l in real if any(
                abs(l[2]-dep.get(l[0],0.))<=window
                and abs(arrival(l)-reach.get(l[1],0.))<=window for dep,reach in bounds)}
            trial["clock_corridor_days"]=window
        model,report=event_mip(self.udp,real,cap,
                               max(.05,min(self.args.mip_seconds,self.deadline-time.monotonic())),
                               flight_primary=flight_primary)
        trial["event_model"]=report
        self.stats["merged_bank_models"]+=1
        if model:
            path,legs=model
            trial["selected_flight_days"]=flight_time(legs)
            self.submit(path,legs,trial,"global event MILP on merged event banks")
            if arrival(legs[-1])<=self.best_days+.75:
                refine_joint(self.udp,legs,self.deadline,
                             lambda s:self.submit(path,s,trial,"merged event bank continuous refinement"),
                             0.,self.rng,False)

    def run(self):
        print(f"ESA baseline {self.baseline['fitness']} | {self.run_id} | {self.loaded} archived plans",
              flush=True)
        print("Necessary geometry allows",int(np.count_nonzero(self.allowed[0])//2),
              "unordered normal pairs and",int(np.count_nonzero(self.allowed[1])//2),
              "unordered exception pairs",flush=True)
        starting_relaxation=self.independent_flight(self.original_path)
        stop="time budget"
        try:
            for i in range(self.args.attempts):
                self.grid(.0625).check()
                trial={"iteration":i}
                self.history.append(trial)
                if self.args.mode=="deadline":
                    self.deadline_probe(i,trial)
                elif self.args.mode=="exception":
                    self.exception_probe(i,trial)
                elif self.args.mode=="audit":
                    self.audit(i,trial)
                elif i==0 and self.args.mode!="topology":
                    self.events(trial,i)
                else:
                    path=self.best_path
                    if self.args.mode!="event":
                        anchor=self.best_path
                        if self.args.radius and i%4==3:
                            anchor=self.archived_starts[(i//4)%min(8,len(self.archived_starts))][1]
                        reference=set(zip(anchor,anchor[1:]))
                        forced=()
                        if self.args.mode=="sweep":
                            # Force changes to the global orientation or open
                            # endpoints, rather than disturbing a local block.
                            warm=(((self.udp.n_tomatoes,32),),((self.udp.n_tomatoes,11),),
                                  ((self.udp.n_tomatoes,14),),((25,2),),((44,40),),
                                  ((36,47),),((6,24),),((26,35),),((34,self.udp.n_tomatoes),))
                            forced=warm[i%len(warm)]
                        solution=self.topology.propose(
                            self.rng,min(self.deadline,time.monotonic()+self.args.topology_seconds),
                            reference,(0,3,6,8,3,11)[i%6] if not self.args.radius else (2,3,5,7,2,4)[i%6],
                            (.14,.30,.65)[i%3],self.args.radius or None,forced)
                        self.stats["topology_solves"]+=1
                        if solution:
                            path,allocated,score=solution
                            trial.update(path=path,surrogate_flight_days=score,proposed_exceptions=allocated,
                                         changed_edges=len(set(zip(self.best_path,self.best_path[1:]))-set(zip(path,path[1:]))))
                            trial["forced_global_arcs"]=forced
                            self.stats["complete_proposed_orders"]+=1
                            self.schedule(path,trial,i)
                    if self.args.mode!="topology":
                        self.price(i,path)
                        self.events(trial,i)
                if i%2==0:
                    print(f"{dict(self.stats)} | best {self.best_days:.12f} | "
                          f"{time.monotonic()-self.started:.1f}s",flush=True)
                if i>=5 and time.monotonic()-max(self.last_gain,self.recent_feasible)>self.args.stall:
                    stop="no newly feasible visit orders"
                    break
            else:
                stop="finite search sweep complete"
        except BudgetExpired:
            pass
        finally:
            file=FILES/f"easy_events_{self.run_id}_summary.json"
            write_json(file,{"seed":str(self.args.seed_file),"mode":self.args.mode,"baseline":self.baseline,
                             "best_official_days":self.best_days,"stop":stop,
                             "elapsed_seconds":time.monotonic()-self.started,"counts":dict(self.stats),
                             "normal_pairs":int(np.count_nonzero(self.allowed[0])//2),
                             "exception_pairs":int(np.count_nonzero(self.allowed[1])//2),
                             "best_new_order":self.best_new,
                             "starting_independent_sampled_flight":starting_relaxation,
                             "final_independent_sampled_flight":self.independent_flight(self.original_path),
                             "deadline_unresolved_events":self.conversion_counts,
                             "priced_events":sorted(self.priced_events|self.fragments),
                             "attempts":self.history})
            print(f"{stop} | {dict(self.stats)} | best {self.best_days:.12f} | {file.name}",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("hybrid","event","topology","sweep","deadline","exception","audit"))
    p.add_argument("--seed-file",type=Path,default=SEED)
    p.add_argument("--seconds",type=float,default=180)
    p.add_argument("--attempts",type=int,default=16)
    p.add_argument("--samples",type=int,default=65)
    p.add_argument("--mip-seconds",type=float,default=16)
    p.add_argument("--topology-seconds",type=float,default=9)
    p.add_argument("--coverage-day",type=float,default=138)
    p.add_argument("--stall",type=float,default=390)
    p.add_argument("--radius",type=int,default=0,
                   help="Maximum changed directed edges from a global reference route, 0 for unrestricted")
    p.add_argument("--flight-first",action="store_true")
    p.add_argument("--event-bank",type=Path,action="append",default=[])
    p.add_argument("--rollout",action="store_true",
                   help="Add reachable transfer events from partial timing repairs to the joint model")
    p.add_argument("--directional",action="store_true",
                   help="Prune Lambert directions using tangential and out-of-plane ΔV bounds")
    p.add_argument("--dual-price",action="store_true",
                   help="Guide transfer generation using reduced costs from the LP relaxation")
    p.add_argument("--repair-deadline",action="store_true",
                   help="Try joint continuous feasibility repair around unresolved deadline events")
    p.add_argument("--label-deadline",action="store_true",
                   help="Use fixed-order resource labels to initialize the deadline repair model")
    p.add_argument("--compound",action="store_true",
                   help="Repair up to three unresolved events with a consecutive multileg optimizer")
    p.add_argument("--audit-corridor",type=float,default=0.,
                   help="Restrict the audit to absolute epochs close to fast archived clocks")
    p.add_argument("--random-seed",type=int,default=712)
    args=p.parse_args()
    if not math.isfinite(args.seconds) or not 0<args.seconds<=3600:
        p.error("--seconds must be in (0, 3600]")
    Search(args).run()


if __name__=="__main__":
    main()

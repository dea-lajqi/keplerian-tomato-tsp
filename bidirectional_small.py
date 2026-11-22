"""Two-ended, exception-aware route construction for ESA Small."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace
from copy import copy
import time
from uuid import uuid4

import numpy as np

from continuous_windows_easy import FILES, SEED, WindowSearch
from elite_easy import anchor_bank, scan_archive
from focus_easy import refine_joint, retime
from global_dp_easy import Grid, evaluator
from global_event_easy import geometry
from scipy.optimize import minimize
from kttsp_esa_solver import validate
from pareto_easy import BudgetExpired, arrival, write_json


def bits(mask):
    while mask:
        bit = mask & -mask
        yield bit.bit_length()-1
        mask ^= bit


class Search:
    cost = WindowSearch.cost
    def __init__(self, args):
        self.args = args
        self.output = args.output
        self.output.mkdir(parents=True,exist_ok=True)
        self.started = time.monotonic()
        self.stop = self.started + args.seconds
        self.deadline = self.stop
        self.rng = random.Random(args.seed)
        self.module = evaluator()
        self.udp = self.module.udp
        self.normal = self.udp.dv_threshold
        self.cap = self.udp.n_dv_exception
        allowed, self.floors = geometry(self.udp,self.module.MU_MOON)
        self.n = self.udp.n_tomatoes
        self.all = (1<<self.n)-1
        self.adj = [[sum(1<<int(b) for b in np.flatnonzero(row)) for row in mat] for mat in allowed]
        self.archive, _ = scan_archive(self.udp)
        self.events = anchor_bank(self.archive)
        self.raw = json.loads(args.incumbent.read_text())
        self.reference = validate(self.udp,self.raw["path"],self.raw["transfers"])
        if not self.reference or not self.reference["valid"]:
            raise ValueError("ESA rejected the incumbent")
        self.best = self.reference["fitness"][0]
        self.best_candidate = None
        self.seen = set()
        self.orders = set()
        self.timing_signatures = set()
        self.novel = set()
        self.elites = {}
        self.stats = Counter()
        self.history = []
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid4().hex[:8]
        self.iteration = 0
        self.families = Counter()
        self.last_record = {}
        self.active = None
        self.saved_at = self.started
        self.members = sorted(min(rows,key=lambda row:row[0]) for rows in self.archive.values())
        relaxed = copy(self.udp)
        relaxed.dv_threshold += args.relax
        relaxed.dv_threshold_exception += args.relax
        self.soft_module = SimpleNamespace(udp=relaxed,MU_MOON=self.module.MU_MOON,pk=self.module.pk)
        if args.resume:
            prev = json.loads(args.resume.read_text())
            self.iteration = prev["next_iteration"]
            self.active = prev.get("active")
            self.orders.update(tuple(p) for p in prev.get("orders", ()))
            self.novel.update(tuple(p) for p in prev.get("novel_orders", ()))
            self.timing_signatures.update(tuple(map(tuple, t)) for t in prev.get("timing_signatures", ()))
            self.rng.setstate((prev["rng"][0],tuple(prev["rng"][1]),prev["rng"][2]))
            self.families.update({tuple(k):v for k,v in prev["families"]})
            record = prev.get("record")
            if record:
                verified = validate(self.udp,record["path"],record["transfers"])
                if verified and verified["valid"] and verified["fitness"][0]<self.best:
                    self.best = verified["fitness"][0]
                    self.last_record = record

    def check(self):
        if time.monotonic()>=self.stop:
            raise BudgetExpired

    @lru_cache(maxsize=250000)
    def residual(self, rem, a, b):
        """Necessary degrees and minimum exceptions in the unbuilt path."""
        nodes = rem | (1<<a) | (1<<b)
        deficit = 0
        undirected_cost = 0.
        for v in bits(nodes):
            endpoint = v in (a,b)
            possible = self.adj[1][v] & nodes
            normal = self.adj[0][v] & nodes
            if rem and endpoint:
                other = b if v==a else a
                possible &= ~(1<<other)
                normal &= ~(1<<other)
            degree = 1 if endpoint else 2
            if possible.bit_count()<degree:
                return self.cap+1,0.
            deficit += max(0,degree-normal.bit_count())
            prices = sorted(self.price(v,k) for k in bits(possible))
            undirected_cost += sum(prices[:degree])
        groups = 0
        remaining = nodes
        while remaining:
            reach = remaining & -remaining
            remaining ^= reach
            groups += 1
            while reach:
                v = (reach & -reach).bit_length()-1
                reach ^= 1<<v
                adj = remaining & self.adj[0][v]
                remaining ^= adj
                reach |= adj
        return max(groups-1,(deficit+1)//2),undirected_cost/2

    @lru_cache(maxsize=3000)
    def price(self,a,b):
        legs = (*self.events.get((a,b),()),*self.events.get((b,a),()))
        flights = [l[3] for l in legs if l[4]<=self.normal]
        if flights:
            return min(flights)
        return .45+.017*min(140.,float(self.floors[a,b]))

    def options(self,a,b,clock,reverse,extra,grid,goal,rich,closing=False):
        possible = []
        threshold = (self.udp.dv_threshold_exception+1e-6 if extra else self.normal)+self.args.relax
        for leg in self.events.get((a,b),()):
            if leg[4]<=threshold and (arrival(leg)<=clock+8e-7 if reverse else leg[2]>=clock-8e-7):
                possible.append(leg)
        best = min(possible,key=lambda l:-l[2] if reverse else arrival(l),default=None)
        delay = (clock-arrival(best) if reverse else best[2]-clock) if best else math.inf
        if closing or delay>(.42 if rich else .9):
            wait = (4.25 if closing else 2.25 if rich else 1.25)
            flight = 9.5 if closing else 7.5
            if reverse:
                release = max(0.,goal-clock)
                raw = grid.edges(b,a,release,wait,flight,extra)
                for l in raw:
                    if l is None:
                        continue
                    dep = goal-arrival(l)
                    if dep>=0. and dep+l[3]<=clock+8e-7:
                        dv = float(self.udp.compute_transfer(a,b,dep,l[3]))
                        if dv<=threshold:
                            possible.append((a,b,dep,l[3],dv))
            else:
                possible.extend(l for l in grid.edges(a,b,clock,wait,flight,extra)
                                if l is not None and l[4]<=threshold)
        choices = {}
        for l in possible:
            e = int(l[4]>self.normal+self.args.relax)
            value = -l[2] if reverse else arrival(l)
            if e not in choices or value<choices[e][0]:
                choices[e] = (value,l)
        return [row[1] for row in choices.values()]

    def boundary_score(self, label, forward, reverse, goal):
        """Rank sampled continuations at both clocks; missing samples are a soft penalty."""
        rem,a,b,spent,start,end = label[:6]
        if not rem:
            return 0.
        delays = []
        for backwards,vertex,clock in ((False,a,start),(True,b,end)):
            candidates = []
            for v in bits(self.adj[1][vertex] & rem):
                r = rem ^ (1<<v)
                need,_ = self.residual(r,a,v) if backwards else self.residual(r,v,b)
                mandatory = not bool(self.adj[0][vertex] & 1<<v)
                if spent+need+mandatory<=self.cap:
                    candidates.append((self.price(vertex,v),v,need))
            best = 5.
            for _,v,need in sorted(candidates)[:2]:
                x,y = (v,b) if backwards else (a,v)
                grid = reverse if backwards else forward
                for leg in self.options(x,y,clock,backwards,spent+need<self.cap,grid,goal,False):
                    if spent+need+int(leg[4]>self.normal+self.args.relax)>self.cap:
                        continue
                    delay = clock-leg[2] if backwards else arrival(leg)-clock
                    best = min(best,delay)
            delays.append(best)
        self.stats["boundary_lookaheads"] += 1
        return 1.55*max(delays)+.35*min(delays)

    def continuous_arc(self,a,b,release,due,extra,reverse):
        """Probe off-grid Lambert branches inside a nearly closed join gap."""
        min_flight = self.udp.min_tof
        if due-release<=min_flight:
            return []
        limit = (self.udp.dv_threshold_exception if extra else self.normal) - 2e-6
        span = min(9.5,due-release)
        if reverse:
            start = max(release,due-span-2.)
            stop = due
        else:
            start = release
            stop = min(due,release+span+2.)

        def decode(x):
            d = start+(stop-start-min_flight)*float(x[0])
            duration = min_flight+min(span-min_flight,stop-d-min_flight)*float(x[1])
            return d,duration

        def dv(x):
            return self.cost(a,b,*decode(x))

        self.stats["continuous_gap_probes"] += 1
        seeds = [(self.rng.random(),self.rng.random()) for _ in range(56)]
        seeds = sorted((dv(x),x) for x in seeds)[:3]
        result = []
        def objective(x):
            d,f = decode(x)
            return -d if reverse else d+f
        for value,x in seeds:
            if value > limit:
                fit = minimize(dv,x,bounds=((0.,1.),(0.,1.)),method="Powell",
                               options={"maxfev":65,"xtol":1e-4})
                if fit.fun<value:
                    value,x = float(fit.fun),fit.x
            if value<=limit:
                estimate = minimize(objective,x,method="SLSQP",bounds=((0.,1.),(0.,1.)),
                                    constraints=[{"type":"ineq","fun":lambda t:limit-dv(t)}],
                                    options={"ftol":1e-8,"maxiter":14,"eps":2e-7})
                v = dv(estimate.x)
                if v<=limit:
                    value,x = v,estimate.x
                d,f = decode(x)
                result.append((a,b,d,f,value))
        if result:
            self.stats["continuous_gap_hits"] += 1
        return result

    def submit(self, legs, trial):
        key = tuple(l[:4] for l in legs)
        if key in self.seen:
            return
        self.seen.add(key)
        path = (legs[0][0],*(l[1] for l in legs))
        self.stats["esa_checks"] += 1
        valid = validate(self.udp,path,legs)
        if not valid or not valid["valid"]:
            self.stats["esa_rejected"] += 1
            return
        self.stats["esa_feasible"] += 1
        self.orders.add(path)
        self.timing_signatures.add(tuple((l[0],l[1],round(l[2]*1000),round(l[3]*1000))
                                         for l in legs))
        if path not in self.archive:
            self.novel.add(path)
        value = valid["fitness"][0]
        trial["valid_days"] = min(value,trial.get("valid_days",math.inf))
        entry = {"path":path,"transfers":legs,"validation":valid,"goal":trial["goal"],
                 "seed":self.args.seed,"incumbent":str(self.args.incumbent)}
        if self.args.output!=FILES and path not in self.archive:
            if path not in self.elites or value<self.elites[path]["validation"]["fitness"][0]:
                self.elites[path] = entry
                if len(self.elites)>12:
                    del self.elites[max(self.elites,key=lambda p:self.elites[p]["validation"]["fitness"][0])]
        if self.best_candidate is None or value < self.best_candidate["validation"]["fitness"][0]:
            self.best_candidate = entry
        if value<self.best:
            self.best = value
            self.last_record = entry
            self.stats["records"] += 1
            write_json(self.output/f"easy_bidir_{self.run_id}_{self.stats['records']:04}.json",entry)
            print(f"ESA record {value:.14f}",flush=True)

    def free_suffixes(self, goal, count, limit, reverse):
        """Construct short suffixes with free order before joining any prefix."""
        endpoints = sorted(range(self.n),key=lambda j:(
            self.udp.tomato_orbital_parameters[j,0]>1e7,self.rng.random()))[:18]
        rows = [(1<<j,j,0,goal,()) for j in endpoints]
        for depth in range(count):
            candidates = {}
            for mask,b,used,clock,post in rows:
                for a in bits(self.adj[1][b] & ~mask):
                    mandatory = not bool(self.adj[0][b] & 1<<a)
                    if used+mandatory>min(2,self.cap):
                        continue
                    for leg in self.options(a,b,clock,True,used<2,reverse,goal,True):
                        used_total = used+int(leg[4]>self.normal+self.args.relax)
                        key = (mask|1<<a,a,used_total)
                        if key not in candidates or leg[2]>candidates[key][0]:
                            candidates[key] = (leg[2],(leg,*post))
            rows = [(*key,clock,post) for key,(clock,post) in sorted(
                candidates.items(), key=lambda item: (-item[1][0]+2.5*item[0][2],
                                                       self.rng.random()))[:limit]]
            if not rows:
                return []
        self.stats["freely_constructed_suffixes"] += len(rows)
        return [(-j-1,row[-1]) for j,row in enumerate(rows)]

    def archive_pairs(self, goal, cut, width, iteration, reverse):
        """Build independent left and right boundary profiles from different tours."""
        if not self.args.hybrid:
            return []
        members = self.members[:min(20,len(self.members))]
        # Translation is never assumed feasible. Suffixes must already fit
        # the target, or pass a fresh backwards Lambert propagation.
        candidates = []
        shift = ((iteration%5)-2)*2 if self.args.long_stems else (iteration%3)-1
        pcount = max(2,cut//2+shift)
        if self.args.free_suffix:
            pcount = cut-(3+iteration%3)
        scount = cut-pcount
        suffixes = self.free_suffixes(goal,scount,min(width,23),reverse) if self.args.free_suffix else []
        for si,(_,suffix,_) in enumerate(members):
            post = suffix[-scount:]
            if arrival(post[-1])<=goal+1e-7:
                suffixes.append((si,post))
            elif self.args.retime_suffix and si<12:
                sequence = (post[0][0],*(l[1] for l in post))
                rev_anchors = defaultdict(tuple)
                for leg in post:
                    a,b,d,f,dv = leg
                    dep = goal-d-f
                    if dep>=0.:
                        rev_anchors[b,a] += ((b,a,dep,f,dv),)
                schedules,_ = retime(reverse,sequence[::-1],rev_anchors,wait=2.5,
                                    flight=8.5,max_exceptions=min(3,self.cap))
                for candidate in schedules:
                    converted = []
                    for b,a,d,f,_ in reversed(candidate):
                        dep = goal-d-f
                        if dep<0.:
                            break
                        dv = float(self.udp.compute_transfer(a,b,dep,f))
                        if dv>self.udp.dv_threshold_exception+1e-6+self.args.relax:
                            break
                        converted.append((a,b,dep,f,dv))
                    else:
                        suffixes.append((si,tuple(converted)))
                        self.stats["new_suffix_profiles"] += 1
        for pi,(_,prefix,_) in enumerate(members):
            pre = prefix[:pcount]
            left = (pre[0][0],*(l[1] for l in pre))
            mask = sum(1<<x for x in left)
            for si,post in suffixes:
                right = (post[0][0],*(l[1] for l in post))
                smask = sum(1<<x for x in right)
                if smask&mask:
                    continue
                rem = self.all ^ (smask|mask)
                first,last = left[0],right[-1]
                start,end = arrival(pre[-1]),post[0][2]
                used = sum(l[4]>self.normal+self.args.relax for l in (*pre,*post))
                need,h = self.residual(rem,left[-1],right[0])
                if used+need>self.cap or end-start<(rem.bit_count()+1)*self.udp.min_tof:
                    continue
                score = h-(end-start)+.45*used + .5*(pi==si)+self.rng.random()*.9
                candidates.append((rem,left[-1],right[0],used,start,end,pre,post,first,last,score))
        seen = set()
        result = []
        for row in sorted(candidates,key=lambda x:x[-1]):
            key = row[:6]
            if key not in seen:
                result.append(row)
                seen.add(key)
            if len(result)>=width:
                break
        return result

    def restore(self,legs,trial):
        if not self.args.relax:
            self.submit(legs,trial)
            return
        threshold = self.udp.dv_threshold_exception+self.args.relax
        over = sum(l[4]>self.normal for l in legs)
        excess = max(l[4] for l in legs)
        if over<=self.cap and excess<=self.udp.dv_threshold_exception+1e-6:
            self.submit(legs,trial)
        elif excess<=threshold:
            self.stats["near_joins"] += 1
            # Assign scarce exceptions to unavoidable geometry edges first.
            forced = {i for i,l in enumerate(legs) if not self.adj[0][l[0]] & (1<<l[1])}
            remaining = sorted(set(range(len(legs)))-forced,key=lambda i:-legs[i][4])
            selected = forced | set(remaining[:max(0,self.cap-len(forced))])
            amended = tuple((*l[:4],max(l[4],self.normal+1e-5) if i in selected
                             else min(l[4],self.normal)) for i,l in enumerate(legs))
            refine_joint(self.udp,amended,self.stop,lambda s:self.submit(s,trial),
                         0.,self.rng,False)

    def sweep(self,goal,iteration):
        # Each label carries two disjoint paths. The remaining mask and the
        # two endpoint clocks are updated at the same time as the route.
        phase = (.00975,.03125,.018,.047)[iteration%4]
        step = .0625 if iteration%3 else .046875
        grid_module = self.soft_module if self.args.relax else self.module
        forward = Grid(grid_module,step,phase,self.stop,directional=True)
        reverse = Grid(grid_module,step,phase,self.stop,finish_day=goal,directional=True)
        forward.horizon = goal
        width = self.args.beam
        if self.active and self.active["iteration"]==iteration:
            if self.active.get("rng"):
                state = self.active["rng"]
                self.rng.setstate((state[0],tuple(state[1]),state[2]))
            first_depth = self.active["depth"]
            labels = [(*r[:6],tuple(map(tuple,r[6])),tuple(map(tuple,r[7])),*r[8:])
                      for r in self.active["labels"]]
            trial = self.active["trial"]
        else:
            low = [i for i,row in enumerate(self.adj[0]) if row.bit_count()==2 and
                   self.udp.tomato_orbital_parameters[i,0]<1e7]
            endsets = [(a,b) for a in low for b in low if a!=b]
            if iteration%3==2:
                high = [i for i in range(self.n) if i not in low]
                endsets += [(a,b) for a in low for b in high] + [(a,b) for b in low for a in high]
            endsets.sort(key=lambda ab:self.families[ab]+self.rng.random())
            if self.args.diverse:
                endsets = endsets[:self.args.beam]
            else:
                endsets = endsets[:max(8,self.args.beam//2)]
            labels = [(self.all ^ (1<<a) ^ (1<<b),a,b,0,0.,goal,(),(),a,b,0.)
                      for a,b in endsets]
            if self.args.free_suffix:
                cuts = (32,35,38,28) if self.args.long_stems else (15,22,27,18,25,12)
            else:
                cuts = (32,38,29,36,24,40) if self.args.long_stems else (12,18,24,8)
            cut = cuts[iteration%len(cuts)]
            seeded = self.archive_pairs(goal,cut,width,iteration,reverse)
            first_depth = cut if seeded else 0
            if seeded:
                labels = seeded
            trial = {"goal":goal,"iteration":iteration,"terminal_pairs":len(labels),"max_visits":first_depth+2,"fragment_seeded":bool(seeded)}
        self.history.append(trial)
        nearest = -math.inf
        last_live = labels
        for depth in range(first_depth,self.n-1):
            self.active = {"iteration":iteration,"depth":depth,"trial":trial,"labels":labels,
                           "rng":self.rng.getstate()}
            if time.monotonic()-self.saved_at>=45.:
                self.checkpoint()
                self.saved_at = time.monotonic()
            self.check()
            next_labels = defaultdict(list)
            self.stats["labels"] += len(labels)
            trial["beam_labels"] = trial.get("beam_labels",0)+len(labels)
            if not labels:
                break
            last_live = labels
            if depth==self.n-2:
                for _,a,b,used,start,end,pre,post,first,last,_ in labels:
                    need = not bool(self.adj[0][a] & 1<<b)
                    if not self.adj[1][a] & 1<<b or used+need>self.cap:
                        continue
                    self.stats["joins_attempted"] += 1
                    choices = self.options(a,b,start,False,used<self.cap,forward,goal,True,closing=self.args.close)
                    if self.args.continuous_gap and not any(arrival(l)<=end+8e-7 for l in choices):
                        choices.extend(self.continuous_arc(a,b,start,end,used<self.cap,False))
                    for leg in choices:
                        if arrival(leg)>end+8e-7 or used+int(leg[4]>self.normal+self.args.relax)>self.cap:
                            continue
                        schedule = (*pre,leg,*post)
                        self.stats["timed_joins"] += 1
                        self.restore(schedule,trial)
                        if not self.args.relax and (self.stats["timed_joins"]<=4 or (trial.get("valid_days",goal)<self.best+.02
                                                          and self.stats["timed_joins"]%9==0)):
                            refine_joint(self.udp,schedule,self.stop,lambda s:self.submit(s,trial),
                                         0., self.rng, False)
                break
            for rem,a,b,used,start,end,pre,post,first,last,_ in labels:
                full_remaining = rem.bit_count()+1
                # Occasionally build several consecutive legs on one side to
                # make crossings at different positions in the two halves.
                backwards = ((depth+iteration)%4 >= (2 if iteration%3 else 3))
                produced = False
                for backwards in ((backwards,not backwards) if self.args.hybrid else (backwards,)):
                    if produced:
                        break
                    vertex = b if backwards else a
                    moves = []
                    for v in bits(self.adj[1][vertex] & rem):
                        x,y = (v,b) if backwards else (a,v)
                        target_rem = rem^(1<<v)
                        v_a,v_b = (a,v) if backwards else (v,b)
                        need,h = self.residual(target_rem,v_a,v_b)
                        unavoidable = int(not self.adj[0][vertex] & 1<<v)
                        if used+need+unavoidable>self.cap:
                            self.stats["resource_prunes"] += 1
                            continue
                        moves.append((h+.35*unavoidable+self.price(x,y),x,y,v,need,h))
                    moves.sort()
                    closing = self.args.close and full_remaining<=10
                    for _,x,y,v,need,h in (moves if closing else moves[:self.args.fanout]):
                        self.stats["timing_queries"] += 1
                        grid = reverse if backwards else forward
                        extra = used+need<self.cap
                        options = self.options(x,y,end if backwards else start,backwards,extra,grid,goal,self.args.diverse,closing)
                        if self.args.continuous_gap and full_remaining<=5 and not options:
                            pad = (full_remaining-1)*self.udp.min_tof
                            options = self.continuous_arc(x,y,start+pad if backwards else start,
                                                          end if backwards else end-pad,extra,backwards)
                        for leg in options:
                            spent = used+int(leg[4]>self.normal+self.args.relax)
                            if spent+need>self.cap:
                                continue
                            new_start,new_end = (start,leg[2]) if backwards else (arrival(leg),end)
                            slack = new_end-new_start
                            if slack+1e-6< (full_remaining-1)*self.udp.min_tof:
                                self.stats["clock_prunes"] += 1
                                continue
                            nearest = max(nearest,slack-h)
                            aa,bb = (a,v) if backwards else (v,b)
                            key = (rem^(1<<v),aa,bb,spent,first,last)
                            # Pareto dominance requires both boundary clocks to
                            # improve. A larger interval alone is insufficient.
                            bucket = next_labels[key]
                            if any(p[4]<=new_start and p[5]>=new_end for p in bucket):
                                continue
                            bucket[:] = [p for p in bucket if not (new_start<=p[4] and new_end>=p[5])]
                            rank = h-slack+.55*spent + .3*self.families[first,last]
                            rank += .035*max(0.,leg[4]-(self.normal if leg[4]<=self.normal+self.args.relax else self.udp.dv_threshold_exception))
                            rank += self.rng.random()*(.8 if self.args.diverse else .2)
                            record = (*key[:4],new_start,new_end,pre if backwards else (*pre,leg),
                                      (leg,*post) if backwards else post,first,last,rank)
                            bucket.append(record)
                            produced = True
            ranked = sorted((r for bucket in next_labels.values() for r in bucket),key=lambda r:r[-1])
            if self.args.lookahead:
                window = ranked[:max(width*2,12)]
                ranked = sorted(((*row[:-1],row[-1]+self.boundary_score(row,forward,reverse,goal))
                                 for row in window),key=lambda row:row[-1])
            labels = []
            coverage = Counter()
            layer_width = width*3 if self.args.close and depth>=self.n-12 else width
            for r in ranked:
                family = (r[8],r[9])
                if coverage[family]>=max(3,layer_width//(5 if self.args.diverse else 3)):
                    continue
                labels.append(r)
                coverage[family]+=1
                if len(labels)>=layer_width:
                    break
            if labels:
                trial["max_visits"] = depth+3
        for r in last_live[:max(1,len(last_live)//4)]:
            self.families[r[8],r[9]] += 1
        trial["last_labels"] = len(labels)
        if math.isfinite(nearest):
            trial["largest_rank_slack"] = nearest
        print(json.dumps(trial),flush=True)

    def checkpoint(self):
        payload = {"next_iteration":self.iteration,"rng":self.rng.getstate(),
                   "elapsed_seconds":time.monotonic()-self.started,"baseline":self.reference,
                   "best_days":self.best,"best_generated":self.best_candidate,
                   "elite_candidates":list(self.elites.values()),
                   "record":self.last_record,"active":self.active,
                   "timing_signatures":sorted(self.timing_signatures),"orders":sorted(self.orders),"novel_orders":sorted(self.novel),
                   "families":[(k,v) for k,v in self.families.items()],
                   "counts":dict(self.stats),"attempts":self.history,
                   "configuration":{k:str(v) if isinstance(v,Path) else v for k,v in vars(self.args).items()}}
        write_json(self.output/f"easy_bidir_{self.run_id}_checkpoint.json",payload)

    def run(self):
        try:
            while self.iteration<self.args.iterations and self.best>101.:
                self.sweep(self.args.goals[self.iteration%len(self.args.goals)],self.iteration)
                self.iteration+=1
                self.active = None
                self.checkpoint()
        except (BudgetExpired, KeyboardInterrupt):
            pass
        finally:
            self.checkpoint()
            print(self.run_id,dict(self.stats),self.best,flush=True)


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=FILES)
    p.add_argument("--lookahead",action="store_true")
    p.add_argument("--incumbent",type=Path,default=SEED)
    p.add_argument("--resume",type=Path)
    p.add_argument("--seconds",type=float,default=690.)
    p.add_argument("--goals",type=float,nargs="+",default=[116.,112.,110.,108.,105.,103.,101.])
    p.add_argument("--iterations",type=int,default=42)
    p.add_argument("--beam",type=int,default=26)
    p.add_argument("--fanout",type=int,default=4)
    p.add_argument("--seed",type=int,default=18101)
    p.add_argument("--hybrid",action="store_true")
    p.add_argument("--retime-suffix",action="store_true")
    p.add_argument("--free-suffix",action="store_true")
    p.add_argument("--long-stems",action="store_true")
    p.add_argument("--continuous-gap",action="store_true")
    p.add_argument("--close",action="store_true")
    p.add_argument("--relax",type=float,default=0.)
    p.add_argument("--diverse",action="store_true")
    Search(p.parse_args()).run()

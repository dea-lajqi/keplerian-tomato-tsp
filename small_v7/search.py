"""Reserve the 24→6 exception and search upstream to measured departures."""
import argparse
from collections import defaultdict
from itertools import permutations
import json
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from small_v6.search import Search as SegmentSearch, GROUPS, FILES, GOALS, resource
from global_event_easy import Topology, geometry
from global_dp_easy import Grid
from pareto_easy import BudgetExpired, arrival
from small_v2 import atomic_json

SPECIAL=(24,6)


class Search(SegmentSearch):
    def __init__(self,args):
        super().__init__(args)
        self.contexts={}
        self.calendar=defaultdict(list)
        self.floors=geometry(self.udp,self.module.MU_MOON)[1]
        self.topology=Topology(self.udp,(self.normal,self.allowed),self.floors,self.load_archive())
        self.edge_scores={}
        self.known_relocated=[]
        self.late=[]
        for p in (FILES/"small_v6/reaudit/relocation").glob("*checkpoint.json"):
            for row in json.loads(p.read_text()).get("valid_candidates",()):
                self.known_relocated.append(tuple(row["path"]))
        if args.resume:
            raw=json.loads(args.resume.read_text())
            best=raw.get("best_generated")
            if best:
                self.known_relocated.insert(0,tuple(best["path"]))
            self.calendar.update({float(k):[tuple(map(tuple,row)) for row in v]
                                  for k,v in raw.get("calendar",[])})
            self.topology.forbidden.extend(set(map(tuple,s)) for s in raw.get("topology_excluded",[]))

    def load_archive(self):
        return {p:[(arrival(l[-1]),l,"verified library")] for p,l in self.parents.items()}

    def prepare_contexts(self):
        """Freeze only the post-jump backbone for each independently tested context."""
        candidates=[self.seed_path,*self.known_relocated,*self.parents]
        for path in candidates:
            if SPECIAL not in zip(path,path[1:]):
                continue
            j=path.index(24)
            if len(path)-j<6:
                continue
            group=tuple(sorted(path[-3:]))
            source=path[-4]
            tail_options=[l for (s,g),legs in self.tails.items()
                          if s==source and tuple(sorted(g))==group for l in legs]
            back=path[j+1:-3]
            if (not tail_options or any(not self.normal[a,b] for a,b in zip(back,back[1:]))
                    or len({v for v in back if v in self.group})):
                continue
            self.contexts.setdefault((back,group),(tail_options,path))
        self.stats["downstream_contexts"]=len(self.contexts)

    def build_calendar(self,stop):
        """Measure the crossing deadline, normal suffix, and exceptional jump backward."""
        opportunities=[]
        if self.args.atlas_jump:
            grid=self.grid(False,True)
            for k in range(72):
                if time.monotonic()>=stop:
                    break
                clock=78.+k*.25
                normal,exception=grid.edges(*SPECIAL,clock,.36,1.5,True)
                if exception and exception[3]>=self.udp.min_tof:
                    dv=self.cost(*exception[:4])
                    if self.limit<dv<=self.extra:
                        opportunities.append((*exception[:4],dv))
            opportunities=sorted(set(opportunities),key=lambda l:-l[2])
            self.stats["direct_fast_jumps"]=len(opportunities)
        for g in GOALS:
            rows=[]
            for (back,_),(tails,_) in self.contexts.items():
                # For fixed orbit and resource, a later departure dominates
                # for upstream reachability. Keep nearby different phases.
                choices=sorted((l for l in tails if arrival(l[-1])<=g),key=lambda l:-l[0][2])
                bins=set()
                for tail in choices:
                    if time.monotonic()>=stop:
                        return
                    tag=int(tail[0][2]*4)
                    if tag in bins:
                        continue
                    bins.add(tag)
                    if len(bins)>self.args.gates:
                        break
                    for fine in (False,True):
                        suffix,depth=self.chain(back,tail[0][2],0,True,fine)
                        self.stats["post_jump_depth_max"]=max(depth,self.stats["post_jump_depth_max"])
                        for _,(latest,post) in suffix.items():
                            direct=next((l for l in opportunities if arrival(l)<=latest+8e-7),None)
                            options=self.edge_options(*SPECIAL,latest,True,True,fine)
                            if direct:
                                options.append(direct)
                            for jump in options:
                                if jump[4]<=self.limit or jump[3]>1.5:
                                    continue
                                legs=(jump,*post,*tail)
                                rows.append(legs)
            keep={}
            for row in (*self.calendar[g],*rows):
                key=(tuple(l[:2] for l in row),int(row[0][2]*8))
                if key not in keep or row[0][2]>keep[key][0][2]:
                    keep[key]=row
            self.calendar[g]=sorted(keep.values(),key=lambda l:-l[0][2])
            self.stats[f"windows_{g}"]=len(keep)
            self.save()

    def grid(self,reverse,fine=False):
        grid=super().grid(reverse,fine)
        if not reverse and self.args.late_control and grid.horizon<125.:
            grid.horizon=125.
        return grid

    def edge_score(self,a,b):
        if (a,b) not in self.edge_scores:
            feasible=[l[3] for l in self.events.get((a,b),())
                      if (l[4]<=self.limit)==bool(self.normal[a,b])]
            if feasible:
                value=min(feasible)+.12
            elif not self.normal[a,b]:
                value=.8
            else:
                value=.75+.032*self.floors[a,b]
            self.edge_scores[a,b]=float(value)
        return self.edge_scores[a,b]

    def propose_prefixes(self,back,terminal_group):
        """Move the complete middle excursion to a nonnormal high-shell seam."""
        forbidden=set(back)|set(terminal_group)
        high_set=set(range(self.udp.n_tomatoes))-forbidden-set(self.group)
        sources=[self.seed_path,*self.known_relocated,*self.parents]
        orders=set()
        for p in sources:
            skeleton=tuple(v for v in p if v in high_set)
            if len(skeleton)!=len(high_set) or 24 not in skeleton:
                continue
            if skeleton[-1]==24:
                orders.add(skeleton)
            if skeleton[0]==24:
                orders.add(skeleton[::-1])
        for p in list(sorted(orders))[:20]:
            for a in range(0,len(p)-3):
                for b in range(a+2,len(p)-1):
                    q=(*p[:a],*p[a:b+1][::-1],*p[b+1:])
                    if sum(not self.normal[x,y] for x,y in zip(q,q[1:]))<=1:
                        orders.add(q)
        pool={}
        reference=set(zip(self.seed_path,self.seed_path[1:]))
        for shell in sorted(orders):
            bad=[j for j,(a,b) in enumerate(zip(shell,shell[1:])) if not self.normal[a,b]]
            if len(bad)>1:
                continue
            for first_group in GROUPS:
                if first_group==terminal_group:
                    continue
                mid=next(g for g in GROUPS if g!=first_group and g!=terminal_group)
                for first in permutations(first_group):
                    if not self.allowed[first[-1],shell[0]]:
                        continue
                    for i in bad or range(len(shell)-1):
                        for middle in permutations(mid):
                            if not self.allowed[shell[i],middle[0]] or not self.allowed[middle[-1],shell[i+1]]:
                                continue
                            path=(*first,*shell[:i+1],*middle,*shell[i+1:])
                            edges=tuple(zip(path,path[1:]))
                            diff=sum(edge not in reference for edge in edges)
                            rank=diff
                            if self.args.rank=="speed":
                                rank=sum(self.edge_score(*edge) for edge in edges)+.06*diff
                            pool[path]=(rank+self.rng.random()*2.5,i+4)
        ranked=sorted(pool,key=lambda p:pool[p][0])
        self.stats["special_reserved_structural_prefixes"]+=len(ranked)
        selected=ranked[:self.args.routes]
        if self.args.diverse and len(ranked)>self.args.routes:
            selected=selected[:self.args.routes//2]+self.rng.sample(
                ranked[self.args.routes:],min(self.args.routes//2,len(ranked)-self.args.routes))
        return [(p,pool[p][1]) for p in selected]

    def mixed_prefixes(self,tail,count):
        """Find alternative orders while forcing the reserved jump and post-jump segment."""
        tail_path=(24,*(l[1] for l in tail))
        edges=tuple(zip(tail_path,tail_path[1:]))
        n=self.udp.n_tomatoes
        ref=set(zip(self.seed_path,self.seed_path[1:]))
        for i in range(count):
            if time.monotonic()+1>=self.deadline:
                return
            self.stats["fixed_suffix_topology_attempts"]+=1
            # Force the terminal structure so the solver can spend the
            # other exceptions only where the budget permits.
            fixed=(*edges,(tail_path[-1],n))
            ans=self.topology.propose(self.rng,min(self.deadline,time.monotonic()+3.),
                    ref,4,1.4,forced=fixed,forced_exception=SPECIAL)
            if not ans or tuple(ans[0][-len(tail_path):])!=tail_path:
                continue
            route=ans[0][:-len(tail_path)+1]
            if sum(not self.normal[a,b] for a,b in zip(route,route[1:]))>3:
                continue
            m=next((i for i in range(3,len(route)) if route[i] in self.group),0)
            self.stats["fixed_suffix_topologies"]+=1
            yield route,m

    def probe(self,route,mid,tail,goal,trial):
        """Join a measured terminal gate with a three-exception prefix."""
        due=tail[0][2]
        schedules,depth=self.chain(route,0.,3)
        trial["forward_depth"]=depth
        if schedules:
            for used,(finish,pre) in schedules.items():
                gap=due-finish
                trial["jump_gap"]=max(gap,trial.get("jump_gap",-1e6))
                trial["prefix_exceptions"]=used
                if self.args.late_control and gap<0.:
                    key=(route,tuple(l[1] for l in tail))
                    self.late.append((finish,key,pre))
                if gap>=-8e-7:
                    self.submit((*pre,*tail),trial)
            if self.args.fine and -1.2<trial["jump_gap"]<0.:
                for _,(finish,pre) in self.chain(route,0.,3,False,True)[0].items():
                    gap=due-finish
                    trial["jump_gap"]=max(gap,trial["jump_gap"])
                    if gap>=-8e-7:
                        self.submit((*pre,*tail),trial)
        if mid and self.args.method=="meet":
            cut=min(len(route)-2,mid+3)
            before,after=route[:cut+1],route[cut:]
            used_before=sum(not self.normal[a,b] for a,b in zip(before,before[1:]))
            used_after=sum(not self.normal[a,b] for a,b in zip(after,after[1:]))
            left,_=self.chain(before,0.,3-used_after)
            right,depth=self.chain(after,due,3-used_before,True)
            trial["backward_depth"]=depth
            for le,(arr,pre) in left.items():
                for re,(latest,post) in right.items():
                    if le+re>3:
                        continue
                    gap=latest-arr
                    trial["middle_gap"]=max(gap,trial.get("middle_gap",-1e6))
                    if gap>=-8e-7:
                        self.submit((*pre,*post,*tail),trial)

    def retime_late_controls(self):
        seen=set()
        for done,key,prefix in sorted(self.late,key=lambda row:row[0]):
            if key in seen:
                continue
            seen.add(key)
            if len(seen)>12:
                break
            for jump in self.edge_options(*SPECIAL,done,False,True,True):
                if jump[4]<=self.limit or jump[3]>1.5:
                    continue
                target=(6,*key[1][1:])
                budget=self.udp.n_dv_exception-1-resource(prefix,self.limit)
                continuations,depth=self.chain(target,arrival(jump),budget,False,True)
                self.stats["late_controls_attempted"]+=1
                for _,(_,suffix) in continuations.items():
                    trial={"kind":"relaxed deadline control","goal":112.,"forward_depth":depth,
                           "jump_departure":jump[2],"jump_flight":jump[3]}
                    self.hist.append(trial)
                    self.submit((*prefix,jump,*suffix),trial)

    def save(self):
        atomic_json(self.out/f"v7_{self.job}_checkpoint.json",
            {"best_days":self.best,"best_generated":self.best_generated,"counts":dict(self.stats),
             "elapsed":time.monotonic()-self.started,"seed":self.args.seed,"history":self.hist,
             "orders":sorted(self.orders),"finish_bins":sorted(self.finishes),
             "attempted":sorted(self.attempted),"calendar":list(self.calendar.items()),
             "topology_excluded":[sorted(edges) for edges in self.topology.forbidden]})

    def run(self):
        try:
            self.prepare_contexts()
            if not self.calendar or self.args.refresh:
                self.build_calendar(self.started+.46*self.args.seconds)
            pools={}
            trials=defaultdict(list)
            for goal in GOALS:
                for tail in self.calendar[goal][:self.args.tail_count]:
                    back=tuple(l[0] for l in tail[1:-3])+(tail[-3][0],)
                    group=tuple(sorted(l[1] for l in tail[-3:]))
                    key=back,group
                    if key not in pools:
                        pools[key]=self.propose_prefixes(back,group)
                    for prefix,m in pools[key]:
                        trials[goal].append((prefix,m,tail))
                    if self.args.topology:
                        trials[goal].append(((),0,tail))
            if self.args.topology:
                for g,entries in trials.items():
                    trials[g]=sorted(entries,key=lambda row:(bool(row[0]),-row[2][0][2]))
            live_limit=self.started+self.args.seconds*(.8 if self.args.late_control else 1.)
            for i in range(max(map(len,trials.values()),default=0)):
                if time.monotonic()>=live_limit:
                    break
                for goal in GOALS:
                    self.check()
                    if i>=len(trials[goal]):
                        continue
                    route,mid,tail=trials[goal][i]
                    candidates=[(route,mid)] if route else self.mixed_prefixes(tail,self.args.topologies_per_gate)
                    for prefix,m in candidates:
                        key=(*prefix,goal,round(tail[0][2],6),tail[-1][1])
                        if key in self.attempted:
                            continue
                        self.attempted.add(key)
                        trial={"goal":goal,"gate":tail[0][2],"terminal_finish":arrival(tail[-1]),
                               "terminal_group":tuple(sorted(l[1] for l in tail[-3:])),
                               "mid_exit":prefix[m+2] if m and m+2<len(prefix) else None,
                               "jump_flight":tail[0][3]}
                        self.hist.append(trial)
                        self.probe(prefix,m,tail,goal,trial)
                    if len(self.hist)%9==0:
                        self.save()
                if self.best<=101.:
                    return
            if self.args.late_control:
                self.retime_late_controls()
        except (BudgetExpired,KeyboardInterrupt):
            pass
        finally:
            self.save()
            print(dict(self.stats),"trials",len(self.hist),"best",self.best,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--atlas",type=Path,default=FILES/"small_v4/diagnostic/window_e3e82fc984_checkpoint.json")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--resume",type=Path)
    p.add_argument("--seconds",type=float,default=110.)
    p.add_argument("--seed",type=int,default=87101)
    p.add_argument("--phase",type=float,default=.021)
    p.add_argument("--gates",type=int,default=2)
    p.add_argument("--rank",choices=("overlap","speed"),default="overlap")
    p.add_argument("--routes",type=int,default=24)
    p.add_argument("--diverse",action="store_true")
    p.add_argument("--topologies-per-gate",type=int,default=2)
    p.add_argument("--tail-count",type=int,default=5)
    p.add_argument("--method",choices=("forward","meet"),default="forward")
    p.add_argument("--refresh",action="store_true")
    p.add_argument("--atlas-jump",action="store_true")
    p.add_argument("--fine",action="store_true")
    p.add_argument("--late-control",action="store_true")
    p.add_argument("--topology",action="store_true")
    args=p.parse_args()
    args.rebuild=False
    args.subset=False
    search=Search(args)
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    search.run()

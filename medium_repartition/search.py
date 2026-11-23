"""Deadline-aware reassignment of tomatoes between the two Medium high passages."""
import argparse
from bisect import bisect_left, bisect_right
from collections import defaultdict
from hashlib import sha256
from itertools import pairwise
import json
import math
from pathlib import Path
import random
import time

from medium_coupled.checkpoint import atomic_json
from medium_coupled.endgame import WindowGrid
from medium_coupled.meet import SideGrid
from medium_coupled.nonlocal_routes import Surgery
from medium_coupled.run import Paused, solve
from medium_coupled.terminal_atlas import pareto
from medium_frontier.nodal import discover, orbit
from medium_repartition.caps import strict_schedule
from medium_overnight.validation import official_udp, solution_payload, to_solution, verify
from pareto_easy import BudgetExpired, arrival

ROOT = Path("results/esa")
PRIOR = ROOT / "medium_deadline180"


class Repartition(Surgery):
    def __init__(self, args):
        super().__init__(args)
        self.rng = random.Random(args.random_seed)
        self.bank = defaultdict(set)
        self.index = {}
        self.state.setdefault("proposals", [])
        self.state.setdefault("near", [])
        self.state.setdefault("complete", {})
        self.state.setdefault("stats_by_operator", {})
        self.archive = args.directory / "feasible"
        self.high = {i for i in range(181) if self.axis(i)>8e6}
        self.seed_pieces = self.split(self.path)
        self.members = set(self.seed_pieces[1])
        self.sources = [(self.seed_pieces, tuple(self.legs))]
        for f in (*args.events, args.directory / "events.json"):
            if f.is_file():
                for key, values in json.loads(f.read_text())["rows"].items():
                    self.bank[tuple(map(int,key.split(",")))].update(map(tuple,values))
        for i,rows in self.anchors.items():
            self.bank[i].update(rows)
        for f in (*args.archive_seeds, *sorted(self.archive.glob("*.json"))):
            data = json.loads(f.read_text())
            candidate=to_solution(data)
            verify(self.udp,candidate)
            pieces=self.split(candidate["path"])
            if pieces and all(pieces != entry[0] for entry in self.sources):
                self.sources.append((pieces,tuple(map(tuple,data["transfers"]))))
            for l in data["transfers"]:
                self.bank[tuple(l[:2])].add(tuple(l))
        self.base_events = {k:frozenset(v) for k,v in self.bank.items()}
        learned = args.directory / "events.json"
        if learned.is_file():
            for key, values in json.loads(learned.read_text())["rows"].items():
                edge=tuple(map(int,key.split(",")))
                self.base_events[edge] = self.base_events[edge] - set(map(tuple,values))
        self.nodes = {node:orbit(self.udp,node) for node in self.high}
        self.tail_options = {}
        terminal_files=[*args.terminals.glob("terminal_*.json")]
        for folder in args.terminal_extra:
            terminal_files.extend(folder.glob("terminal_*.json"))
        for f in terminal_files:
            info=json.loads(f.read_text())
            self.tail_options[info["deadline"]] = {
                int(node): (row["latest"],tuple(map(tuple,row["tail"])))
                for node,row in info["windows"].items()}
        incumbent_tail=tuple(self.legs[161:])
        self.tail_options[182.2542] = self.tail_options.get(182.2543,{}).copy()
        self.tail_options[183.1]=self.tail_options[182.2542].copy()
        if incumbent_tail:
            a=incumbent_tail[0][0]
            self.tail_options[183.1][a]=(incumbent_tail[0][2],incumbent_tail)
        self.grids = {}
        self.tested = {it["id"] for it in self.state["proposals"]}

    def split(self, path):
        pieces=[]
        last=None
        for x in path:
            high=x in self.high
            if high!=last:
                pieces.append([])
                last=high
            pieces[-1].append(x)
        return tuple(map(tuple,pieces)) if len(pieces)==5 and path[0] not in self.high else None

    def grids_at(self, due, step, reverse, stop):
        key=due if reverse else self.args.horizon,step,reverse
        if key not in self.grids:
            if len(self.grids)>=9:
                oldest=next(iter(self.grids))
                self.count.update(self.grids.pop(oldest).stats)
            cls=SideGrid if reverse else WindowGrid
            argv=(self.udp,0.,key[0],step,.011,stop)
            self.grids[key]=cls(*argv,True) if reverse else cls(*argv)
        grid=self.grids[key]
        grid.deadline=stop
        return grid

    def events(self, a,b, clock, extra, reverse, step, due, stop, mode):
        edge=a,b
        grid=self.grids_at(due,step,reverse,stop)
        options=set()
        if edge not in self.index:
            rows=[pareto(l for l in self.bank[edge] if int(l[-1]>self.normal)==i)
                  for i in (0,1)]
            self.index[edge]=[(tuple(l[2] for l in r),tuple(map(arrival,r)),r) for r in rows]
        for departures,arrivals,rows in self.index[edge][:1+extra]:
            j=bisect_right(arrivals,clock+1e-9)-1 if reverse else bisect_left(departures,clock-1e-9)
            if 0<=j<len(rows):
                options.add(rows[j])
        distance=min(((clock-l[2]) if reverse else arrival(l)-clock for l in options),
                     default=math.inf)
        if not options or distance>max(0.10, 3*step) or (mode and distance>.05):
            duration=grid.origin-clock if reverse else clock-grid.origin
            for l in grid.edges(b if reverse else a,a if reverse else b,duration,
                                6.5 if mode else 4.7,9.,extra):
                if l is None:
                    continue
                dep=(grid.origin-arrival(l)) if reverse else grid.origin+l[2]
                f=l[3]
                dv=float(self.udp.compute_transfer(a,b,dep,f))
                self.count["transfer_replays"]+=1
                if dv<=self.cap and dep>=0:
                    options.add((a,b,dep,f,dv))
            if not options and mode>=2 and a in self.nodes and b in self.nodes:
                left,right=((max(0.,clock-9.),clock) if reverse else (clock,min(due,clock+9.)))
                found,_,calls=discover(self.udp,self.nodes,a,b,left,right,
                       self.udp.dv_threshold_exception if extra else self.normal,
                       min(stop,time.monotonic()+1.1))
                self.count["nodal_samples"]+=calls
                options.update(found)
        novel=options-self.bank[edge]
        if novel:
            self.bank[edge].update(novel)
            self.index.pop(edge,None)
        return options

    def pass_edges(self, route, left, right, frontier, reverse, remaining, step, due, stop, mode):
        seq=list(range(left,right))
        if reverse:
            seq.reverse()
        furthest=None
        for j in seq:
            if time.monotonic()>=stop:
                raise BudgetExpired
            a,b=route[j:j+2]
            labels={}
            reserve=remaining[j]
            for spend,(clock,flight,chain) in frontier.items():
                if spend+reserve>5:
                    continue
                for leg in self.events(a,b,clock,spend+reserve<5,reverse,step,due,stop,mode):
                    cost=spend+int(leg[-1]>self.normal)
                    ready=leg[2] if reverse else arrival(leg)
                    if ((arrival(leg)>clock+1e-9 if reverse else leg[2]<clock-1e-9)
                            or cost+reserve>5 or arrival(leg)>due+1e-9):
                        continue
                    previous=labels.get(cost)
                    rank=(-ready if reverse else ready,flight+leg[3])
                    if previous is None or rank<(-previous[0] if reverse else previous[0],previous[1]):
                        labels[cost]=(ready,flight+leg[3],(leg,*chain) if reverse else (*chain,leg))
            if not labels:
                return {},furthest,[a,b]
            frontier=labels
            furthest=j
        return frontier,furthest,None

    def stagger(self,pieces,base,day,step,mode,stop,split_bias,repair=True):
        route=sum(pieces[:4],())
        first_group, one, middle, two, final=pieces
        count=len(route)
        if len(set(sum(pieces,())))!=181 or len(final)!=20 or any(
                node in self.high for node in (*first_group,*middle,*final)):
            raise ValueError("Partition broke complete, complementary coverage")
        if not self.all_allowed(route) or self.route_floor(route)>4:
            return {"status":"geometry"}
        before=[0]*count
        for j,(a,b) in enumerate(pairwise(route)):
            before[j+1]=before[j]+int(not self.normal_edges[a,b])
        after=[before[-1]-c+1 for c in before]
        target=self.tail_options.get(day,{})
        if not target or set(final)!=set(self.seed_pieces[-1]):
            return {"status":"no_terminal"}
        back={}
        gate=route[-1]
        latest_crossing=None
        for low,(clock,tail) in sorted(target.items(),key=lambda x:-x[1][0]):
            if tail[-1][2]+tail[-1][3]>day+1e-9 or not self.allowed[gate,low]:
                continue
            for l in self.events(gate,low,clock,True,True,step,day,stop,mode):
                if arrival(l)>clock+1e-9 or not self.normal<l[-1]<=self.cap:
                    continue
                if not back or l[2]>back[1][0]:
                    back[1]=(l[2],sum(x[3] for x in tail)+l[3],(l,*tail))
                    latest_crossing=l
        if not back:
            return {"status":"crossing"}
        match=0
        for old,edge in zip(base,pairwise(route)):
            if tuple(old[:2])!=edge:
                break
            match+=1
        fixed=max(0,match-(2 if mode else 0))
        split=min(count-2,max(fixed,len(first_group)+len(one)+len(middle)+split_bias))
        stock=tuple(base[:fixed])
        forward={sum(l[-1]>self.normal for l in stock):
                 (arrival(stock[-1]) if stock else 0.,sum(l[3] for l in stock),stock)}
        forward,depth,blocked=self.pass_edges(route,fixed,split,forward,False,
                                              after[1:],step,day,stop,mode)
        if not forward:
            return {"status":"forward","depth":depth if depth is not None else fixed,
                    "block":blocked,"split":split,"crossing":latest_crossing}
        back,depth,blocked=self.pass_edges(route,split,count-1,back,True,before,step,day,stop,mode)
        if not back:
            return {"status":"backward","depth":depth,"block":blocked,
                    "split":split,"crossing":latest_crossing}
        pairs=sorted((f[0]-b[0],i+j,f,b) for i,f in forward.items() for j,b in back.items() if i+j<=5)
        if not pairs:
            return {"status":"exceptions","split":split}
        gap,used,f,b=pairs[0]
        result={"status":"gap","gap":gap,"split":split,"tomato":route[split],
                "forward":f[0],"latest":b[0],"crossing":latest_crossing,
                "exceptions":used,"flight":f[1]+b[1],"size":[len(one),len(two)]}
        if gap<=1e-9:
            result["status"]="valid" if self.keep((*f[2],*b[2]),day,step,mode) else "rejected"
        elif gap<.28 and repair:
            solution=self.build((*f[2],*b[2]))
            name=sha256(bytes(solution["path"])).hexdigest()[:20]
            atomic_json(self.args.directory/"near"/f"{day}_{name}.json",
                     {"solution":solution,"chronology_gap":gap,"trial":["partition",split,gap]})
            try:
                solve(self.udp,solution,max(fixed,split-16),3.7,0.,min(stop,time.monotonic()+5.),
                    lambda q:self.keep_solution(q,day,step,mode),self.count)
            except Paused:
                pass
        return result

    def keep(self, legs, day,step,mode):
        return self.keep_solution(self.build(legs),day,step,mode)

    def keep_solution(self, sol, day,step,mode):
        self.count["official_checks"]+=1
        try:
            fit=verify(self.udp,sol)
        except ValueError:
            return False
        tightened=strict_schedule(self.udp,sol)
        if tightened is None:
            self.count["strict_rejections"]+=1
            # Keep a candidate for independent cap-tightening, not in the feasible archive.
            key=sha256(bytes(sol["path"])).hexdigest()
            if fit[0] < self.state["best"]-.0001:
                atomic_json(self.args.directory/"cap_repair"/f"{key}.json",
                         {"solution":sol,"chronology_gap":0.,"trial":[day,160,0.]})
            return False
        sol=tightened
        try:
            fit=verify(self.udp,sol)
        except ValueError:
            return False
        self.count["valid"]+=1
        key=sha256(bytes(sol["path"])).hexdigest()
        flight=sum(sol["tof"])
        previous=self.state["complete"].get(key)
        if previous is None or (fit[0],flight)<(previous["finish"],previous["flight"]):
            payload=solution_payload(sol,fit,udp=self.udp,
                provenance={"method":"partition_deadline","deadline":day,"step":step,"mode":mode,
                            "context":getattr(self,"current_proposal",None)})
            file=self.archive/f"{key}.json"
            atomic_json(file,{"solution":sol,**payload})
            pieces=self.split(sol["path"])
            moved=len(set(pieces[1])^self.members)
            self.state["complete"][key]={"finish":fit[0],"flight":flight,"changes":moved,
                                         "file":str(file)}
            if all(pieces!=p for p,_ in self.sources):
                self.sources.append((pieces,tuple(map(tuple,payload["transfers"]))))
            if fit[0]<self.state["best"]-.0001:
                fit=verify(official_udp(Path.cwd()),sol)
                file=self.args.directory/"records"/f"{fit[0]:.12f}_{key[:13]}.json"
                atomic_json(file,{"solution":sol,**payload,"validation":
                            {"fitness":fit,"valid":True,"source":"original ESA Medium fitness"}})
                self.state.update(best=fit[0],record=str(file.resolve()))
                print("RECORD",fit,file,flush=True)
            self.save()
        return True

    def places(self, segment, chunk):
        x,y=chunk[0],chunk[-1]
        if any(not self.allowed[a,b] for a,b in pairwise(chunk)):
            return ()
        opportunities=[]
        for j in range(len(segment)+1):
            a=segment[j-1] if j else None
            b=segment[j] if j<len(segment) else None
            edges=[p for p in ((a,x),(y,b)) if None not in p]
            if not all(self.allowed[i,k] for i,k in edges):
                continue
            floor=sum(not self.normal_edges[i,k] for i,k in edges)
            if floor>1:
                continue
            # Preserve multiple insertion points. Actual timing is evaluated
            # by the forward/backward route decoder.
            cost=sum(self.geometry_floor[i,k] for i,k in edges)
            if a is not None and b is not None:
                cost-=.55*self.geometry_floor[a,b]
            opportunities.append((floor,cost,j))
        return tuple(j for _,_,j in sorted(opportunities)[:8])

    def propose(self, pieces, operator, focus=None):
        low,first,mid,second,end=pieces
        arms=[first,second]
        if operator=="balance":
            # Move two orbitally coherent node sets in opposite directions.
            i=0
            x=self.rng.randrange(max(1,len(first)-24),len(first)-1)
            y=self.rng.randrange(0,min(25,len(second)-1))
            k=self.rng.choice((1,1,2,3))
            l=self.rng.choice((1,1,2,3))
            block,other=first[x:x+k],second[y:y+l]
            a,b=first[:x]+first[x+k:],second[:y]+second[y+l:]
            q,p=self.places(a,other),self.places(b,block)
            if not q or not p:
                return None
            x,y=self.rng.choice(q),self.rng.choice(p)
            arms=[a[:x]+other+a[x:],b[:y]+block+b[y:]]
        elif operator=="cluster":
            src=self.rng.randrange(2)
            src_arm,dst_arm=arms[src],arms[1-src]
            # Cluster moves preserve internal membership but change passage size.
            k=self.rng.choice((3,4,5,6,8))
            indexes=list(range(len(src_arm)-k+1))
            focus_range=indexes[-25:] if src==0 else indexes[:25]
            j=self.rng.choice(focus_range)
            cluster=src_arm[j:j+k]
            p=self.places(dst_arm,cluster)
            if not p:
                return None
            at=self.rng.choice(p)
            arms[src],arms[1-src]=src_arm[:j]+src_arm[j+k:],dst_arm[:at]+cluster+dst_arm[at:]
        elif operator=="move":
            src=self.rng.randrange(2)
            src_arm,dst_arm=arms[src],arms[1-src]
            a=self.rng.randrange(len(src_arm)) if self.rng.random()<.4 else (
                self.rng.randrange(len(src_arm)-20,len(src_arm)) if src==0 else self.rng.randrange(20))
            k=self.rng.choice((1,1,1,2,3))
            cluster=src_arm[a:a+k]
            positions=self.places(dst_arm,cluster)
            if not positions:
                return None
            b=self.rng.choice(positions)
            arms[src],arms[1-src]=src_arm[:a]+src_arm[a+k:],dst_arm[:b]+cluster+dst_arm[b:]
        elif operator=="endpoint":
            # Use the final terminal geometry only as a necessary screen, then
            # choose among timed alternatives during the bidirectional decode.
            n=len(second)
            options=[i for i in range(max(0,n-15),n-1) if
                     any(self.allowed[second[i],b] for b in end)]
            if not options:
                return None
            a=self.rng.choice(options)
            item=second[a]
            if self.rng.random()<.4:
                arms[1]=second[:a]+second[a:][::-1]
            else:
                arms[1]=second[:a]+second[a+1:]+(item,)
        elif operator=="seam":
            # Maintain the proposed partition while changing the adjacent
            # directed adjacencies that blocked its last decoding.
            source=self.rng.randrange(2)
            if focus:
                for i,arm in enumerate(arms):
                    if focus[0] in arm:
                        source=i
            arm=arms[source]
            center=arm.index(focus[0]) if focus and focus[0] in arm else (
                len(arm)-2 if source==0 else 1)
            lo,hi=max(0,center-3),min(len(arm),center+5)
            a,b=sorted(self.rng.sample(range(lo,hi),2))
            if self.rng.random()<.5:
                change=list(arm)
                change[a],change[b]=change[b],change[a]
                arms[source]=tuple(change)
            else:
                arms[source]=arm[:a]+arm[a+1:b+1]+arm[a:a+1]+arm[b+1:]
        return low,arms[0],mid,arms[1],end

    def exact_window(self, source, nodes, sink, release, due, budget, step, stop):
        indices = {node:1<<i for i,node in enumerate(nodes)}
        labels = {(0,source,0):(release,())}
        for depth in range(len(nodes)+1):
            update={}
            for (mask,a,used),(clock,order) in labels.items():
                for b in (nodes if depth<len(nodes) else (sink,)):
                    bit=indices.get(b,0)
                    if mask & bit or not self.allowed[a,b]:
                        continue
                    for leg in self.events(a,b,clock,used<budget,False,step,due,stop,0):
                        finish=arrival(leg)
                        count=used+int(leg[-1]>self.normal)
                        if leg[2]<clock-1e-9 or finish>due+1e-9 or count>budget:
                            continue
                        key=(mask|bit,b,count)
                        if key not in update or finish<update[key][0]:
                            update[key]=finish,(*order,b)
            labels=update
            self.count["exact_route_labels"]+=len(labels)
            if not labels:
                return None
        return min(labels.values())[1][:-1]

    def rearrange(self, pieces, base, day, step, stop):
        base_times = {l[1]:arrival(l) for l in base}
        old_edges = {tuple(l[:2]) for l in base}
        result = list(pieces)
        for g in (1,3):
            arm=pieces[g]
            changes=[i for i,(a,b) in enumerate(pairwise(arm)) if (a,b) not in old_edges]
            if not changes or time.monotonic()>=stop:
                continue
            j=self.rng.choice(changes)
            lo=max(0,j-3)
            hi=min(len(arm)-1,j+4)
            if hi-lo<3:
                continue
            source,sink=arm[lo],arm[hi]
            release=max(0.,base_times[source]-self.rng.choice((.10,.5,1.1)))
            due=min(day,base_times[sink]+2.3)
            if release>=due:
                continue
            try:
                order=self.exact_window(source,arm[lo+1:hi],sink,release,due,
                                        self.rng.randrange(2),step,
                                        min(stop,time.monotonic()+2.4))
            except BudgetExpired:
                continue
            if order:
                result[g]=arm[:lo+1]+order+arm[hi:]
        return tuple(result)

    def checkpoint(self):
        self.save()
        delta={f"{a},{b}":sorted(rows-self.base_events.get((a,b),set()))
               for (a,b),rows in self.bank.items() if rows-self.base_events.get((a,b),set())}
        atomic_json(self.args.directory/"events.json",{"rows":delta})

    def run(self):
        iterations=0
        warmups=[(p,l,"control") for p,l in self.sources[:4]]
        near=[]
        for item in self.state["near"]:
            near.append((item["gap"],tuple(map(tuple,item["pieces"])),item.get("block")))
        operators=self.args.operators
        while time.monotonic()<self.stop and iterations<self.args.attempts:
            iterations+=1
            day=self.args.days[len(self.state["proposals"])%len(self.args.days)]
            step=self.args.steps[len(self.state["proposals"])%len(self.args.steps)]
            mode=(len(self.state["proposals"])//7)%3
            if warmups:
                pieces,base,op=warmups.pop(0)
                day=183.1
            else:
                prior,base=self.rng.choice(self.sources[:32])
                op=self.rng.choice(operators)
                focus=None
                if near and self.rng.random()<.64:
                    _,prior,focus=self.rng.choice(near[:16])
                pieces=self.propose(prior,op,focus)
                if self.args.exact_windows and pieces is not None and op!="seam":
                    pieces=self.rearrange(pieces,base,day,step,time.monotonic()+4.8)
            if pieces is None or not self.all_allowed(sum(pieces[:4],())):
                continue
            digest=sha256(bytes(sum(pieces,()))).hexdigest()
            key=f"{digest}:{day}:{step}:{mode}"
            if key in self.tested:
                continue
            started=time.monotonic()
            self.current_proposal={"operator":op,"proposal":digest,
                                   "members_moved":len(set(pieces[1])^self.members)}
            try:
                result=self.stagger(pieces,base,day,step,mode,min(self.stop,started+self.args.trial_seconds),
                                    self.rng.choice((6,15,25,37)))
            except BudgetExpired:
                result={"status":"time_cap"}
            entry={"id":key,"phase":op,"day":day,"seconds":time.monotonic()-started,
                   "movement":len(set(pieces[1])^self.members),
                   "passage_sizes":[len(pieces[1]),len(pieces[3])],**result}
            self.state["proposals"].append(entry)
            self.tested.add(key)
            if result.get("gap",100.)<(6. if entry["movement"] else 1.):
                near.append((result["gap"] - min(entry["movement"],3)*.8,pieces,result.get("block")))
                near.sort(key=lambda x:x[0])
                near=near[:28]
                self.state["near"]=[{"gap":gap,"pieces":p,"block":edge} for gap,p,edge in near]
            elif result.get("depth") is not None and entry["movement"] and self.rng.random()<.20:
                near.append((3.9,pieces,result.get("block")))
                near=near[:28]
            if len(self.state["proposals"])%8==0 or result["status"] in ("valid","gap"):
                self.checkpoint()
            if result["status"] in ("valid","gap") or len(self.state["proposals"])%15==0:
                print(entry,"best",self.state["best"],flush=True)
        for grid in self.grids.values():
            self.count.update(grid.stats)
        self.state["wall_seconds"]=self.state.get("wall_seconds",0.)+time.monotonic()-self.started
        self.checkpoint()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed-record",type=Path,required=True)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--archive-seeds",type=Path,nargs="*",default=())
    p.add_argument("--events",type=Path,nargs="*",default=())
    p.add_argument("--anchors",type=Path,nargs="*",default=())
    p.add_argument("--terminals",type=Path,default=PRIOR/"bidirectional_dense")
    p.add_argument("--terminal-extra",type=Path,nargs="*",default=())
    p.add_argument("--seconds",type=float,default=420.)
    p.add_argument("--trial-seconds",type=float,default=12.)
    p.add_argument("--days",type=float,nargs="+",default=(182.2542,183.1,180.,182.2542,181.25,183.1))
    p.add_argument("--steps",type=float,nargs="+",default=(.047,.056,.063,.037))
    p.add_argument("--horizon",type=float,default=183.5)
    p.add_argument("--random-seed",type=int,default=717180)
    p.add_argument("--attempts",type=int,default=100000)
    p.add_argument("--exact-windows",action="store_true")
    p.add_argument("--operators",nargs="+",choices=("move","balance","cluster","seam","endpoint"),
                   default=("move","balance","cluster","seam"))
    p.set_defaults(mode="high_repartition",beam=None,goal=180.,phase=None,baseline_record=None)
    Repartition(p.parse_args()).run()

"""Jointly choose passage membership by matching complementary timed subset labels."""
import argparse
from collections import defaultdict
from itertools import pairwise
import json
from pathlib import Path
import time

from medium_repartition.search import PRIOR, Repartition
from pareto_easy import BudgetExpired, arrival


class SeamDP(Repartition):
    def layers(self, nodes, start, max_spend, step, day, stop, reverse):
        bits={n:1<<i for i,n in enumerate(nodes)}
        current={(0,a,e):label for (a,e),label in start.items()}
        result=dict(current)
        for _ in range(len(nodes)):
            nxt={}
            for (mask,a,used),(clock,flight,chain) in current.items():
                if time.monotonic()>=stop:
                    raise BudgetExpired
                for b,bit in bits.items():
                    if bit&mask:
                        continue
                    x,y=(b,a) if reverse else (a,b)
                    if not self.allowed[x,y] or (used>=max_spend and not self.normal_edges[x,y]):
                        continue
                    options=self.events(x,y,clock,used<max_spend,reverse,step,day,stop,0)
                    for leg in options:
                        spend=used+int(leg[-1]>self.normal)
                        ready=leg[2] if reverse else arrival(leg)
                        if (spend>max_spend or arrival(leg)>day+1e-9 or
                                (arrival(leg)>clock+1e-9 if reverse else leg[2]<clock-1e-9)):
                            continue
                        key=mask|bit,b,spend
                        value=ready,flight+leg[3],(leg,*chain) if reverse else (*chain,leg)
                        previous=nxt.get(key)
                        if previous is None or (-ready if reverse else ready,value[1])<(
                                -previous[0] if reverse else previous[0],previous[1]):
                            nxt[key]=value
            if not nxt:
                break
            result.update(nxt)
            current=nxt
        return result

    def core(self, right, day, step, stop, remaining):
        target=self.tail_options.get(day,{})
        back={}
        gate=right[-1]
        for low,(clock,tail) in target.items():
            if not self.allowed[gate,low] or arrival(tail[-1])>day+1e-9:
                continue
            for leg in self.events(gate,low,clock,True,True,step,day,stop,1):
                if self.normal<leg[-1]<=self.cap and arrival(leg)<=clock+1e-9:
                    if not back or leg[2]>back[1][0]:
                        back[1]=leg[2],sum(row[3] for row in tail)+leg[3],(leg,*tail)
        if not back:
            return {}
        profile,_,_=self.pass_edges(right,0,len(right)-1,back,True,
                                     [remaining]*(len(right)-1),step,day,stop,1)
        return profile

    def middle_orders(self, middle, n):
        yield middle
        # Preserve candidates with other entry / exit tomatoes.
        for j in range(1,n):
            child=middle[j:]+middle[:j]
            if self.all_allowed(child) and not self.route_floor(child):
                yield child
        if n>2 and not self.route_floor(middle[::-1]):
            yield middle[::-1]

    def join(self, pieces, base, k1,k2, day,step, max_tries, stop):
        low, one, middle, two, _=pieces
        left=one[:-k1]
        right=two[k2:]
        nodes=(*one[-k1:],*two[:k2])
        seedmask=(1<<k1)-1
        full=(1<<len(nodes))-1
        prefix=tuple(base[:len(low)+len(left)-1])
        first_spent=sum(l[-1]>self.normal for l in prefix)
        prefix_time=arrival(prefix[-1])
        core=self.core(right,day,step,stop,first_spent+2)
        if not core:
            return {"status":"right_core"}
        backward=self.layers(nodes,{(right[0],i):row for i,row in core.items()},
                             5-first_spent-2,step,day,stop,True)
        forward=self.layers(nodes,{(left[-1],first_spent):(prefix_time,
                sum(l[3] for l in prefix),prefix)},5-min(core)-2,step,day,stop,False)
        groups=defaultdict(list)
        for (mask,b,used),row in backward.items():
            groups[mask].append((b,used,row))
        comparisons=[]; exact_kept=0
        good=[]
        for middle_order in self.middle_orders(middle,self.args.middle_orders):
            incoming={}
            for (mask,a,spend),(clock,flight,chain) in forward.items():
                target=groups[full^mask]
                if not target or spend+2+min(t[1] for t in target)>5:
                    continue
                b=middle_order[0]
                if not self.allowed[a,b]:
                    continue
                for leg in self.events(a,b,clock,True,False,step,day,stop,1):
                    use=spend+1
                    if leg[-1]<=self.normal or leg[2]<clock-1e-9 or arrival(leg)>day:
                        continue
                    ready=arrival(leg)
                    key=mask,use
                    if key not in incoming or ready<incoming[key][0]:
                        incoming[key]=ready,flight+leg[3],(*chain,leg)
            rank=[]
            for (mask,used),(clock,flight,chain) in incoming.items():
                targets=[r for r in groups[full^mask] if used+r[1]+1<=5 and
                         self.allowed[middle_order[-1],r[0]]]
                if targets:
                    deadline=max(row[2][0] for row in targets)
                    rank.append((clock-deadline,mask==seedmask,mask,used,clock,flight,chain,targets))
            ordered=[]
            occupied=set()
            for row in sorted(rank):
                if row[2] in occupied:
                    continue
                lead=row[:4]
                occupied.add(row[2])
                ordered.append(row)
                # Retain both exception-spend classes on selected memberships.
                ordered.extend(r for r in rank if r[2]==lead[2] and r[3]!=lead[3])
                if len(occupied)>=max_tries:
                    break
            for _,_,mask,used,clock,flight,chain,targets in ordered:
                if time.monotonic()>=stop:
                    break
                reserve=min(row[1] for row in targets)+1
                mids,_,_=self.pass_edges(middle_order,0,len(middle_order)-1,
                   {used:(clock,flight,chain)},False,[reserve]*(len(middle_order)-1),
                   step,day,stop,1)
                for actual,(clock,flight,chain) in mids.items():
                    a=middle_order[-1]
                    for b,spend,(due,tail_flight,tail) in targets:
                        if actual+spend+1>5:
                            continue
                        for leg in self.events(a,b,clock,True,False,step,day,stop,1):
                            if leg[2]<clock-1e-9 or leg[-1]<=self.normal:
                                continue
                            gap=arrival(leg)-due
                            row={"gap":gap,"movement":(mask^seedmask).bit_count(),
                                 "left_moved":[n for i,n in enumerate(nodes) if
                                                      mask&(1<<i) and i>=k1],
                                 "right_moved":[n for i,n in enumerate(nodes) if
                                                      not mask&(1<<i) and i<k1],
                                 "edges":[chain[-len(middle_order)][:2],leg[:2]],
                                 "flight":flight+leg[3]+tail_flight,"spend":actual+spend+1}
                            comparisons.append(row)
                            if gap<=1e-9:
                                if self.keep((*chain,leg,*tail),day,step,3):
                                    exact_kept+=1
                                    good.append(row)
        comparisons.sort(key=lambda row:row["gap"])
        changed=[x for x in comparisons if x["movement"]]
        return {"status":"done","forward_states":len(forward),"backward_states":len(backward),
                "matched":len(comparisons),"timed_partitions":len({
                    (tuple(x["left_moved"]),tuple(x["right_moved"])) for x in comparisons}),
                "valid":exact_kept,"best_changed":changed[:8],"best_overall":comparisons[:5],
                "feasible":good[:30]}

    def normalized(self, piece, forbidden):
        bad = [i for i,(a,b) in enumerate(pairwise(piece)) if not self.normal_edges[a,b]]
        if len(bad)!=1:
            return []
        seam=bad[0]
        found=set()
        stop=time.monotonic()+2.5
        for radius in range(4,10):
            lo,hi=max(0,seam-radius),min(len(piece)-1,seam+1+radius)
            local=piece[lo:hi+1]
            rank=dict(zip(local,range(len(local))))
            sink=local[-1]
            def dfs(path,pending):
                if len(found)>=16 or time.monotonic()>=stop:
                    return
                a=path[-1]
                if not pending:
                    if self.normal_edges[a,sink] and (a,sink) not in forbidden:
                        found.add(piece[:lo]+path+(sink,)+piece[hi+1:])
                    return
                for b in sorted(pending,key=lambda node: abs(rank[node]-rank[a]-1)):
                    if self.normal_edges[a,b] and (a,b) not in forbidden:
                        dfs((*path,b),pending-{b})
            dfs(local[:1],set(local[1:-1]))
            if len(found)>=16 or time.monotonic()>=stop:
                break
        return sorted(found)

    def run(self):
        tested={tuple(t["key"]) for t in self.state["trials"]}
        if self.args.normalize_core:
            sources=[]
            for pieces,base in self.sources[:6]:
                variants=self.normalized(pieces[3],{(62,85),(85,62)})
                for new in variants:
                    sources.append(((*pieces[:3],new,pieces[4]),base))
            self.sources=sources
            self.state["normal_core_layouts"]=len(sources)
            self.save()
            if not sources:
                return
        for ordinal in range(self.args.tries):
            if time.monotonic()>=self.stop:
                break
            day=self.args.days[ordinal%len(self.args.days)]
            k1,k2=self.args.windows[(ordinal//len(self.args.days))%len(self.args.windows)]
            step=self.args.steps[(ordinal//(len(self.args.days)*len(self.args.windows)))%len(self.args.steps)]
            index=(ordinal//len(self.args.days))%len(self.sources)
            pieces, base=self.sources[index]
            key=(ordinal,index,k1,k2,day,step)
            if key in tested or day not in self.tail_options:
                continue
            start=time.monotonic()
            try:
                result=self.join(pieces,base,k1,k2,day,step,self.args.middle_limit,
                                 min(self.stop,start+self.args.trial_seconds))
            except BudgetExpired:
                result={"status":"time_cap"}
            self.state["trials"].append({"key":key,"seconds":time.monotonic()-start,**result})
            self.checkpoint()
            print(key, {k:v for k,v in result.items() if k not in ('best_overall','feasible')},
                  self.state["best"],flush=True)
        for grid in self.grids.values():
            self.count.update(grid.stats)
        self.state["wall_seconds"]=time.monotonic()-self.started
        self.checkpoint()


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed-record",type=Path,required=True)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--archive-seeds",type=Path,nargs="*",default=())
    p.add_argument("--events",type=Path,nargs="*",default=())
    p.add_argument("--anchors",type=Path,nargs="*",default=())
    p.add_argument("--terminals",type=Path,default=PRIOR/"bidirectional_dense")
    p.add_argument("--terminal-extra",type=Path,nargs="*",default=())
    p.add_argument("--seconds",type=float,default=400.)
    p.add_argument("--trial-seconds",type=float,default=32.)
    p.add_argument("--horizon",type=float,default=185.5)
    p.add_argument("--days",type=float,nargs="+",default=(183.1,182.2542,181.25,180.,184.5))
    p.add_argument("--steps",type=float,nargs="+",default=(.052,.039,.061))
    p.add_argument("--windows",type=int,nargs=2,action="append")
    p.add_argument("--middle-orders",type=int,default=2)
    p.add_argument("--middle-limit",type=int,default=20)
    p.add_argument("--tries",type=int,default=65)
    p.add_argument("--normalize-core",action="store_true")
    p.set_defaults(mode="complementary_partition",beam=None,goal=180.,phase=None,
                   baseline_record=None,random_seed=713750)
    return p


if __name__=="__main__":
    args=parser().parse_args()
    if not args.windows:
        args.windows=[(3,4),(4,4),(5,5),(6,5),(7,5)]
    SeamDP(args).run()

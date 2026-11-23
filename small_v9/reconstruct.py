"""Exception-budgeted middle-shell reconstruction from a verified Small prefix."""
import argparse
from functools import lru_cache
from pathlib import Path
import json
import math
import signal
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from small_v6.search import FILES, Search, arrival, resource
from kttsp_esa_solver import validate
from pareto_easy import BudgetExpired


class Reconstruction(Search):
    def __init__(self,args):
        super().__init__(args)
        self.at37=self.seed_path.index(37)
        self.middle=frozenset((4,11,17))
        self.stats["fixed_day_37_millidays"]=round(arrival(self.seed_legs[self.at37-1])*1000)
        for p in (FILES/"small_v7").glob("earliest_prefix_control/*checkpoint.json"):
            self.relocated=json.loads(p.read_text())
            break
        else:
            self.relocated={}

    def known_bridges(self,head,endpoint,nodes,start,budget):
        references=[self.seed_legs]
        record=self.relocated.get("best_generated")
        if record and validate(self.udp,record["path"],record["transfers"])["valid"]:
            references.append(tuple(map(tuple,record["transfers"])))
        for legs in references:
            path=(legs[0][0],*(l[1] for l in legs))
            a,b=path.index(head),path.index(endpoint)
            bridge=legs[a:b]
            if (bridge and set(path[a+1:b])==set(nodes) and bridge[0][2]>=start-8e-7
                    and resource(bridge,self.limit)<=budget):
                yield resource(bridge,self.limit),arrival(bridge[-1]),bridge

    def options_for(self,a,b,day,reverse,allow,fine):
        key=("transfer",a,b,round(day,9),reverse,allow,fine)
        if key not in self.cache:
            self.cache[key]=self.edge_options(a,b,day,reverse,allow,fine)
        return self.cache[key]

    def subset(self,start,end,nodes,clock,budget,reverse,fine=False):
        """All orders for small frontiers, bounded Pareto layer for larger ones."""
        if budget<0:
            return [],0
        if reverse:
            start,end=end,start
        full=(1<<len(nodes))-1
        bits={a:1<<i for i,a in enumerate(nodes)}
        low=sum(bits[a] for a in nodes if a in self.middle)
        @lru_cache(maxsize=180000)
        def needed(a,left):
            if not left:
                if self.allowed[a,end]:
                    return int(not self.normal[a,end])
                return budget+1
            best=budget+1
            for b in nodes:
                bit=bits[b]
                if not left&bit or not self.allowed[a,b]:
                    continue
                if a in self.middle and left&low and b not in self.middle:
                    continue
                value=int(not self.normal[a,b])+needed(b,left^bit)
                best=min(value,best)
                if best==0:
                    break
            return best

        states={(0,start,0):(clock,())}
        depth=0
        for layer in range(len(nodes)+1):
            nxt={}
            self.stats["backward_labels" if reverse else "forward_labels"]+=len(states)
            for (mask,a,spent),(epoch,seq) in states.items():
                left=full^mask
                choices=nodes if left else (end,)
                for b in choices:
                    bit=bits.get(b,0)
                    if bit and not left&bit:
                        continue
                    if a in self.middle and left&low and b not in self.middle:
                        continue
                    x,y=(b,a) if reverse else (a,b)
                    if not self.allowed[x,y]:
                        continue
                    remainder=left^bit
                    crossings=(0 if b==end else needed(b,remainder)) if self.args.connectivity else (
                        1 if b in self.middle else 2 if remainder&low else 0)
                    spare=budget-spent-crossings
                    if spare<0 or (spare==0 and not self.normal[x,y]):
                        continue
                    if remainder and not any(self.allowed[b,z] for z in nodes if remainder&bits[z]):
                        continue
                    for row in self.options_for(x,y,epoch,reverse,spare>0,fine):
                        used=spent+int(row[4]>self.limit)
                        if used+crossings>budget:
                            continue
                        t=row[2] if reverse else arrival(row)
                        key=mask|bit,b,used
                        value=(t,(row,*seq) if reverse else (*seq,row))
                        if key not in nxt or (t>nxt[key][0] if reverse else t<nxt[key][0]):
                            nxt[key]=value
            if not nxt:
                return [],depth
            depth+=1
            if len(nxt)>self.args.beam:
                direction=-1. if reverse else 1.
                states=dict(sorted(nxt.items(),key=lambda item:(
                    direction*item[1][0]+1.1*item[0][2],item[0]))[:self.args.beam])
                self.stats["beam_layers"]+=1
            else:
                states=nxt
        return [(e,t,seq) for (mask,b,e),(t,seq) in states.items() if mask==full and b==end],depth

    def terminal_plans(self,endpoint):
        idx=self.seed_path.index(endpoint)
        if self.args.mode=="reserve":
            calendars=self.relocated.get("calendar",[])
            for goal,rows in calendars:
                if goal not in (110.999,108.,105.,101.,112.):
                    continue
                # Compare only downstream tails visiting the complement set.
                expected=set(self.seed_path[self.seed_path.index(24)+1:])
                filtered=[tuple(map(tuple,r)) for r in rows if set(l[1] for l in r)==expected]
                filtered.sort(key=lambda r:-r[0][2])
                for tail in filtered[:self.args.gates]:
                    connector=self.seed_path[idx:self.seed_path.index(24)+1]
                    yield goal,connector,tail
            return
        for goal in (110.999,108.,105.,101.,112.):
            plans=[]
            for (a,order),tails in self.tails.items():
                if a!=14 or set(order)!={16,27,32}:
                    continue
                for tail in tails:
                    if arrival(tail[-1])<=goal:
                        plans.append(tail)
            plans.sort(key=lambda t:-t[0][2])
            for tail in plans[:self.args.gates]:
                yield goal,self.seed_path[idx:self.seed_path.index(14)+1],tail

    def full_candidate(self,pre,bridge,connector,tail,trial):
        self.submit((*pre,*bridge,*connector,*tail),trial)

    def forward_candidate(self,pre,bridge,day,endpoint,trial):
        used=resource((*pre,*bridge),self.limit)
        if self.args.mode=="reserve":
            route=self.seed_path[self.seed_path.index(endpoint):]
            j=route.index(24)
            at24,_=self.chain(route[:j+1],day,3-used,False,True)
            for e,(release,front) in at24.items():
                remaining=5-used-e-1
                if remaining<1:
                    continue
                for jump in self.edge_options(24,6,release,False,True,True):
                    if jump[4]<=self.limit or jump[3]>1.5:
                        continue
                    after,_=self.chain(route[j+1:],arrival(jump),remaining,False,True)
                    for _,(_,end) in after.items():
                        trial["fast_jump_day"]=jump[2]
                        self.submit((*pre,*bridge,*front,jump,*end),trial)
        else:
            route=self.seed_path[self.seed_path.index(endpoint):]
            suffix,_=self.chain(route,day,5-used,False,True)
            for _,(_,legs) in suffix.items():
                self.submit((*pre,*bridge,*legs),trial)

    def inspect(self,cut,join,fine):
        head=self.seed_path[cut]
        endpoint=self.seed_path[join]
        pre=self.seed_legs[:cut]
        start=arrival(pre[-1])
        nodes=self.seed_path[cut+1:join]
        available=(3 if self.args.mode=="reserve" else 4)-resource(pre,self.limit)
        trial={"cut":head,"join":endpoint,"fixed_prefix_release":start,"nodes":nodes,
               "bridge_budget":available,"fine":fine}
        self.hist.append(trial)
        forward,depth=self.subset(head,endpoint,nodes,start,available,False,fine)
        trial["forward_depth"]=depth
        if self.args.connectivity:
            known=list(self.known_bridges(head,endpoint,nodes,start,available))
            self.stats["measured_bridge_controls"]+=len(known)
            for used,day,bridge in known:
                same=[row for row in forward if row[0]==used]
                if not same or day<same[0][1]:
                    forward=[row for row in forward if row[0]!=used]+[(used,day,bridge)]
            if known:
                trial["best_measured_bridge"]=min(row[1] for row in known)
        if forward:
            trial["earliest_join"]=min(row[1] for row in forward)
            trial["gain_at_join_vs_incumbent"]=arrival(self.seed_legs[join-1])-trial["earliest_join"]
            first=min(forward,key=lambda row:row[1])
            trial["exceptions_used_in_bridge"]=first[0]
            trial["middle_entry"]=next(({"arc":l[:2],"departure":l[2],"arrival":arrival(l)} for l in first[2]
                                       if l[0] not in self.middle and l[1] in self.middle),None)
            trial["middle_exit"]=next(({"arc":l[:2],"departure":l[2],"arrival":arrival(l)} for l in first[2]
                                      if l[0] in self.middle and l[1] not in self.middle),None)
        for _,day,branch in sorted(forward,key=lambda r:r[1])[:2]:
            self.forward_candidate(pre,branch,day,endpoint,trial)
        self.save()
        for goal,connector,tail in self.terminal_plans(endpoint):
            self.check()
            post,rd=self.chain(connector,tail[0][2],0,True,fine)
            trial["connector_depth"]=max(trial.get("connector_depth",0),rd)
            for _,(latest,suffix) in post.items():
                comparison={"goal":goal,"latest_join":latest,"exit_slack":None,
                            "prefix_slack":None,"terminal_gate":tail[0][2]}
                trial.setdefault("gaps",[]).append(comparison)
                for _,t,branch in forward:
                    gap=latest-t
                    comparison["exit_slack"]=max(gap,comparison["exit_slack"] if comparison["exit_slack"] is not None else -math.inf)
                    if gap>=-8e-7:
                        self.full_candidate(pre,branch,suffix,tail,trial)
                if self.args.backward:
                    backwards,bd=self.subset(head,endpoint,nodes,latest,available,True,fine)
                    trial["backward_depth"]=max(trial.get("backward_depth",0),bd)
                    for _,leave,branch in backwards:
                        gap=leave-start
                        comparison["prefix_slack"]=max(gap,comparison["prefix_slack"] if comparison["prefix_slack"] is not None else -math.inf)
                        if gap>=-8e-7:
                            self.full_candidate(pre,branch,suffix,tail,trial)

    def run(self):
        try:
            points=[(37,19),(37,43),(37,0),(31,19),(31,43),(36,19)]
            if self.args.expanded:
                points=[(37,0),(31,43),(31,0),(36,19),(36,43),(37,39),*points]
            for fine in (False,True):
                for begin,end in points:
                    self.check()
                    key=(begin,end,int(fine))
                    if key in self.attempted:
                        continue
                    self.attempted.add(key)
                    if self.args.cases and len(self.hist)>=self.args.cases:
                        return
                    self.inspect(self.seed_path.index(begin),self.seed_path.index(end),fine)
                    self.save()
                    if self.best<=101:
                        return
        except (BudgetExpired,KeyboardInterrupt):
            pass
        finally:
            self.save()
            print(self.args.mode,dict(self.stats),"cases",len(self.hist),"orders",len(self.orders),
                  "best",self.best,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--mode",choices=("reserve","incumbent"),required=True)
    p.add_argument("--seconds",type=float,default=100.)
    p.add_argument("--gates",type=int,default=2)
    p.add_argument("--cases",type=int,default=0)
    p.add_argument("--beam",type=int,default=48)
    p.add_argument("--phase",type=float,default=.017)
    p.add_argument("--seed",type=int,default=89101)
    p.add_argument("--backward",action="store_true")
    p.add_argument("--connectivity",action="store_true")
    p.add_argument("--expanded",action="store_true")
    p.add_argument("--resume",type=Path)
    args=p.parse_args()
    args.atlas=FILES/"small_v4/diagnostic/window_e3e82fc984_checkpoint.json"
    args.rebuild=True
    args.subset=False
    search=Reconstruction(args)
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    search.run()

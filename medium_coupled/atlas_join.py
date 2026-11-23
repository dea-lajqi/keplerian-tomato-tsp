"""Join exact backward complement labels to a forward beam on the terminal atlas."""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import itertools
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from global_event_easy import fit_transfer_window
from medium_coupled.run import Paused, solve
from medium_coupled.terminal_atlas import Atlas, pareto
from pareto_easy import BudgetExpired, arrival


class AtlasJoin(Atlas):
    def backward(self,group,bit,events,day,depth,stop,stats):
        states={(bit[b],b):(day,()) for b in group}
        queries=0
        for k in range(1,depth):
            expanded={}
            for (mask,b),(latest,chain) in states.items():
                queries+=1
                if queries%512==0 and time.monotonic()>=stop:
                    raise BudgetExpired
                for a in group:
                    values=events.get((a,b))
                    if mask&bit[a] or not values:
                        continue
                    e=bisect_right(values[1],latest+1e-9)-1
                    if e<0:
                        continue
                    leg=values[2][e]
                    key=mask|bit[a],a
                    if key not in expanded or leg[2]>expanded[key][0]:
                        expanded[key]=(leg[2],(*chain,leg))
            states=expanded
            stats["backward_labels"]+=len(states)
            if not states:
                break
        result=defaultdict(list)
        for (mask,b),(t,chain) in states.items():
            if mask.bit_count()==depth:
                result[mask].append((b,t,chain))
        stats["reverse_complements"]=len(result)
        return result

    def forward(self,group,bit,events,starts,count,goal,width,mode,stop,stats):
        choices={(bit[b],b):(arrival(p[-1]),(),b) for b,p in starts.items()}
        template=tuple(self.path[-len(group):])
        queries=0
        for k in range(1,count):
            update={}
            for (mask,a),(clock,chain,source) in choices.items():
                queries+=1
                if queries%512==0 and time.monotonic()>=stop:
                    raise BudgetExpired
                for b in group:
                    edge=events.get((a,b))
                    if mask&bit[b] or not edge:
                        continue
                    e=bisect_left(edge[0],clock-1e-9)
                    if e==len(edge[0]):
                        continue
                    leg=edge[2][e]
                    t=arrival(leg)
                    if t>goal+1e-9:
                        continue
                    key=mask|bit[b],b
                    if key not in update or t<update[key][0]:
                        update[key]=(t,(*chain,leg),source)
            stats["forward_labels"]+=len(update)
            guide=sum(bit[a] for a in template[:k+1])
            ranked=sorted(update.items(),key=lambda x:x[1][0]+mode*.028*(x[0][0]^guide).bit_count())
            extra=sorted((x for x in update.items() if (x[0][0]^guide).bit_count()<=2),
                         key=lambda x:x[1][0])[:max(20,width//8)]
            choices={}
            occupancy=Counter()
            for key,row in itertools.chain(extra,ranked):
                if key in choices or occupancy[key[0]]>=12:
                    continue
                occupancy[key[0]]+=1
                choices[key]=row
                if len(choices)>=width:
                    break
            if not choices:
                break
        return choices

    def test(self,group,starts,events,goal,width,depth,mode,stop,stats,key):
        n=len(group)
        bit={a:1<<i for i,a in enumerate(group)}
        full=(1<<n)-1
        right=self.backward(group,bit,events,goal,depth,stop,stats)
        if not right:
            return {}
        left=self.forward(group,bit,events,starts,n-depth,goal,width,mode,stop,stats)
        best_gap=None
        near=[]
        feasible={}
        for (mask,a),(release,pre,source) in left.items():
            for b,due,post in right.get(full^mask,()):
                stats["complements"]+=1
                gap=due-release-self.udp.min_tof
                best_gap=gap if best_gap is None else max(gap,best_gap)
                edge=events.get((a,b))
                idx=bisect_left(edge[0],release-1e-9) if edge else -1
                if idx>=0 and idx<len(edge[0]) and edge[1][idx]<=due+1e-9:
                    leg=edge[2][idx]
                    chain=(*pre,leg,*post[::-1])
                    signature=source,tuple(l[1] for l in chain)
                    if signature not in feasible or arrival(chain[-1])<arrival(feasible[signature][-1]):
                        feasible[signature]=chain
                elif gap>-(.43 if self.args.relax else .055):
                    near.append((gap,a,b,release,due,source,pre,post))
        rng=random.Random(43971+depth+mode)
        count=0
        for (source,_),legs in sorted(feasible.items(),key=lambda kv:arrival(kv[1][-1]))[:18]:
            if time.monotonic()>=stop:
                break
            candidate=self.build((*starts[source],*legs))
            self.submit(candidate,"atlas_bidirectional_join",key)
            count+=1
            if arrival(legs[-1])<self.state["best"]+.075:
                try:
                    solve(self.udp,candidate,154,4.2,0.,min(stop,time.monotonic()+4.),
                            lambda q:self.submit(q,"atlas_bidirectional_polish",key),stats)
                except Paused:
                    pass
        seen=set()
        for _,a,b,release,due,source,pre,post in sorted(near,key=lambda row:-row[0])[:140]:
            if time.monotonic()>=stop or len(seen)>=14:
                break
            slot=(a,b,round(release,3),round(due,3))
            if slot in seen:
                continue
            seen.add(slot)
            if due-release<=self.udp.min_tof and not self.args.relax:
                continue
            pad=.37 if self.args.relax else 0.
            rows,_=fit_transfer_window(self.udp,a,b,release-pad,due+pad,self.normal,stop,rng)
            stats["continuous_bridges"]+=1
            ranked=sorted(rows,key=lambda l:max(0.,release-l[2])+max(0.,arrival(l)-due))
            for leg in ranked[:1]:
                candidate=self.build((*starts[source],*pre,leg,*post[::-1]))
                if leg[2]>=release-1e-9 and arrival(leg)<=due+1e-9:
                    self.submit(candidate,"atlas_continuous_bridge",key)
                if candidate["dep"][-1]+candidate["tof"][-1]<self.state["best"]+.10:
                    try:
                        result=solve(self.udp,candidate,150,4.2,0.,min(stop,time.monotonic()+4.),
                                    lambda q:self.submit(q,"atlas_continuous_polish",key),stats)
                        stats["bridge_polishes"]+=1
                    except Paused:
                        pass
        return {"best_slack_after_min_tof":best_gap,"graph_feasible":len(feasible),"submitted":count}

    def run(self):
        bank=json.loads(self.args.atlas.read_text())
        entries={}
        for key,rows in bank["rows"].items():
            a,b=map(int,key.split(","))
            data=pareto(rows)
            if data:
                entries[a,b]=tuple(l[2] for l in data),tuple(map(arrival,data)),data
        self.state["atlas"]={"events":sum(len(x[2]) for x in entries.values()),
                             "pair_meshes":len(bank["done"])}
        group=tuple(self.path[-20:])
        starts=self.starts(self.profiles()[:2],group,min(self.stop,time.monotonic()+20.))
        self.state["gate_options"]=len(starts)
        done={tuple(x["key"]) for x in self.state["trials"]}
        for day,(width,depth,mode) in itertools.product(self.args.deadlines,
                                      ((4000,6,0),(8000,6,1),(14000,5,2),(18000,7,1))):
            key=day,width,depth,mode
            if key in done or time.monotonic()>=self.stop:
                continue
            start=time.monotonic()
            stats=Counter()
            result={}
            try:
                result=self.test(group,starts,entries,day,width,depth,mode,
                         min(self.stop,start+self.args.trial_seconds),stats,key)
            except BudgetExpired:
                result["time_cap"]=True
            self.count.update(stats)
            self.state["trials"].append({"key":key,"seconds":time.monotonic()-start,
                                         "stats":dict(stats),**result})
            self.save()
            print(key,dict(stats),result,self.state["best"],flush=True)
        self.state["wall_seconds"]=self.state.get("wall_seconds",0.)+time.monotonic()-self.started
        self.save()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--seed-record",type=Path,required=True)
    p.add_argument("--baseline-record",type=Path)
    p.add_argument("--anchors",type=Path,nargs="*",default=())
    p.add_argument("--profiles",type=Path,nargs="+",required=True)
    p.add_argument("--atlas",type=Path,required=True)
    p.add_argument("--seconds",type=float,default=230.)
    p.add_argument("--trial-seconds",type=float,default=21.)
    p.add_argument("--deadlines",nargs="+",type=float,default=(192.,192.17,192.32,191.8))
    p.add_argument("--relax",action="store_true")
    p.set_defaults(mode="atlas_join",goal=192.36,beam=None,phase=None)
    AtlasJoin(p.parse_args()).run()

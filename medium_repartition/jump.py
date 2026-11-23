"""Discover earlier long-leg phase branches and repair their upstream arrival clocks."""
import argparse
from itertools import pairwise
import json
from pathlib import Path
import time

from medium_frontier.nodal import discover
from medium_frontier.repair import Repair
from medium_coupled.run import Paused
from medium_repartition.propagate import Propagate
from medium_repartition.search import PRIOR
from medium_repartition.caps import strict_schedule
from pareto_easy import BudgetExpired, arrival


class Jump(Propagate):
    def prefix(self, sol, end, spend):
        subset={name:seq[:end+int(name=="path")] for name,seq in sol.items()}
        fixed=strict_schedule(self.udp,subset)
        if fixed is None:
            return None
        rows=[]
        for (a,b),t,f in zip(pairwise(fixed["path"]),fixed["dep"],fixed["tof"]):
            if f<self.udp.min_tof-1e-9 or t<0 or (rows and t<arrival(rows[-1])-1e-9):
                return None
            dv=float(self.udp.compute_transfer(a,b,t,f))
            rows.append((a,b,t,f,dv))
        return tuple(rows) if sum(l[-1]>self.normal for l in rows)<=spend else None

    def run(self):
        out=self.args.directory/"gate_pool.json"
        self.profiles=json.loads(out.read_text()) if out.exists() else {}
        attempted={tuple(t["key"]) for t in self.state["trials"]}
        for index,(pieces,legs) in enumerate(self.sources[:self.args.limit]):
            path=self.build(legs)
            route=sum(pieces[:4],())
            reserve=[sum(not self.normal_edges[a,b] for a,b in pairwise(route[i:]))+1
                     for i in range(1,len(route))]
            modes={j:legs[j][3]+max(0.,legs[j][2]-arrival(legs[j-1])) for j in range(1,159)
                   if legs[j][-1]<=self.normal and legs[j][0] in self.high and legs[j][1] in self.high}
            pivots=sorted(modes,key=lambda j:-modes[j])[:18]
            flags=[l[-1]>self.normal for l in legs]
            for i in pivots:
                if time.monotonic()>=self.stop:
                    break
                a,b,d,f=legs[i][:4]
                ready=arrival(legs[i-1])
                started=time.monotonic()
                key=(index,i)
                if key in attempted:
                    continue
                e,v,calls=discover(self.udp,self.nodes,a,b,max(0.,ready-5.),d+f-.25,
                                   self.normal,min(self.stop,started+3.))
                self.count["nodal_evaluations"]+=calls
                nearest={}
                for leg in sorted(e,key=arrival):
                    bucket=round(leg[2]/.20)
                    if bucket not in nearest:
                        nearest[bucket]=leg
                viable=0
                results=[]
                for event in list(nearest.values())[:6]:
                    if time.monotonic()>=self.stop:
                        break
                    now=time.monotonic()
                    solved=None
                    if event[2] >= ready-1e-9:
                        solved=tuple(legs[:i])
                    for width in (8,16,26,38):
                        if solved is not None or time.monotonic()>=self.stop:
                            break
                        try:
                            p,info=Repair.fit(self,path,max(0,i-width),i,4.8,
                                            min(self.stop,time.monotonic()+2.1),
                                            seam=i-1,kick=-.13,deadline=event[2],
                                            hard_flags=flags,iterations=180)
                            if info["max_dv_excess_m_s"]<=0 and info["min_chronology_slack"]>=-1e-9:
                                label=self.prefix(p,i,sum(flags[:i]))
                                if label and arrival(label[-1])<=event[2]+1e-9:
                                    solved=label
                        except Paused:
                            continue
                    if solved is None:
                        continue
                    viable+=1
                    chain=(*solved,event)
                    spend=sum(l[-1]>self.normal for l in chain)
                    state={spend:(arrival(event),sum(l[3] for l in chain),chain)}
                    detail={"event":event,"saved_at_pivot":d+f-arrival(event),"ends":[]}
                    try:
                        end,depth,block=self.pass_edges(route,i+1,160,state,False,
                                 reserve,.044,self.args.horizon,min(self.stop,now+10.),1)
                        detail.update(depth=depth,blocked=block)
                        for _,(finish,flight,full) in end.items():
                            if self.profile(full,pieces[-1],(*key,event[:4])):
                                detail["ends"].append(finish)
                    except BudgetExpired:
                        detail["time_cap"]=True
                    results.append(detail)
                result={"key":key,"edge":[a,b],"phase_branches":len(nearest),
                        "sampled_min_dv":v,"prefix_recoveries":viable,"propagation":results,
                        "seconds":time.monotonic()-started}
                self.state["trials"].append(result)
                self.save()
                print(result,flush=True)
        for grid in self.grids.values():
            self.count.update(grid.stats)
        self.state["wall_seconds"]=time.monotonic()-self.started
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
    p.add_argument("--seconds",type=float,default=400.)
    p.add_argument("--horizon",type=float,default=179.7)
    p.add_argument("--limit",type=int,default=6)
    p.set_defaults(mode="earlier_phase_jump",beam=None,goal=180.,phase=None,
                   random_seed=713741,baseline_record=None)
    Jump(p.parse_args()).run()

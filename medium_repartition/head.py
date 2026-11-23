"""Exact timed reconstruction of the second high passage after an early polar exit."""
from itertools import permutations
from bisect import bisect_left
from pathlib import Path
import json
import time

from medium_coupled.checkpoint import atomic_json
from medium_coupled.exact_terminal import ExactTerminal
from medium_coupled.meet import SideGrid
from medium_coupled.terminal_atlas import Atlas, pareto
from medium_repartition.seam_dp import SeamDP, parser
from pareto_easy import BudgetExpired, arrival


class Head(SeamDP):
    def compile(self, group, release, horizon, stop):
        file=self.args.directory/"atlas.json"
        done=set(json.loads(file.read_text())["done"]) if file.exists() else set()
        try:
            for phase,step in ((.006,.083),(.022,.061)):
                grid=SideGrid(self.udp,release,horizon,step,phase,stop,False)
                links=list(permutations(group,2))
                links.sort(key=lambda e:(not bool(self.anchors[e]),self.geometry_floor[e]))
                try:
                    for a,b in links:
                        grid.check()
                        key=f"{phase}:{a},{b}"
                        if key in done or not self.normal_edges[a,b]:
                            continue
                        self.bank[a,b].update(Atlas.cells(self,grid,a,b))
                        self.index.pop((a,b),None)
                        done.add(key)
                        if len(done)%50==0:
                            self.checkpoint()
                            atomic_json(file,{"done":sorted(done)})
                finally:
                    self.count.update(grid.stats)
        except BudgetExpired:
            self.count["atlas_time_cap"]+=1
        atomic_json(file,{"done":sorted(done)})
        self.state["pair_meshes"]=len(done)
        self.checkpoint()

    def run(self):
        left, right=self.shells[2],self.shells[2]+1+self.args.width
        group=tuple(self.path[left+1:right])
        prefix=tuple(self.legs[:left])
        prefixes=[prefix]
        visited={self.path[0],*(l[1] for l in prefix)}
        for p in self.args.middle_profiles:
            for item in json.loads(p.read_text()).values():
                legs=tuple(map(tuple,item["legs"]))
                if item["spend"]==2 and len(legs)==left and {legs[0][0],*(l[1] for l in legs)}==visited:
                    prefixes.append(legs)
        prefixes.sort(key=lambda c:arrival(c[-1]))
        release=min(arrival(c[-1]) for c in prefixes)
        reference=arrival(self.legs[right-1])
        horizon=reference+.85
        start_nodes={}
        start=time.monotonic()
        for pre in prefixes:
            a=pre[-1][1]
            clock=arrival(pre[-1])
            for b in group:
                if not self.allowed[a,b]:
                    continue
                try:
                    for leg in self.events(a,b,clock,True,False,.037,horizon,
                                           min(self.stop,start+24.),2):
                        if leg[2]>=clock-1e-9 and self.normal<leg[-1]<=self.cap:
                            if b not in start_nodes or arrival(leg)<arrival(start_nodes[b][-1]):
                                start_nodes[b]=(*pre,leg)
                except BudgetExpired:
                    break
        self.state["starts"]={str(b):arrival(chain[-1]) for b,chain in start_nodes.items()}
        if self.args.compile_seconds > 0:
            self.compile(group,release,horizon,min(self.stop-65.,time.monotonic()+self.args.compile_seconds))
        for folder in self.args.terminal_extra:
            for file in folder.glob("terminal_*.json"):
                data=json.loads(file.read_text())
                self.tail_options[data["deadline"]]={
                    int(n):(x["latest"],tuple(map(tuple,x["tail"])))
                    for n,x in data["windows"].items()}
        rows={(a,b):pareto(l for l in self.bank[a,b] if l[-1]<=self.normal)
              for a,b in permutations(group,2)}
        deps={k:tuple(l[2] for l in v) for k,v in rows.items()}
        core=tuple(self.path[right:161])
        solver=ExactTerminal.__new__(ExactTerminal)
        solver.stop=self.stop
        for deadline in self.args.days:
            if deadline not in self.tail_options or time.monotonic()>=self.stop:
                continue
            started=time.monotonic()
            try:
                end=self.core(core,deadline,self.args.core_step,min(self.stop,started+26.),3)
                if not end:
                    result={"status":"core"}
                else:
                    bound=min(horizon,max(t[0] for t in end.values()))
                    metric,paths=solver.execute(group,{n:arrival(c[-1]) for n,c in start_nodes.items()},
                                               rows,bound,False)
                    best_gap=None
                    for depart,order in paths:
                        chain=list(start_nodes[order[0]])
                        for a,b in zip(order,order[1:]):
                            j=bisect_left(deps[a,b],arrival(chain[-1])-1e-9)
                            chain.append(rows[a,b][j])
                        for spend,(clock,flight,tail) in end.items():
                            for leg in self.events(order[-1],core[0],depart,spend<2,False,self.args.core_step,
                                                   deadline,min(self.stop,started+36.),2):
                                if leg[2]<depart-1e-9 or spend+3+int(leg[-1]>self.normal)>5:
                                    continue
                                gap=arrival(leg)-clock
                                if best_gap is None or gap<best_gap:
                                    best_gap=gap
                                if gap<=1e-9:
                                    self.keep((*chain,leg,*tail),deadline,self.args.core_step,4)
                    result={"status":"done","latest_core_ready":max(t[0] for t in end.values()),
                            "best_gap":best_gap,"full_orders":len(paths),**metric}
            except (BudgetExpired,TimeoutError):
                result={"status":"time_cap"}
            self.state["trials"].append({"key":deadline,"seconds":time.monotonic()-started,**result})
            self.checkpoint()
            print(deadline,result,self.state["best"],flush=True)
        self.state["wall_seconds"]=time.monotonic()-self.started
        self.checkpoint()


if __name__=="__main__":
    p=parser()
    p.add_argument("--width",type=int,default=17)
    p.add_argument("--middle-profiles",type=Path,nargs="*",default=())
    p.add_argument("--compile-seconds",type=float,default=225.)
    p.add_argument("--core-step",type=float,default=.046)
    p.set_defaults(max_flight=8.)
    Head(p.parse_args()).run()

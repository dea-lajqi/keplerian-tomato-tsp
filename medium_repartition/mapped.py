"""Compile orbital events once, then search bigger complementary passage sets."""
from bisect import bisect_left, bisect_right
from itertools import permutations
import json
import time

from medium_coupled.checkpoint import atomic_json
from medium_coupled.meet import SideGrid
from medium_coupled.terminal_atlas import Atlas, pareto
from medium_repartition.seam_dp import SeamDP, parser
from pareto_easy import BudgetExpired, arrival


class Mapped(SeamDP):
    archive_only=False

    def layers(self,*args,**kwargs):
        self.archive_only=True
        try:
            return super().layers(*args,**kwargs)
        finally:
            self.archive_only=False

    def events(self,a,b,clock,extra,reverse,step,due,stop,mode):
        if not self.archive_only:
            return super().events(a,b,clock,extra,reverse,step,due,stop,mode)
        edge=a,b
        if edge not in self.index:
            rows=[pareto(l for l in self.bank[edge] if int(l[-1]>self.normal)==i)
                  for i in (0,1)]
            self.index[edge]=[(tuple(l[2] for l in r),tuple(map(arrival,r)),r) for r in rows]
        result=[]
        for departures,arrivals,rows in self.index[edge][:1+extra]:
            j=bisect_right(arrivals,clock+1e-9)-1 if reverse else bisect_left(departures,clock-1e-9)
            if 0<=j<len(rows):
                result.append(rows[j])
        return result

    def compile(self, pieces, schedule):
        old=[*schedule]
        _,one,_,two,_=pieces
        n=12
        anchors=(one[-n-1],two[n])
        pool=(*one[-n:],*two[:n])
        old_by_pair={l[:2]:l for l in old}
        clock_at={l[1]:arrival(l) for l in old}
        before_port=clock_at[one[-1]]
        mid_end=clock_at[pieces[2][-1]]
        horizon=[(clock_at[anchors[0]]-.004,before_port+2.7),
                 (mid_end-2.6,clock_at[anchors[1]]+1.8)]
        data=self.args.directory/"compiled.json"
        tested=set(json.loads(data.read_text()).get("done",[])) if data.exists() else set()
        end=min(self.stop-90.,time.monotonic()+self.args.compile_seconds)
        try:
            for j,(floor,due) in enumerate(horizon):
                grid=SideGrid(self.udp,floor,due,self.args.mesh,.016,end,False)
                links=[(a,b) for a,b in permutations((*pool,anchors[j]),2)
                       if self.normal_edges[a,b]]
                links.sort(key=lambda key:(key not in old_by_pair,self.geometry_floor[key]))
                try:
                    for a,b in links:
                        grid.check()
                        k=f"{j}:{a},{b}"
                        if k in tested:
                            continue
                        rows=Atlas.cells(self,grid,a,b)
                        self.bank[a,b].update(rows)
                        self.index.pop((a,b),None)
                        self.count["compiled_rows"]+=len(rows)
                        tested.add(k)
                        if len(tested)%55==0:
                            self.checkpoint()
                            atomic_json(data,{"done":sorted(tested)})
                finally:
                    self.count.update(grid.stats)
        except BudgetExpired:
            self.count["compile_time_cap"]+=1
        self.state["completed_pair_meshes"]=len(tested)
        self.checkpoint()
        atomic_json(data,{"done":sorted(tested)})

    def run(self):
        self.compile(*self.sources[0])
        super().run()


if __name__=="__main__":
    p=parser()
    p.add_argument("--compile-seconds",type=float,default=180.)
    p.add_argument("--mesh",type=float,default=.083)
    p.set_defaults(max_flight=8.)
    args=p.parse_args()
    if not args.windows:
        args.windows=[(6,6),(7,7),(8,7),(9,7)]
    Mapped(args).run()

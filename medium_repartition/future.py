"""Build later terminal phases to admit diverse fully feasible search seeds."""
import argparse
from collections import defaultdict
from itertools import permutations
import json
from pathlib import Path
import time

from medium_coupled.checkpoint import atomic_json
from medium_coupled.exact_terminal import ExactTerminal
from medium_coupled.terminal_atlas import Atlas, pareto


class Future(Atlas):
    def run(self):
        group=tuple(self.path[-20:])
        stop=min(self.stop-35.,time.monotonic()+self.args.atlas_seconds)
        bank,count=({},0) if self.args.skip_build else self.build_atlas(group,self.args.floor,stop)
        pooled=defaultdict(list)
        for key,rows in bank.items():
            pooled[key].extend(rows)
        for file in self.args.shared:
            for key, rows in json.loads(file.read_text())["rows"].items():
                pooled[tuple(map(int,key.split(",")))].extend(rows)
        for leg in self.legs[-19:]:
            if leg[-1]<=self.normal:
                pooled[leg[:2]].append(leg)
        rows={k:pareto(l for l in pooled[k] if l[-1]<=self.normal) for k in permutations(group,2)}
        solver=ExactTerminal.__new__(ExactTerminal)
        solver.stop=self.stop
        for day in self.args.days:
            if time.monotonic()>=self.stop:
                break
            try:
                info, labels=solver.execute(group,dict.fromkeys(group,-day),rows,0.,True)
            except TimeoutError:
                break
            windows={str(path[-1]):{"latest":-t,"tail":solver.reconstruct_reverse(path,rows,day)}
                     for t,path in labels}
            self.state["trials"].append({"key":day,"depth":info["depth"],"windows":len(windows)})
            atomic_json(self.args.directory/f"terminal_{day}.json",
                        {"deadline":day,"floor":self.args.floor,"windows":windows,**info})
            print(day,len(windows),flush=True)
        self.state["atlas"]={"pair_meshes":count,"events":sum(map(len,rows.values()))}
        self.state["wall_seconds"]=time.monotonic()-self.started
        self.save()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed-record",type=Path,required=True)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--shared",type=Path,nargs="*",default=())
    p.add_argument("--seconds",type=float,default=260.)
    p.add_argument("--atlas-seconds",type=float,default=205.)
    p.add_argument("--floor",type=float,default=181.)
    p.add_argument("--goal",type=float,default=185.22)
    p.add_argument("--skip-build",action="store_true")
    p.add_argument("--days",type=float,nargs="+",default=(183.5,184.5,185.2))
    p.set_defaults(mode="later_terminal_atlas",max_flight=1.5,seed_atlas=None,
                   phase=None,beam=None,anchors=(),baseline_record=None,
                   meshes=((.0234375,.007),(.0185546875,.0025)))
    Future(p.parse_args()).run()

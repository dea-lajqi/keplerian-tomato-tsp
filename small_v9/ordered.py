"""Compare complete geometry-admissible orders without a subset beam."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from small_v9.reconstruct import Reconstruction, FILES, arrival, resource, validate
from pareto_easy import BudgetExpired


class Ordered(Reconstruction):
    def grid(self,reverse,fine=False):
        grid=super().grid(reverse,fine)
        if not reverse:
            grid.horizon=125.
        return grid

    def run(self):
        budget=2
        cat=json.loads((FILES/"small_v9/geometry_audit.json").read_text())[str(budget)]
        before=self.seed_legs[:self.at37]
        start=arrival(before[-1])
        high=self.seed_path.index(19)
        nodes=self.seed_path[self.at37+1:high]
        for row in [self.relocated.get("best_generated")]:
            if row and validate(self.udp,row["path"],row["transfers"])["valid"]:
                for l in row["transfers"]:
                    leg=tuple(l)
                    self.events[leg[:2]]=(*self.events.get(leg[:2],()),leg)
        reps=list(self.known_bridges(37,19,nodes,start,budget))
        routes=[tuple(p) for p in cat["paths"]]
        def crossings(p):
            return tuple((a,b) for a,b in zip(p,p[1:]) if (a in self.middle)!=(b in self.middle))
        def merit(p):
            return sum(min((l[3] for l in self.events.get((a,b),()) if l[4]<=self.limit
                            or (a in self.middle)!=(b in self.middle)),default=2.3)
                       for a,b in zip(p,p[1:]))+self.rng.random()*.5
        routes.sort(key=merit)
        families=Counter()
        trial={"cut":37,"join":19,"fixed_prefix_release":start,"bridge_budget":2,
               "method":"geometry-order enumeration"}
        self.hist.append(trial)
        try:
            best={}
            for used,t,legs in reps:
                best[used]=(t,legs)
                trial["measured_control_exit"]=t
                self.forward_candidate(before,legs,t,19,trial)
            self.save()
            for order in routes:
                self.check()
                family=crossings(order)
                if families[family]>=self.args.family_cap:
                    continue
                families[family]+=1
                self.stats["geometry_orders_timed"]+=1
                profiles,depth=self.chain(order,start,budget,False,False)
                trial["forward_depth"]=max(depth,trial.get("forward_depth",0))
                for used,(t,legs) in profiles.items():
                    self.stats["timed_middle_bridges"]+=1
                    if used not in best or t<best[used][0]:
                        best[used]=t,legs
                        trial["earliest_join"]=t
                        trial["middle_crossings"]=family
                        self.forward_candidate(before,legs,t,19,trial)
                if self.stats["geometry_orders_timed"]%35==0:
                    self.save()
            for goal,connector,tail in self.terminal_plans(19):
                profiles,depth=self.chain(connector,tail[0][2],0,True,True)
                trial["connector_depth"]=max(depth,trial.get("connector_depth",0))
                for _,(latest,suffix) in profiles.items():
                    for _,(t,bridge) in best.items():
                        gap=latest-t
                        trial.setdefault("gaps",[]).append({"goal":goal,"latest_join":latest,"exit_slack":gap})
                        if gap>=-8e-7:
                            self.full_candidate(before,bridge,suffix,tail,trial)
        except (BudgetExpired,KeyboardInterrupt):
            pass
        finally:
            self.save()
            print(dict(self.stats),trial,len(self.orders),self.best,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seconds",type=float,default=85.)
    p.add_argument("--phase",type=float,default=.023)
    p.add_argument("--seed",type=int,default=89105)
    p.add_argument("--family-cap",type=int,default=12)
    args=p.parse_args()
    args.atlas=FILES/"small_v4/diagnostic/window_e3e82fc984_checkpoint.json"
    args.resume=None
    args.mode="reserve"
    args.rebuild=True
    args.gates=1
    Ordered(args).run()

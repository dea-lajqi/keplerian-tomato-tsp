"""Rebuild polar entry and exit clocks after moving the first high-passage boundary."""
import argparse
from pathlib import Path
from itertools import pairwise
from medium_coupled.checkpoint import atomic_json
from pareto_easy import arrival
import json

from medium_frontier.polar import Polar
from medium_repartition.caps import strict_schedule


class StrictPolar(Polar):
    def reconstruct(self,route,starts,edges):
        sol=super().reconstruct(route,starts,edges)
        if self.args.export_middle:
            cleaned=strict_schedule(self.udp,sol)
            if cleaned is None:
                return sol
            rows=[(*edge,d,f,float(self.udp.compute_transfer(*edge,d,f))) for edge,d,f in
                  zip(pairwise(cleaned["path"]),cleaned["dep"],cleaned["tof"])]
            use=sum(x[-1]>self.normal for x in rows)
            if use<=2 and all(a[2]+a[3]<=b[2]+1e-9 for a,b in zip(rows,rows[1:])):
                file=self.args.directory/"middle_profiles.json"
                data=json.loads(file.read_text()) if file.exists() else {}
                ident=str(route[-1])
                epoch=arrival(rows[-1])
                if ident not in data or epoch<data[ident]["arrival"]:
                    data[ident]={"arrival":epoch,"legs":rows,"spend":use}
                    atomic_json(file,data)
        return sol

    def complete(self,*args,**kwargs):
        if not self.args.export_middle:
            return super().complete(*args,**kwargs)
        return None

    def submit(self, sol, method, plan):
        sol=strict_schedule(self.udp,sol)
        return sol is not None and super().submit(sol,method,plan)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed-record",type=Path,required=True)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--profiles",type=Path,required=True)
    p.add_argument("--seconds",type=float,default=420.)
    p.add_argument("--atlas-seconds",type=float,default=205.)
    p.add_argument("--goal",type=float,default=89.5)
    p.add_argument("--mission-goal",type=float,default=179.06)
    p.add_argument("--atlas-input",type=Path)
    p.add_argument("--change-exit",action="store_true")
    p.add_argument("--export-middle",action="store_true")
    p.set_defaults(mode="strict_polar_repartition",phase=None,beam=None,seed_atlas=None,
                   max_flight=2.4,meshes=((.02734375,.006),(.01953125,.013)),
                   reserve_gate=True,save_prefixes=True,polish=True,baseline_record=None,
                   anchors=())
    StrictPolar(p.parse_args()).run()

"""Test the measured 62→41→85 detour in complete five-crossing route orders."""
import argparse
from hashlib import sha256
from itertools import pairwise
from pathlib import Path
import time

from medium_coupled.checkpoint import atomic_json
from medium_frontier.nodal import orbit
from medium_frontier.topology import Topology
from medium_repartition.caps import strict_schedule
from medium_overnight.validation import solution_payload, verify


class Bypass(Topology):
    def submit(self,sol,method,plan):
        tightened=strict_schedule(self.udp,sol)
        if tightened is None or not super().submit(tightened,method,plan):
            return False
        fitness=verify(self.udp,tightened)
        key=sha256(bytes(sol["path"])).hexdigest()
        file=self.args.directory/"feasible"/f"{key}.json"
        if not file.exists():
            atomic_json(file,{"solution":tightened,**solution_payload(tightened,fitness,udp=self.udp,
                                               provenance={"method":"normal_bypass","trial":plan})})
        return True

    def variants(self, path):
        if (62,85) not in set(pairwise(path)):
            return ()
        children=[]
        for v in (41,91):
            order=tuple(n for n in path if n!=v)
            j=order.index(62)+1
            new=(*order[:j],v,*order[j:])
            if self.all_allowed(new) and self.route_floor(new)<=5:
                children.append((v,new))
            else:
                i=order.index(85)
                for a in range(max(0,i-4),min(len(order)-2,i+4)):
                    if a in (i-1,i):
                        continue
                    local=list(new)
                    local[a],local[a+1]=local[a+1],local[a]
                    tup=tuple(local)
                    if (62,v) in set(pairwise(tup)) and (v,85) in set(pairwise(tup)) and (
                            self.all_allowed(tup) and self.route_floor(tup)<=5):
                        children.append((v,tup))
        return children

    def run(self):
        self.orbits=[orbit(self.udp,i) for i in range(181)]
        options=self.layouts()
        examined=set()
        for family,rows in options.items():
            for rank,(_,path) in enumerate(rows):
                if time.monotonic()>=self.stop:
                    break
                if rank>72:
                    break
                for via,order in self.variants(path):
                    key=sha256(bytes(order)).hexdigest()
                    if key in examined:
                        continue
                    examined.add(key)
                    start=time.monotonic()
                    result=self.measure(order,.043 if rank%2 else .057,
                                        min(self.stop,start+11.),(family,rank,via))
                    self.state["trials"].append({"key":[family,rank,via],
                               "seconds":time.monotonic()-start,**result})
                    self.save()
                    print(family,rank,via,result,flush=True)
                    if len(examined)>40:
                        break
        self.state["unique_bypass_routes"]=len(examined)
        self.state["wall_seconds"]=time.monotonic()-self.started
        self.save()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed-record",type=Path,required=True)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--seconds",type=float,default=180.)
    p.add_argument("--goal",type=float,default=228.)
    p.set_defaults(mode="bypass_62_85",anchors=(),baseline_record=None,phase=None,beam=None,
                   normalize_passage=True,wide_retime=True)
    Bypass(p.parse_args()).run()

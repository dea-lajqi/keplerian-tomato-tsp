"""Keep the repair's early high-shell phase and reconstruct the terminal group."""
import argparse
from itertools import pairwise
import json
from pathlib import Path
import time

from medium_coupled.checkpoint import atomic_json
from medium_coupled.run import Paused, solve
from medium_repartition.caps import strict_schedule
from medium_repartition.near import NearRepair


class Prefix(NearRepair):
    def run(self):
        data=json.loads(self.args.near.read_text())
        sol=data["solution"]
        gate=160
        # The repaired candidate crosses 600, while its many small normal
        # excesses remain assigned to the normal class from the original join.
        original=json.loads(self.args.original.read_text())["solution"]
        caps=[x>self.normal for x in self.inspect(original,[True]*180)[0]]
        origins=[sol,original]
        profiles={}
        path=self.args.directory/"gate_pool.json"
        if path.exists():
            profiles=json.loads(path.read_text())
        checked={tuple(t["key"]) for t in self.state["trials"]}
        arms=[(idx,left,margin,due,k) for due in (179.05,178.95,178.8,179.2,179.02)
              for idx,left,margin in ((0,130,3.5),(1,127,5.5),(0,110,6.5),(0,0,2.0))
              for k in (0.,.15,-.23)]
        for key in arms:
            if key in checked or time.monotonic()>=self.stop:
                continue
            start=time.monotonic()
            idx,left,margin,due,k=key
            try:
                trial, info=self.fit(origins[idx], left,gate,margin,
                                   min(self.stop,start+self.args.trial_seconds),
                                   seam=155,kick=k,deadline=due,hard_flags=caps,iterations=240)
                prefix={name:values[:gate+int(name=="path")] for name,values in trial.items()}
                def accept(q):
                    q=strict_schedule(self.udp,q)
                    if q is None:
                        return False
                    rows=[]
                    for (a,b),d,f in zip(pairwise(q["path"]),q["dep"],q["tof"]):
                        if d<0 or f<self.udp.min_tof-1e-10 or (rows and d<rows[-1][2]+rows[-1][3]-1e-9):
                            return False
                        v=float(self.udp.compute_transfer(a,b,d,f))
                        rows.append((a,b,d,f,v))
                    used=sum(l[-1]>self.normal for l in rows)
                    if used>4 or any(l[-1]>self.udp.dv_threshold_exception for l in rows):
                        return False
                    finish=rows[-1][2]+rows[-1][3]
                    tag=str(tuple(q["path"][-12:]))
                    if tag not in profiles or finish<profiles[tag]["arrival"]-1e-6:
                        profiles[tag]={"legs":rows,"arrival":finish,"spend":used,
                                       "remaining":sol["path"][161:],"plan":key}
                        atomic_json(path,profiles)
                        print("GATE",finish,used,key,flush=True)
                    return True
                info["valid_prefix"]=accept(prefix)
                if info["valid_prefix"] or info["max_dv_excess_m_s"]<.6:
                    try:
                        solve(self.udp,prefix,left,margin,0.,min(self.stop,time.monotonic()+8.),
                              accept,self.count,cap_flags=caps[:gate])
                    except Paused:
                        pass
            except Paused:
                info={"time_cap":True}
            self.state["trials"].append({"key":key,"seconds":time.monotonic()-start,**info})
            self.save()
        self.state["wall_seconds"]=time.monotonic()-self.started
        self.save()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--seed-record",type=Path,required=True)
    p.add_argument("--near",type=Path,required=True)
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--seconds",type=float,default=330.)
    p.add_argument("--trial-seconds",type=float,default=9.)
    p.set_defaults(baseline_record=None)
    Prefix(p.parse_args()).run()

"""Test correlated changes at separated Medium crossings and the free maneuver."""
import argparse
from itertools import pairwise, product
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medium_coupled.nonlocal_routes import Surgery
from pareto_easy import BudgetExpired


class MultiGate(Surgery):
    def rank(self,path,left,original):
        changed=set(pairwise(path[left:]))-set(pairwise(original[left:]))
        return (sum(not self.anchors[e] for e in changed),
                sum(abs(self.inc[a]-self.inc[b]) for a,b in changed),
                len(changed))

    def paired(self):
        _,out,enter,gate=self.shells
        source=self.path
        proposals=[]
        for i,j,k,m in product((0,1,2,3),(0,1,2),(0,1,2,3),(0,1,2)):
            if not i+j or not k+m:
                continue
            changed=source[:]
            for a,b in ((out,out-i),(out+1,out+1+j),(enter,enter-k),(enter+1,enter+1+m)):
                changed[a],changed[b]=changed[b],changed[a]
            if not self.all_allowed(changed) or self.route_floor(changed)>5:
                continue
            left=out-i-1
            proposals.append((self.rank(changed,left,source),(i,j,k,m),left,tuple(changed)))
        for _,label,left,path in sorted(proposals)[:self.args.cases]:
            yield ("paired_crossings",*label),left,tuple(self.legs[:left]),path

    def allocated(self):
        items=[]
        for fname in self.args.profiles:
            for row in json.loads(fname.read_text()).values():
                pre=tuple(map(tuple,row["legs"]))
                cut=len(pre)
                if cut not in (112,116,120,121) or row["spend"]!=4:
                    continue
                assembled=tuple(self.build(pre)["path"]+row["remaining"])
                if len(assembled)!=181:
                    continue
                # Make one or two order changes through the formerly
                # exceptional region while leaving the last exception free.
                lo,hi=max(cut+1,120),134
                ordered=tuple(sorted(assembled[lo:hi],key=lambda a:self.inc[a]))
                candidates=[ordered]
                early=assembled[lo:hi]
                for x in range(len(ordered)-1):
                    after=list(ordered)
                    after[x],after[x+1]=after[x+1],after[x]
                    candidates.append(tuple(after))
                for order in candidates:
                    new=(*assembled[:lo],*order,*assembled[hi:])
                    if not self.all_allowed(new) or self.route_floor(new[cut:])!=1:
                        continue
                    items.append((self.rank(new,cut,assembled),row["arrival"],cut,order,pre,new))
        dedup=set()
        for _,clock,cut,group,pre,path in sorted(items,key=lambda item:item[:4]):
            key=cut,group
            if key in dedup:
                continue
            if len(dedup)>=self.args.cases:
                break
            dedup.add(key)
            code="-".join(map(str,group))
            yield ("early_exception",cut,round(clock,8),code),cut,pre,path

    def run(self):
        tested={tuple(row["key"]) for row in self.state["trials"]}
        proposals=self.paired() if self.args.mode=="paired" else self.allocated()
        for key,left,prefix,path in proposals:
            if time.monotonic()>=self.stop:
                break
            if key in tested:
                continue
            start=time.monotonic()
            observations={}
            try:
                self.complete(path,prefix,self.args.goal,.072265625 if self.args.mode=="paired" else .05078125,
                              min(self.stop,start+self.args.trial_seconds),key,observations)
            except BudgetExpired:
                observations["time_cap"]=True
            self.state["trials"].append({"key":key,"seconds":time.monotonic()-start,**observations})
            self.save()
            print(key,observations,self.state["best"],flush=True)
        self.state["wall_seconds"]=self.state.get("wall_seconds",0.)+time.monotonic()-self.started
        self.save()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",required=True,type=Path)
    p.add_argument("--seed-record",required=True,type=Path)
    p.add_argument("--baseline-record",type=Path)
    p.add_argument("--anchors",type=Path,nargs="*",default=())
    p.add_argument("--profiles",type=Path,nargs="*",default=())
    p.add_argument("--seconds",type=float,default=175.)
    p.add_argument("--trial-seconds",type=float,default=9.)
    p.add_argument("--goal",type=float,default=192.5)
    p.add_argument("--cases",type=int,default=36)
    p.add_argument("--mode",choices=("paired","allocation"),required=True)
    p.add_argument("--reserve-gate",action="store_true")
    p.set_defaults(beam=None,phase=None,save_prefixes=True,polish=True)
    MultiGate(p.parse_args()).run()

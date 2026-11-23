"""Overlap-aware Medium decoder with coverage, clock and exception labels."""
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from global_event_easy import fit_transfer_window, geometry
from medium_coupled.checkpoint import atomic_json
from medium_coupled.high_shell import RECORD_IMPROVEMENT_DAYS
from medium_coupled.meet import SideGrid
from medium_coupled.run import ROOT, Paused, solve
from medium_coupled.terminal_atlas import pareto
from medium_overnight.validation import official_udp, solution_payload, verify
from pareto_easy import BudgetExpired, arrival

ARCH=ROOT/"results/esa/medium_night/nov23_overnight"
THRESHOLD=RECORD_IMPROVEMENT_DAYS


class Frontier:
    def __init__(self,args):
        self.args=args
        self.started=time.monotonic()
        self.stop=self.started+args.seconds
        self.udp=official_udp(ROOT)
        self.raw=json.loads(args.seed_record.read_text())
        self.seed=self.raw["solution"]
        self.base=verify(self.udp,self.seed)[0]
        self.path=self.seed["path"]
        self.legs=tuple(map(tuple,self.raw["transfers"]))
        self.n=len(self.path)
        self.all_mask=(1<<self.n)-1
        self.normal,self.cap=self.udp.dv_threshold,self.udp.dv_threshold_exception+1e-6
        self.sizes=self.udp.tomato_orbital_parameters[:,0]
        self.natural,self.allowed=geometry(self.udp,4.9048695e12)[0]
        a=self.udp.tomato_orbital_parameters
        low=sorted((x for x in range(self.n) if self.sizes[x]<8e6),key=lambda x:a[x,2])
        self.low_masks=[sum(1<<x for x in low[j:j+20]) for j in (0,20,40)]
        self.high_mask=self.all_mask^sum(self.low_masks)
        self.bank=defaultdict(set)
        for f in (args.seed_record,*sorted((ARCH/"records").glob("*.json")),*args.anchors):
            row=json.loads(f.read_text())
            verify(self.udp,row["solution"])
            for a,b,d,t,_ in row["transfers"]:
                dv=float(self.udp.compute_transfer(a,b,d,t))
                if dv<=self.cap:
                    self.bank[a,b].add((a,b,d,t,dv))
        for f in args.events:
            timed=json.loads(f.read_text())["rows"]
            for values in timed.values():
                for leg in values:
                    if leg[-1]<=self.cap:
                        self.bank[leg[0],leg[1]].add(tuple(leg))
        args.directory.mkdir(parents=True,exist_ok=True)
        self.file=args.directory/"checkpoint.json"
        self.poolfile=args.directory/"gate_pool.json"
        self.state=json.loads(self.file.read_text()) if self.file.exists() else {
              "source":str(args.seed_record.resolve()),"baseline":self.base,"best":self.base,
              "record":None,"trials":[],"counts":{},"order_hashes":[]}
        if self.state["source"]!=str(args.seed_record.resolve()):
            raise ValueError("Checkpoint belongs to a different seed")
        self.stats=Counter(self.state["counts"])
        self.orders=set(self.state["order_hashes"])
        self.gates=json.loads(self.poolfile.read_text()) if self.poolfile.exists() else {}

    def save(self):
        self.state.update(counts=dict(self.stats),order_hashes=sorted(self.orders))
        atomic_json(self.file,self.state)

    def submit(self,legs,plan):
        sol={"path":[legs[0][0],*(l[1] for l in legs)],"dep":[l[2] for l in legs],
             "tof":[l[3] for l in legs]}
        self.stats["official_checks"]+=1
        try:
            f=verify(self.udp,sol)
        except ValueError:
            return
        self.stats["valid_schedules"]+=1
        self.orders.add(hashlib.sha256(bytes(sol["path"])).hexdigest())
        def save(candidate):
            try:
                replay=verify(official_udp(ROOT),candidate)
            except ValueError:
                return False
            if replay[0]<self.state["best"]-THRESHOLD:
                payload=solution_payload(candidate,replay,udp=self.udp,
                     provenance={"method":"overlapping_coverage_frontier","plan":plan})
                out=self.args.directory/"records"/f"{replay[0]:.12f}_{time.time_ns()}.json"
                atomic_json(out,{"solution":candidate,**payload})
                self.state.update(best=replay[0],record=str(out.resolve()))
                self.save()
                print("RECORD",replay,out,flush=True)
            return True
        if f[0]<self.state["best"]-THRESHOLD:
            save(sol)
        if f[0]<=self.state["best"]+.10 and time.monotonic()+.4<self.stop:
            try:
                solve(self.udp,sol,143,4.1,0.,min(self.stop,time.monotonic()+3.),
                      save,self.stats)
            except Paused:
                pass

    def shells_left(self,mask,a):
        remaining=self.all_mask^mask
        group=next((g for g in self.low_masks if g&(1<<a)),0)
        k=sum(bool(g&remaining) for g in self.low_masks if g!=group)
        if not group:
            return max(0,2*k-1)
        return 2*k if k else int(bool(self.high_mask&remaining))

    def save_gate(self,mask,node,spent,chain,key):
        if len(chain)!=160 or spent>4:
            return
        remaining=self.all_mask^mask
        if remaining.bit_count()!=20 or not any(remaining==g for g in self.low_masks):
            return
        id=f"{mask}:{node}:{spent}"
        t=arrival(chain[-1])
        if id not in self.gates or t<self.gates[id]["arrival"]-1e-9:
            self.gates[id]={"arrival":t,"legs":chain,"spend":spent,
                      "remaining":[x for x in self.path if remaining&(1<<x)],"plan":key}
            atomic_json(self.poolfile,self.gates)

    def run_case(self,span,band,beam,phase,limit,grid_enabled,plan,*,prefix_override=None):
        # Each block shares the moving coverage frontier with its neighbors.
        # No return to an incumbent intermediate arrival or membership is required.
        left,right=span
        order=self.path
        at=[1<<b for b in order]
        masks=[0]
        for b in at:
            masks.append(masks[-1]|b)
        origin=0.0
        grid=SideGrid(self.udp,origin,self.base+.35,.078125,phase,limit,False)
        finer=SideGrid(self.udp,origin,self.base+.35,.052734375,.016,limit,False)
        prefix=tuple(self.legs[:left]) if prefix_override is None else prefix_override
        spent=sum(l[-1]>self.normal for l in prefix)
        frontier={(masks[left+1],order[left],spent):(arrival(prefix[-1]) if prefix else 0.,prefix)}
        depth=left+1
        rows=Counter()
        archive={}
        for edge,pool in self.bank.items():
            atlas=tuple(pareto(l for l in pool if int(l[-1]>self.normal)==kind)
                        for kind in (0,1))
            archive[edge]=tuple((tuple(l[2] for l in rows),rows) for rows in atlas)
        rng=random.Random(self.args.seed+int(phase*10000)+left*5)
        try:
            for d in range(left+1,self.n):
                grid.check()
                bwidth=band if d<=right else (2 if d>=164 else 1)
                lo,hi=max(0,d-bwidth),min(self.n,d+bwidth+1)
                mandatory=masks[lo+1]
                eligible=masks[hi]^masks[lo]
                nxt={}
                ranked=sorted(frontier.items(),key=lambda x:x[1][0]+.37*x[0][2])
                for rank,((visited,a,e),(clock,chain)) in enumerate(ranked):
                    missing=mandatory&~visited
                    if missing.bit_count()>1:
                        continue
                    available=(eligible&~visited) if not missing else missing
                    while available:
                        bit=available&-available
                        available^=bit
                        b=bit.bit_length()-1
                        lower=self.shells_left(visited|bit,b)
                        extra_limit=self.udp.n_dv_exception-e-lower
                        if (extra_limit<0 or not self.allowed[a,b] or
                                (not self.natural[a,b] and not extra_limit)):
                            continue
                        ready=[]
                        for cls,(deps,opts) in enumerate(archive.get((a,b),())):
                            if cls<=extra_limit:
                                j=bisect_left(deps,clock-1e-9)
                                if j<len(deps):
                                    ready.append(opts[j])
                        sample=grid_enabled and (rank<self.args.sample_labels or not ready and rank<12)
                        if sample:
                            for lat in ((grid,finer) if not ready and rank<5 else (grid,)):
                                for leg in lat.edges(a,b,clock,4.5,8.,extra_limit>0):
                                    if leg is None:
                                        continue
                                    _,_,t,f,_=leg
                                    v=float(self.udp.compute_transfer(a,b,t,f))
                                    rows["replays"]+=1
                                    if v<=self.cap:
                                        row=(a,b,t,f,v)
                                        ready.append(row)
                                        self.bank[a,b].add(row)
                        if self.args.revive and grid_enabled and not ready and rank<5:
                            # The fast lattice may miss an entire phase branch.
                            # Search longer from the actual release only when
                            # this predecessor has no measured support.
                            for leg in finer.edges(a,b,clock,10.5,10.,extra_limit>0):
                                if leg is None:
                                    continue
                                _,_,d0,f0,_=leg
                                v=float(self.udp.compute_transfer(a,b,d0,f0))
                                rows["long_wait_replays"]+=1
                                if v<=self.cap:
                                    row=(a,b,d0,f0,v)
                                    ready.append(row)
                                    self.bank[a,b].add(row)
                        if grid_enabled and not ready and rank==0 and (d%4==0 or not self.natural[a,b]):
                            cap=self.cap if extra_limit else self.normal
                            fit,_=fit_transfer_window(self.udp,a,b,clock,
                                  min(self.base+.35,clock+6.2),cap,limit,rng)
                            rows["continuous_queries"]+=1
                            ready.extend(fit)
                            self.bank[a,b].update(fit)
                        for leg in ready:
                            t=arrival(leg)
                            cost=int(leg[-1]>self.normal)
                            if e+cost+lower>5 or leg[2]<clock-1e-9 or t>self.base+.35:
                                continue
                            key=visited|bit,b,e+cost
                            if key not in nxt or t<nxt[key][0]:
                                nxt[key]=(t,(*chain,leg))
                rows["raw_labels"]+=len(nxt)
                if not nxt:
                    break
                depth=d+1
                if depth==self.n:
                    for _,chain in sorted(nxt.values(),key=lambda row:row[0])[:12]:
                        self.submit(chain,plan)
                    break
                if depth==161:
                    for (mask,node,spent),(_,chain) in nxt.items():
                        self.save_gate(mask,node,spent,chain,plan)
                if beam:
                    by_mask=Counter()
                    frontier={}
                    ordered=sorted(nxt.items(),key=lambda x:(x[1][0]+.37*x[0][2],
                                             (x[0][0]^masks[depth]).bit_count()))
                    # Retain the best subset labels and the incumbent coverage
                    # as a control. Timing within that coverage remains free.
                    control=[item for item in ordered if item[0][0]==masks[depth]][:8]
                    for key,value in (*control,*ordered):
                        if key in frontier or by_mask[key[0],key[2]]>=8:
                            continue
                        by_mask[key[0],key[2]]+=1
                        frontier[key]=value
                        if len(frontier)>=beam:
                            break
                else:
                    frontier=nxt
        except BudgetExpired:
            rows["time_cap"]=True
        finally:
            rows.update(grid.stats)
            rows.update(finer.stats)
        return {"depth":depth,**dict(rows)}

    def run(self):
        done={tuple(t["key"]) for t in self.state["trials"]}
        # Archive-only control is exact on the tested event graph and
        # displacement domain. Timed arms enrich this graph and bound the beam.
        plans=[(18,155,3,0,.013,False),(18,75,3,68,.013,True),
               (89,145,3,78,.025,True),(120,161,6,95,.037,True),
               (39,125,4,90,.006,True),(68,164,5,96,.041,True)]
        if self.args.mode=="atlas":
            plans=[(126,160,6,0,.013,False),(125,165,8,0,.013,False),
                   (141,165,10,0,.013,False),(126,160,5,100,.031,True)]
        if self.args.mode=="allocate":
            plans=[(i+1,150,self.args.band,94,.011,True)
                   for i in self.args.pivots]
        for plan in plans:
            if plan in done or time.monotonic()>=self.stop:
                continue
            l,r,w,beam,phase,grid=plan
            start=time.monotonic()
            local=min(self.stop,start+self.args.trial_seconds)
            if self.args.mode=="allocate":
                info={}
                i=l-1
                a,b=self.path[i:i+2]
                release=arrival(self.legs[i-1])
                try:
                    candidates=[]
                    for branch in range(3):
                        found,_=fit_transfer_window(self.udp,a,b,release,arrival(self.legs[i]),
                                   self.cap,local,random.Random(self.args.seed+branch+i*61))
                        candidates.extend(x for x in found if x[-1]>self.normal)
                    if candidates:
                        earliest=min(candidates,key=arrival)
                        prefix=(*self.legs[:i],earliest)
                        info["reserved_edge"]=[a,b]
                        info["immediate_gain"]=arrival(self.legs[i])-arrival(earliest)
                        info.update(self.run_case((l,r),w,beam,phase,local,grid,plan,
                                                  prefix_override=prefix))
                except BudgetExpired:
                    info["time_cap"]=True
            else:
                info=self.run_case((l,r),w,beam,phase,local,grid,plan)
            self.stats.update({k:v for k,v in info.items() if k not in
                               ("depth","immediate_gain","reserved_edge")})
            self.state["trials"].append({"key":plan,"seconds":time.monotonic()-start,**info})
            self.save()
            print(plan,info,self.state["best"],flush=True)
        self.state["worker_seconds"]=self.state.get("worker_seconds",0.)+time.monotonic()-self.started
        self.save()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--seed-record",type=Path,required=True)
    p.add_argument("--anchors",nargs="*",type=Path,default=())
    p.add_argument("--events",nargs="*",type=Path,default=())
    p.add_argument("--mode",choices=("mixed","atlas","allocate"),default="mixed")
    p.add_argument("--seconds",type=float,default=240.)
    p.add_argument("--trial-seconds",type=float,default=45.)
    p.add_argument("--sample-labels",type=int,default=12)
    p.add_argument("--revive",action="store_true")
    p.add_argument("--band",type=int,default=5)
    p.add_argument("--pivots",type=int,nargs="+",default=(117,112,66,59,56))
    p.add_argument("--seed",type=int,default=80180)
    Frontier(p.parse_args()).run()

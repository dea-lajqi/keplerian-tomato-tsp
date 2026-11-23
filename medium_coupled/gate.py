"""Reconstruct the final shell exit and continuously retime the remaining tour."""
import argparse
from collections import Counter
from itertools import pairwise
import json
from pathlib import Path
import random
import sys
import time
from uuid import uuid4

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from global_event_easy import fit_transfer_window
from medium_coupled.checkpoint import atomic_json
from medium_coupled.run import ROOT, Paused, solve
from medium_overnight.validation import official_udp, solution_payload, verify
from medium_v4.driver import unpack
from pareto_easy import BudgetExpired


def run(args):
    start=time.monotonic()
    end=start+args.seconds
    u=official_udp(ROOT)
    s=unpack(json.loads(args.seed_record.read_text()))
    baseline=verify(u,s)[0]
    path,d,f=(s[k] for k in ("path","dep","tof"))
    costs=[float(u.compute_transfer(a,b,t,q))
           for (a,b),t,q in zip(pairwise(path),d,f)]
    flags=[x>u.dv_threshold for x in costs]
    axes=u.tomato_orbital_parameters[:,0]
    gate=max(i for i,(a,b) in enumerate(pairwise(path)) if min(axes[a],axes[b])*2<max(axes[a],axes[b]))
    file=args.directory/"checkpoint.json"
    state=json.loads(file.read_text()) if file.exists() else {
        "source":str(args.seed_record.resolve()),"best":baseline,"record":None,"trials":[],"gate":gate}
    if state["source"]!=str(args.seed_record.resolve()):
        raise ValueError("Checkpoint belongs to another seed")
    done={tuple(x["plan"]) for x in state["trials"]}
    count=Counter(state.get("counts",{}))

    def save():
        state["counts"]=dict(count)
        atomic_json(file,state)

    try:
        for previous,following in [(0,1),(0,2),(0,3),(1,0),(1,1),(0,4),(1,2),(0,0)]:
            candidate_path=path[:]
            i,j=gate-previous,gate+1+following
            candidate_path[i],candidate_path[gate]=candidate_path[gate],candidate_path[i]
            candidate_path[j],candidate_path[gate+1]=candidate_path[gate+1],candidate_path[j]
            release=d[gate-1]+f[gate-1] if not previous else d[gate]-.65
            a,b=candidate_path[gate:gate+2]
            candidate={"path":candidate_path,"dep":d,"tof":f}
            for offset in (.05,.45):
                key=(previous,following,offset)
                if key in done:
                    continue
                if time.monotonic()>=end:
                    raise Paused
                begin=time.monotonic()
                candidates,cost=fit_transfer_window(u,a,b,max(0.,release-offset),d[gate]+f[gate]+.22,
                                  u.dv_threshold_exception,end,random.Random(3401+31*previous+following))
                finishes=[]
                count["gate_queries"]+=1
                for _,_,dep,tof,_ in sorted(candidates,key=lambda l:l[2]+l[3])[:2]:
                    initialized=list(zip(d,f))
                    initialized[gate]=(dep,tof)
                    diff=dep+tof-d[gate]-f[gate]
                    for k in range(gate+1,len(d)):
                        initialized[k]=(d[k]+.65*diff,f[k])
                    def accept(sol):
                        count["official_checks"]+=1
                        try:
                            fit=verify(u,sol)
                        except ValueError:
                            return False
                        count["valid_schedules"]+=1
                        if fit[0]<state["best"]:
                            fit=verify(official_udp(ROOT),sol)
                            item=solution_payload(sol,fit,udp=u,provenance={"gate":gate,"plan":key,
                                           "seed_crossing":[a,b,dep,tof]})
                            record=args.directory/"records"/f"{fit[0]:.12f}_{uuid4().hex[:8]}.json"
                            atomic_json(record,{"solution":sol,**item})
                            state.update(best=fit[0],record=str(record.resolve()))
                            save()
                            print("RECORD",fit,record,flush=True)
                        return True
                    result=solve(u,candidate,gate-3,2.2,0.,min(end,begin+8.),
                                 accept,count,cap_flags=flags,start_times=initialized)
                    finishes.append(result)
                state["trials"].append({"plan":key,"edge":[a,b],"cost":cost,"fits":finishes,
                                        "seconds":time.monotonic()-begin})
                save()
    except (BudgetExpired,Paused):
        pass
    finally:
        state["wall_seconds"]=state.get("wall_seconds",0.)+time.monotonic()-start
        save()
    print("FINISH",state["best"],dict(count),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",required=True,type=Path)
    p.add_argument("--seed-record",required=True,type=Path)
    p.add_argument("--seconds",type=float,default=80)
    run(p.parse_args())

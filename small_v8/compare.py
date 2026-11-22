"""Independently validate and align the incumbent, V6 relocation, and V7."""
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from global_dp_easy import evaluator
from continuous_windows_easy import SEED, FILES
from kttsp_esa_solver import validate
from pareto_easy import arrival
from small_v2 import atomic_json


def main():
    out=FILES/"small_v8"
    old=json.loads(SEED.read_text())
    v7=json.loads(next((FILES/"small_v7/earliest_prefix_control").glob("*checkpoint.json")).read_text())["best_generated"]
    v6data=json.loads(next((FILES/"small_v6/reaudit/relocation").glob("*checkpoint.json")).read_text())
    v6=min(v6data["valid_candidates"],key=lambda row:row["validation"]["fitness"][0])
    u=evaluator().udp
    results={}
    arcs={}
    for label,row in (("incumbent",old),("v6",v6),("v7",v7)):
        legs=row["transfers"]
        check=validate(u,row["path"],legs)
        if not check or not check["valid"]:
            raise ValueError(f"ESA rejected {label}")
        actual=[]
        ready=0.
        for i,l in enumerate(legs):
            dv=float(u.compute_transfer(*l[:4]))
            actual.append({"arc":l[:2],"position":i,"departure":l[2],"flight":l[3],
                           "arrival":arrival(l),"wait":l[2]-ready,"dv":dv,
                           "exception":dv>u.dv_threshold})
            ready=arrival(l)
        arcs[label]={tuple(row["arc"]):row for row in actual}
        flight=sum(l[3] for l in legs)
        results[label]={"validation":check,"flight_days":flight,"waiting_days":ready-flight,
                        "exceptions":[x for x in actual if x["exception"]],"legs":actual}
    rows=[]
    for edge in sorted(set(arcs["incumbent"])|set(arcs["v7"])):
        a,b=arcs["incumbent"].get(edge),arcs["v7"].get(edge)
        rows.append({"arc":edge,"incumbent":a,"v7":b,
                     "change_flight":b["flight"]-a["flight"] if a and b else None,
                     "change_arrival":b["arrival"]-a["arrival"] if a and b else None})
    atomic_json(out/"route_comparison.json",{"tours":results,"aligned_transfers":rows})
    for k,r in results.items():
        print(k,r["validation"]["fitness"][0],r["flight_days"],r["waiting_days"])
    print("shared",sum(x["change_flight"] is not None for x in rows))
    ranked=sorted((x for x in rows if x["change_flight"] is not None),key=lambda x:x["change_flight"])
    print("largest changes",[(r["arc"],round(r["change_flight"],4),round(r["change_arrival"],4))
                            for r in ranked[:5]+ranked[-5:]])


if __name__=="__main__":
    main()

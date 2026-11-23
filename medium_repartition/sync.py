"""Independently replay and share records and changed-membership elites."""
import argparse
import json
from pathlib import Path

from medium_coupled.checkpoint import atomic_json
from medium_overnight.validation import official_udp, to_solution, verify


def rank(base, score_limit):
    outputs=[]
    for p in base.rglob("records/*.json"):
        if p.name.startswith("imported_"):
            continue
        raw=json.loads(p.read_text())
        if raw["arrival_day"]>=score_limit:
            continue
        udp=official_udp(Path.cwd())
        sol=to_solution(raw)
        score=verify(udp,sol)
        if any(float(udp.compute_transfer(*l[:4]))>udp.dv_threshold_exception for l in raw["transfers"]):
            continue
        outputs.append((score[0],p,raw))
    return sorted(outputs,key=lambda row:row[0])


def share(base):
    work=base/"nov23_campaign"
    inputs=json.loads((work/"campaign.json").read_text())
    results=rank(base,inputs["best"]-.0001)
    target=inputs.get("active",{}).get("partition",{}).get("directory")
    if target:
        for _,p,raw in results:
            if p.parent.parent==Path(target):
                continue
            out=Path(target)/"records"/("imported_"+p.name)
            if not out.exists():
                atomic_json(out,{**raw,"imported_campaign_record_from":str(p)})
    original=base/"seam_dp/checkpoint.json"
    if original.exists() and target:
        state=json.loads(original.read_text())
        rows=[row for row in state.get("complete",{}).values() if row.get("changes",0)>0]
        choices={row["file"]:row for row in sorted(rows,key=lambda r:r["flight"])[:8]}
        choices.update({row["file"]:row for row in sorted(rows,key=lambda r:-r["changes"])[:5]})
        for file,row in choices.items():
            p=Path(file)
            raw=json.loads(p.read_text())
            udp=official_udp(Path.cwd())
            verify(udp,to_solution(raw))
            out=Path(target)/"feasible"/("elite_"+p.name)
            if not out.exists():
                atomic_json(out,{**raw,"imported_campaign_record_from":str(p)})
    return [(score,str(p)) for score,p,_ in results]


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",type=Path,default=Path("results/esa/medium_repartition"))
    print(share(p.parse_args().directory),flush=True)

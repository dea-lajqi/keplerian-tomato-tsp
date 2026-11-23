"""Independently reproduce campaign record scores and archive metrics."""
from itertools import groupby
import json
from pathlib import Path

from medium_coupled.checkpoint import atomic_json
from medium_overnight.validation import official_udp, to_solution, verify
from medium_repartition.campaign import BASE
from medium_repartition.sync import rank

RUN=Path("results/esa/medium_repartition")


def main():
    baseline=json.loads(BASE.read_text())
    scores=rank(RUN,baseline["arrival_day"]-.0001)
    if not scores:
        raise RuntimeError("No campaign record passed strict replay")
    best,path,data=scores[0]
    u=official_udp(Path.cwd())

    def parts(rows):
        return [tuple(group) for _,group in groupby(rows,key=lambda x:
                        u.tomato_orbital_parameters[x,0]>8e6)]

    archive={}
    checkpoints=[]
    for file in RUN.rglob("checkpoint.json"):
        row=json.loads(file.read_text())
        checkpoints.append(row)
        for key,value in row.get("complete",{}).items():
            if key not in archive or value["finish"]<archive[key]["finish"]:
                archive[key]=value
    best_passes=parts(data["path"])
    original_passes=parts(baseline["path"])
    changed=set(original_passes[1])^set(best_passes[1])
    past=baseline["arrival_day"]
    history=[]
    for score,file,record in sorted(scores,key=lambda x:x[1].stat().st_mtime):
        if score<past-.0001:
            history.append({"fitness":record["validation"]["fitness"],"path":str(file),
                            "provenance":record["provenance"]})
            past=score
    checked=verify(u,to_solution(data))
    impulse=[float(u.compute_transfer(*row[:4])) for row in data["transfers"]]
    output={"best_path":str(path),"fitness":checked,
            "start_fitness":baseline["arrival_day"],"improvement_days":baseline["arrival_day"]-best,
            "flight_days":sum(row[3] for row in data["transfers"]),
            "initial_flight_days":sum(row[3] for row in baseline["transfers"]),
            "exception_transfers":[[i,*row[:4],dv] for i,(row,dv) in enumerate(zip(data["transfers"],impulse))
                                   if dv>u.dv_threshold],
            "max_dv":max(impulse),"passage_sizes":list(map(len,best_passes)),
            "moved_high_tomatoes":sorted(changed),"global_record_history":history,
            "strictly_replayed_record_files":len(scores),
            "unique_archived_orders":len(archive),
            "nominal_changed_membership_archive_orders":sum(x["changes"]>0 for x in archive.values()),
            "validated_archive_orders_by_180":sum(x["finish"]<=180 for x in archive.values()),
            "logged_official_checks":sum(r["counts"].get("official_checks",0) for r in checkpoints),
            "logged_valid_checks":sum(r["counts"].get("valid",0) for r in checkpoints),
            "summed_worker_seconds":sum(r.get("wall_seconds",0) for r in checkpoints),
            "campaigns":{p.parent.name:json.loads(p.read_text())["elapsed"]
                         for p in RUN.glob("*/campaign.json")}}
    atomic_json(RUN/"results_summary.json",output)
    print(checked,path,len(history),len(archive),flush=True)


if __name__=="__main__":
    main()

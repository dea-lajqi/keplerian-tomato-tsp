"""Audit the completed Small overnight campaign and revalidate its representatives."""
from collections import Counter, defaultdict
import json
from pathlib import Path

from continuous_windows_easy import SEED
from global_dp_easy import evaluator
from kttsp_esa_solver import validate
from pareto_easy import write_json

root = Path("results/esa/small_overnight/run_62201")
out = Path("results/esa/small_v2")
udp = evaluator().udp
campaign = json.loads((root/"campaign.json").read_text())
seed = json.loads(SEED.read_text())
reference = set(zip(seed["path"],seed["path"][1:]))
orders = {tuple(x) for x in campaign["order_keys"]}
representatives = {}
counts = defaultdict(Counter)
depth = defaultdict(list)
errors = []
for batch in campaign["history"]:
    counts[batch["mode"]].update(batch["counts"])
    counts[batch["mode"]].update(batches=1, failures=int(batch["exit_code"]!=0))
    if batch["exit_code"]:
        errors.append({"batch":batch["number"],"exit":batch["exit_code"],
                       "log":str(Path(batch["folder"])/"worker.log")})
    if not batch["checkpoint"]:
        continue
    raw = json.loads(Path(batch["checkpoint"]).read_text())
    for attempt in raw["attempts"]:
        depth[attempt["goal"]].append((attempt["max_visits"],attempt.get("valid_days")))
    for candidate in [raw.get("best_generated"),*raw.get("elite_candidates",[])]:
        if not candidate:
            continue
        path = tuple(candidate["path"])
        result = validate(udp,path,candidate["transfers"])
        if result and result["valid"]:
            score = result["fitness"][0]
            if path not in representatives or score<representatives[path]["validation"]["fitness"][0]:
                representatives[path] = {**candidate,"validation":result}
for file in (root/"elites").glob("*.json"):
    candidate = json.loads(file.read_text())
    path = tuple(candidate["path"])
    result = validate(udp,path,candidate["transfers"])
    if result and result["valid"] and (path not in representatives or
            result["fitness"][0]<representatives[path]["validation"]["fitness"][0]):
        representatives[path] = {**candidate,"validation":result}
report = {"baseline":validate(udp,seed["path"],seed["transfers"]),
          "counts":{k:dict(v) for k,v in counts.items()}, "errors":errors,
          "all_orders":[{"path":p,"changed_arcs":len(set(zip(p,p[1:]))-reference),
                         "days":representatives[p]["validation"]["fitness"][0] if p in representatives else None}
                        for p in sorted(orders)],
          "representatives":list(representatives.values()),
          "deadlines":{g:{"attempts":len(rows),"max_visits":max(k for k,v in rows),
                          "with_feasible":sum(v is not None for k,v in rows)} for g,rows in depth.items()}}
write_json(out/"audit.json",report)
print("orders",len(orders),"independently validated timing representatives",len(orders & representatives.keys()),
      "deadline stats",report["deadlines"],"errors",errors)

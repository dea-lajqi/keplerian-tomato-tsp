
import json
import time
import random
from pathlib import Path

import numpy as np
import beam_search as bs

RESULTS = Path("results")
SOURCE = RESULTS / "KTTSP_V44_best.json"
OUTPUT = RESULTS / "KTTSP_V46_best.json"

with open(SOURCE) as f:
    record = json.load(f)

best_count = len(record["path"])
best_record = record.copy()

# Ruajme rezultatet e perkohshme vecmas
bs.RESULTS = RESULTS / "v46_runs"
bs.RESULTS.mkdir(parents=True, exist_ok=True)

def adaptive_candidates(
    current, visited, arrival_day, exceptions_used,
    max_per_type=3
):
    candidates = []

    remaining = bs.t0 + bs.t_max - arrival_day
    if remaining <= bs.dt_min:
        return candidates

    # Kohet ndryshojne sipas kohes se mbetur
    offsets = [0, 0.5, 2, 5, 10]
    offsets += [random.uniform(0, min(12, remaining))
                for _ in range(2)]

    flights = [0.5, 1, 2, 4, 7, 11, 15]
    flights += [random.uniform(0.5, 15)
                for _ in range(2)]

    for target in range(len(bs.tomatoes)):
        if target in visited:
            continue

        normal = []
        exceptional = []

        for offset in offsets:
            dep = arrival_day + offset

            for tof in flights:
                finish = dep + tof

                if finish > bs.t0 + bs.t_max:
                    continue

                dv = bs.transfer_dv_cached(
                    current, target, float(dep), float(tof)
                )

                if not np.isfinite(dv):
                    continue

                candidate = (
                    target, dv, float(dep), float(tof),
                    float(finish), int(dv > bs.dv_max)
                )

                if dv <= bs.dv_max:
                    normal.append(candidate)
                elif dv <= bs.dv_exception and exceptions_used < bs.E:
                    exceptional.append(candidate)

        for group in (normal, exceptional):
            group.sort(key=lambda x: x[4])
            selected = []

            for candidate in group:
                if all(abs(candidate[4] - old[4]) >= 0.5
                       for old in selected):
                    selected.append(candidate)

                if len(selected) >= max_per_type:
                    break

            candidates.extend(selected)

    return candidates


bs.get_next_candidates_v16 = adaptive_candidates

# Pika te ndryshme ku rindertojme rrugen
experiments = [
    (30, 12),
    (25, 12),
    (20, 12),
    (15, 10),
]

for run, (prefix_length, width) in enumerate(experiments, 1):
    random.seed(200 + run)

    path = record["path"][:prefix_length]
    transfers = record["transfers"][:prefix_length - 1]

    arrival = float(transfers[-1][2]) + float(transfers[-1][3])
    used = sum(float(tr[4]) > bs.dv_max for tr in transfers)

    start_state = (path, arrival, used, transfers)

    print(f"\nV46 PROVA {run}: prefix={prefix_length}, "
          f"width={width}", flush=True)

    start = time.time()
    best = bs.beam_search_v44(start_state, beam_width=width)

    count = len(best[0])
    print(f"REZULTAT: {count}/49 | "
          f"Koha: {(time.time()-start)/60:.1f} min",
          flush=True)

    if count > best_count:
        best_count = count
        best_record = {
            "version": "V46",
            "path": best[0],
            "arrival_day": float(best[1]),
            "exceptions": int(best[2]),
            "transfers": best[3],
            "verified": False
        }

        with open(OUTPUT, "w") as f:
            json.dump(best_record, f, indent=2)

        print(f"REKORD I RI: {best_count}/49!", flush=True)

    if best_count >= 45:
        print("ARRITEM 45/49!", flush=True)
        break

print(f"\nREZULTATI ME I MIRE: {best_count}/49")

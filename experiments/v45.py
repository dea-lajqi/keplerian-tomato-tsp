
import json
import time
import random
from pathlib import Path

import numpy as np
import beam_search as bs

RESULTS = Path("results")
SOURCE = RESULTS / "KTTSP_V44_best.json"
OUTPUT = RESULTS / "KTTSP_V45_best.json"

with open(SOURCE) as f:
    record = json.load(f)

best_record = record.copy()
best_count = len(record["path"])

# V45 shkruan ne folder te vecante
bs.RESULTS = RESULTS / "v45_runs"
bs.RESULTS.mkdir(parents=True, exist_ok=True)

original_generator = bs.get_next_candidates_v16

def randomized_candidates(
    current, visited, arrival_day, exceptions_used,
    max_per_type=3
):
    candidates = []

    remaining = bs.t0 + bs.t_max - arrival_day
    if remaining <= bs.dt_min:
        return candidates

    # Kohe te ndryshme ne cdo prove
    departure_offsets = [0.0, 1.0, 3.0, 7.0, 12.0]
    departure_offsets += [
        random.uniform(0, min(15, remaining))
        for _ in range(3)
    ]

    flight_times = [0.5, 1.0, 2.0, 4.0, 7.0, 11.0]
    flight_times += [
        random.uniform(0.5, 15.0)
        for _ in range(3)
    ]

    for target in range(len(bs.tomatoes)):
        if target in visited:
            continue

        normal = []
        exceptional = []

        for offset in departure_offsets:
            dep = float(arrival_day + offset)

            for tof in flight_times:
                tof = float(tof)
                finish = dep + tof

                if finish > bs.t0 + bs.t_max:
                    continue

                dv = bs.transfer_dv_cached(
                    current, target, dep, tof
                )

                if not np.isfinite(dv):
                    continue

                candidate = (
                    target, dv, dep, tof, finish,
                    int(dv > bs.dv_max)
                )

                if dv <= bs.dv_max:
                    normal.append(candidate)
                elif dv <= bs.dv_exception and exceptions_used < bs.E:
                    exceptional.append(candidate)

        for group in (normal, exceptional):
            group.sort(key=lambda x: x[4])
            selected = []

            for candidate in group:
                if all(
                    abs(candidate[4] - old[4]) >= 0.5
                    for old in selected
                ):
                    selected.append(candidate)

                if len(selected) >= max_per_type:
                    break

            candidates.extend(selected)

    return candidates


# Zevendesojme gjeneratorin per kete proces
bs.get_next_candidates_v16 = randomized_candidates

experiments = [
    (25, 8),
    (20, 8),
    (15, 8),
]

for run, (prefix_length, width) in enumerate(experiments, 1):
    random.seed(100 + run)

    path = record["path"][:prefix_length]
    transfers = record["transfers"][:prefix_length - 1]

    arrival = float(transfers[-1][2]) + float(transfers[-1][3])
    used = sum(float(tr[4]) > bs.dv_max for tr in transfers)

    start_state = (path, arrival, used, transfers)

    print(
        f"\nV45 PROVA {run}: "
        f"prefix={prefix_length}, width={width}",
        flush=True
    )

    start = time.time()
    best = bs.beam_search_v44(start_state, beam_width=width)
    count = len(best[0])

    print(
        f"REZULTAT: {count}/49 | "
        f"Koha: {(time.time()-start)/60:.1f} min",
        flush=True
    )

    if count > best_count:
        best_count = count

        best_record = {
            "version": "V45",
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

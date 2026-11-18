
import json
import numpy as np
from scipy.optimize import differential_evolution
from beam_search import transfer_dv_cached

with open("results/KTTSP_41_verified.json") as f:
    record = json.load(f)

current = int(record["path"][-1])
start_day = float(record["arrival_day"])
missing = sorted(set(range(49)) - set(record["path"]))

remaining = 200.0 - start_day

print("Nisemi nga:", current)
print("Koha e mbetur:", round(remaining, 4), "dite")
print("Domatet qe mungojne:", missing, flush=True)

best = None

for target in missing:
    def objective(x):
        wait, tof = x

        departure = start_day + wait
        arrival = departure + tof

        if tof < 0.001 or arrival > 200:
            return 1e6 + max(0, arrival - 200) * 1e5

        dv = transfer_dv_cached(
            current, target,
            float(departure), float(tof)
        )

        if not np.isfinite(dv):
            return 1e6

        return float(dv)

    result = differential_evolution(
        objective,
        bounds=[
            (0, remaining - 0.001),
            (0.001, remaining)
        ],
        maxiter=25,
        popsize=6,
        seed=42,
        polish=False,
        tol=0.01
    )

    wait, tof = result.x
    departure = start_day + wait
    arrival = departure + tof
    dv = objective(result.x)

    print(
        "Domatja", target,
        "| DV:", round(dv, 3),
        "| Mberritja:", round(arrival, 3),
        flush=True
    )

    if dv <= 100 and arrival <= 200:
        if best is None or arrival < best[0]:
            best = (arrival, target, departure, tof, dv)

if best is None:
    print("Nuk u gjet transferim normal nga fundi i rruges.")
else:
    arrival, target, departure, tof, dv = best

    new_record = dict(record)
    new_record["path"] = record["path"] + [target]
    new_record["transfers"] = record["transfers"] + [
        [current, target, departure, tof, dv]
    ]
    new_record["arrival_day"] = arrival
    new_record["verified"] = False

    with open("results/KTTSP_42_candidate.json", "w") as f:
        json.dump(new_record, f, indent=2)

    print("KANDIDAT 42/49!")
    print("Domatja:", target)
    print("DV:", dv)
    print("U ruajt: results/KTTSP_42_candidate.json")

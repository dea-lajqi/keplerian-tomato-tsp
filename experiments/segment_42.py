
import json
import time
import itertools
import numpy as np
from scipy.optimize import minimize
from beam_search import transfer_dv_cached

SOURCE = "results/KTTSP_41_verified.json"
OUTPUT = "results/KTTSP_SEGMENT_42.json"
LIMIT = 60.0

with open(SOURCE) as f:
    record = json.load(f)

T = record["transfers"]
path = record["path"]
missing = sorted(set(range(49)) - set(path))

start = time.monotonic()
attempts = 0
found = False

print("SEGMENT SEARCH START", flush=True)

def dv(a, b, dep, tof):
    if tof < 0.001:
        return float("inf")
    try:
        return float(transfer_dv_cached(
            int(a), int(b), float(dep), float(tof)
        ))
    except Exception:
        return float("inf")

# Zgjedhim 3 domate te brendshme dhe provojme
# renditje te ndryshme + 1 domate te re.
for p in range(1, len(path) - 4):
    if time.monotonic() - start >= LIMIT:
        break

    left = path[p - 1]
    middle = path[p:p + 3]
    right = path[p + 3]

    earliest = (
        0.0 if p == 1
        else T[p - 2][2] + T[p - 2][3]
    )
    latest = (
        200.0 if p + 3 >= len(T)
        else T[p + 3][2]
    )

    window = latest - earliest
    if window <= 0.005:
        continue

    old_segment = T[p - 1:p + 3]
    old_exceptions = sum(t[4] > 100 for t in old_segment)

    for x in missing:
        if time.monotonic() - start >= LIMIT:
            break

        for order in itertools.permutations(middle + [x]):
            if time.monotonic() - start >= LIMIT:
                break

            nodes = [left] + list(order) + [right]
            nlegs = len(nodes) - 1

            def decode(z):
                # Koha e nisjes + ndarja e dritares
                wait = z[0] * max(0, window - 0.001*nlegs)
                available = window - wait

                weights = np.maximum(np.asarray(z[1:]), 0.001)
                tofs = (
                    0.001
                    + (available - 0.001*nlegs)
                    * weights / weights.sum()
                )

                dep = earliest + wait
                legs = []

                for i, tof in enumerate(tofs):
                    legs.append((nodes[i], nodes[i+1], dep, tof))
                    dep += tof

                return legs

            def objective(z):
                legs = decode(z)
                values = [
                    dv(a, b, dep, tof)
                    for a, b, dep, tof in legs
                ]

                if not all(np.isfinite(v) for v in values):
                    return 1e8

                excess = sorted(
                    [max(0, v - 100) for v in values]
                )

                # Lejojme vetem aq perjashtime
                # sa kishte segmenti origjinal.
                keep = max(0, len(excess) - old_exceptions)
                penalty = sum(v*v for v in excess[:keep])

                return penalty + 0.001*sum(values)

            z0 = np.ones(nlegs + 1) * 0.5
            z0[0] = 0.0

            result = minimize(
                objective,
                z0,
                method="SLSQP",
                bounds=[(0, 1)] * len(z0),
                options={"maxiter": 15, "ftol": 0.1}
            )

            attempts += 1
            legs = decode(result.x)

            new_legs = []
            for a, b, dep, tof in legs:
                new_legs.append([
                    a, b, float(dep), float(tof),
                    dv(a, b, dep, tof)
                ])

            if (
                all(t[4] <= 600 for t in new_legs)
                and sum(t[4] > 100 for t in new_legs)
                    <= old_exceptions
            ):
                new_T = (
                    T[:p - 1]
                    + new_legs
                    + T[p + 3:]
                )

                new_path = (
                    [new_T[0][0]]
                    + [t[1] for t in new_T]
                )

                candidate = {
                    "path": new_path,
                    "transfers": new_T,
                    "arrival_day": (
                        new_T[-1][2] + new_T[-1][3]
                    ),
                    "exceptions": sum(
                        t[4] > 100 for t in new_T
                    ),
                    "verified": False
                }

                if (
                    len(new_path) == 42
                    and len(set(new_path)) == 42
                    and candidate["arrival_day"] <= 200
                    and candidate["exceptions"] <= 5
                ):
                    with open(OUTPUT, "w") as f:
                        json.dump(candidate, f, indent=2)

                    print("U GJET 42/49 KANDIDAT!", flush=True)
                    print("Domatja e re:", x)
                    print("Ruajtur:", OUTPUT)
                    found = True
                    break

        if found:
            break
    if found:
        break

print("Prova:", attempts)
print("Koha:", round(time.monotonic() - start, 1))
print("Rezultati:", "42 kandidat" if found else "41/49")


import json
import time
import numpy as np
from scipy.optimize import differential_evolution
from beam_search import transfer_dv_cached

with open("results/KTTSP_41_verified.json") as f:
    record = json.load(f)

T = record["transfers"]
missing = sorted(set(range(49)) - set(record["path"]))

k = 36
a, b = 12, 10

earliest = T[k-1][2] + T[k-1][3]
latest = T[k+1][2]
window = latest - earliest

start = time.monotonic()

print("START - 8 domate", flush=True)
print("Dritarja:", round(window, 3), "dite", flush=True)

def dv(u, v, dep, tof):
    try:
        value = float(transfer_dv_cached(
            int(u), int(v), float(dep), float(tof)
        ))
        return value if np.isfinite(value) else 1e6
    except Exception:
        return 1e6

for x in missing:

    # 3 parametra brenda [0,1].
    # Kohet e krijuara jane gjithmone brenda dritares.
    def decode(z):
        wait = z[0] * (window - 0.002)
        available = window - wait

        tof1 = 0.001 + z[1] * (available - 0.002)
        remaining = available - tof1
        tof2 = 0.001 + z[2] * (remaining - 0.001)

        dep = earliest + wait
        middle = dep + tof1
        arrival = middle + tof2

        return dep, tof1, middle, tof2, arrival

    def objective(z):
        dep, tof1, middle, tof2, arrival = decode(z)

        d1 = dv(a, x, dep, tof1)
        d2 = dv(x, b, middle, tof2)

        return max(d1, d2) + 0.01 * (d1 + d2)

    result = differential_evolution(
        objective,
        bounds=[(0, 1)] * 3,
        maxiter=35,
        popsize=7,
        seed=42,
        polish=False
    )

    dep, tof1, middle, tof2, arrival = decode(result.x)
    d1 = dv(a, x, dep, tof1)
    d2 = dv(x, b, middle, tof2)

    print(
        "Domatja", x,
        "| DV:", round(d1, 2), round(d2, 2),
        flush=True
    )

    if d1 <= 100 and d2 <= 100:
        new_T = (
            T[:k]
            + [
                [a, x, dep, tof1, d1],
                [x, b, middle, tof2, d2]
            ]
            + T[k+1:]
        )

        path = [new_T[0][0]] + [t[1] for t in new_T]

        candidate = {
            "path": path,
            "transfers": new_T,
            "arrival_day": record["arrival_day"],
            "exceptions": sum(t[4] > 100 for t in new_T),
            "verified": False
        }

        with open("results/KTTSP_42_candidate.json", "w") as f:
            json.dump(candidate, f, indent=2)

        print("U GJET KANDIDAT 42/49!", flush=True)
        break

print("Koha:", round(time.monotonic()-start, 1), "sekonda")
print("PERFUNDOI")

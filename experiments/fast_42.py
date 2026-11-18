
import json
import time
import numpy as np
from scipy.optimize import minimize
from beam_search import transfer_dv_cached

SOURCE = "results/KTTSP_41_verified.json"
OUTPUT = "results/KTTSP_FAST_42.json"
LIMIT = 45

with open(SOURCE) as f:
    record = json.load(f)

T = record["transfers"]
missing = sorted(set(range(49)) - set(record["path"]))
start = time.monotonic()
calls = 0

print("FAST SEARCH START", flush=True)
print("Mungojne:", missing, flush=True)

def check(a, b, dep, tof):
    if tof < 0.001:
        return float("inf")
    return transfer_dv_cached(
        int(a), int(b), float(dep), float(tof)
    )

best = None

# Provojme insertion ne secilin transferim
for k, old in enumerate(T):
    if time.monotonic() - start > LIMIT:
        break

    a, b, old_dep, old_tof, old_dv = old

    earliest = 0 if k == 0 else T[k-1][2] + T[k-1][3]
    latest = 200 if k == len(T)-1 else T[k+1][2]

    # Duhet te ruajme buxhetin e perjashtimeve
    allowed = int(old_dv > 100)

    for x in missing:
        if time.monotonic() - start > LIMIT:
            break

        # Nisemi afer kohes ekzistuese
        seeds = [
            (old_dep, old_dep + old_tof / 2,
             old_dep + old_tof),
            (earliest, (earliest + latest) / 2, latest),
        ]

        for seed in seeds:
            if time.monotonic() - start > LIMIT:
                break

            def objective(z):
                dep, middle, arrival = z

                if not (
                    earliest <= dep
                    and dep + 0.001 <= middle
                    and middle + 0.001 <= arrival
                    and arrival <= latest
                ):
                    return 1e6

                dv1 = check(a, x, dep, middle-dep)
                dv2 = check(x, b, middle, arrival-middle)

                if not np.isfinite(dv1 + dv2):
                    return 1e6

                # Penalizojme DV mbi 100
                penalty = (
                    max(0, dv1-100)**2
                    + max(0, dv2-100)**2
                )

                return dv1 + dv2 + 1000*penalty

            result = minimize(
                objective,
                seed,
                method="Nelder-Mead",
                options={"maxiter": 35, "xatol": 0.01}
            )

            dep, middle, arrival = result.x

            if not (
                earliest <= dep
                and dep + 0.001 <= middle
                and middle + 0.001 <= arrival
                and arrival <= latest
            ):
                continue

            dv1 = check(a, x, dep, middle-dep)
            dv2 = check(x, b, middle, arrival-middle)

            calls += 1

            if not np.isfinite(dv1 + dv2):
                continue

            needed = int(dv1 > 100) + int(dv2 > 100)

            if (
                dv1 <= 600 and dv2 <= 600
                and needed <= allowed
            ):
                new_T = (
                    T[:k]
                    + [
                        [a, x, dep, middle-dep, dv1],
                        [x, b, middle, arrival-middle, dv2]
                    ]
                    + T[k+1:]
                )

                new_path = [new_T[0][0]]
                new_path += [t[1] for t in new_T]

                candidate = {
                    "path": new_path,
                    "transfers": new_T,
                    "arrival_day": new_T[-1][2] + new_T[-1][3],
                    "exceptions": sum(t[4] > 100 for t in new_T),
                    "verified": False
                }

                if (
                    len(set(new_path)) == 42
                    and candidate["exceptions"] <= 5
                ):
                    best = candidate

                    with open(OUTPUT, "w") as f:
                        json.dump(best, f, indent=2)

                    print("U GJET 42/49!", flush=True)
                    print("Domatja:", x)
                    print("U ruajt:", OUTPUT)
                    break

        if best:
            break
    if best:
        break

print("Koha:", round(time.monotonic()-start, 1), "sekonda")
print("Rezultati:", "42/49 kandidat" if best else "41/49")


import json
import time
import numpy as np
from beam_search import transfer_dv_cached

SOURCE = "results/KTTSP_41_verified.json"
OUTPUT = "results/KTTSP_42_candidate.json"

with open(SOURCE) as f:
    record = json.load(f)

transfers = record["transfers"]
missing = sorted(set(range(49)) - set(record["path"]))

start = time.monotonic()
LIMIT = 180  # maksimumi 3 minuta
best = None
checked = 0

# Zevendesojme A -> B me A -> X -> B
for k, old in enumerate(transfers):
    if time.monotonic() - start >= LIMIT:
        break

    a, b, old_dep, old_tof, old_dv = old

    earliest = (
        0.0 if k == 0
        else transfers[k-1][2] + transfers[k-1][3]
    )

    latest = (
        200.0 if k == len(transfers)-1
        else transfers[k+1][2]
    )

    # Sa perjashtime mund te perdorim ne vend te
    # transferimit ekzistues?
    available = 5 - record["exceptions"] + int(old_dv > 100)

    if available < 0:
        continue

    for x in missing:
        if time.monotonic() - start >= LIMIT:
            break

        # Kohe te ndryshme per dy fluturimet
        for dep in np.linspace(earliest, min(old_dep+1, latest), 5):
            if time.monotonic() - start >= LIMIT:
                break

            for middle in np.linspace(dep+0.1, latest-0.1, 7):
                if middle <= dep:
                    continue

                dv1 = transfer_dv_cached(
                    int(a), int(x),
                    float(dep), float(middle-dep)
                )

                if not np.isfinite(dv1) or dv1 > 600:
                    continue

                for arrival in np.linspace(middle+0.1, latest, 7):
                    if arrival <= middle:
                        continue

                    dv2 = transfer_dv_cached(
                        int(x), int(b),
                        float(middle), float(arrival-middle)
                    )
                    checked += 1

                    if not np.isfinite(dv2) or dv2 > 600:
                        continue

                    needed = int(dv1 > 100) + int(dv2 > 100)

                    if needed > available:
                        continue

                    candidate = (
                        arrival, k, x, dep, middle,
                        dv1, dv2
                    )

                    if best is None or arrival < best[0]:
                        best = candidate
                        print(
                            "U gjet kandidat:",
                            x, "pas", a,
                            "| DV:", round(dv1, 2),
                            round(dv2, 2),
                            flush=True
                        )

elapsed = time.monotonic() - start

print("Koha:", round(elapsed, 1), "sekonda")
print("Kombinime:", checked)

if best is None:
    print("Nuk u gjet insertion ne kete test.")
else:
    arrival, k, x, dep, middle, dv1, dv2 = best
    a, b = transfers[k][0], transfers[k][1]

    updated = [list(t) for t in transfers]
    updated[k:k+1] = [
        [a, x, float(dep), float(middle-dep), float(dv1)],
        [x, b, float(middle), float(arrival-middle), float(dv2)]
    ]

    new_record = dict(record)
    new_record["transfers"] = updated
    new_record["path"] = [updated[0][0]] + [t[1] for t in updated]
    new_record["exceptions"] = sum(t[4] > 100 for t in updated)
    new_record["verified"] = False

    with open(OUTPUT, "w") as f:
        json.dump(new_record, f, indent=2)

    print("KANDIDAT 42/49")
    print("Domatja e shtuar:", x)
    print("Perjashtime:", new_record["exceptions"])
    print("U ruajt:", OUTPUT)

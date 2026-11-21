
import json
import time
from pathlib import Path

import beam_search as bs

# Rekordi aktual 41/49
results = Path("results")
with open(results / "KTTSP_V44_best.json") as f:
    record = json.load(f)

best_count = len(record["path"])

# Folder i vecante - nuk prek V44
bs.RESULTS = results / "fast_search"
bs.RESULTS.mkdir(parents=True, exist_ok=True)

# Provojme disa pika nisjeje
experiments = [
    (30, 8),
    (25, 8),
    (20, 8),
]

for prefix_length, width in experiments:
    print(f"\n=== PROVA: prefix={prefix_length}, width={width} ===",
          flush=True)

    path = record["path"][:prefix_length]
    transfers = record["transfers"][:prefix_length - 1]

    arrival = float(transfers[-1][2]) + float(transfers[-1][3])
    used = sum(float(tr[4]) > bs.dv_max for tr in transfers)

    start_state = (path, arrival, used, transfers)

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

        with open(results / "KTTSP_FAST_BEST.json", "w") as f:
            json.dump({
                "version": "FAST",
                "path": best[0],
                "arrival_day": float(best[1]),
                "exceptions": int(best[2]),
                "transfers": best[3],
                "verified": False
            }, f, indent=2)

        print(f"REKORD I RI: {best_count}/49!", flush=True)

    if best_count >= 45:
        print("ARRITEM SYNIMIN 45/49!", flush=True)
        break

print(f"\nPERFUNDIMI: {best_count}/49")

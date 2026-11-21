
import json
import time
from pathlib import Path
from collections import defaultdict

from beam_search import transfer_dv_cached, tomatoes

RESULTS = Path("results")
SOURCE = RESULTS / "KTTSP_42_verified.json"
OUTPUT = RESULTS / "KTTSP_TARGETED_BEST.json"

LIMIT = 180
WIDTH = 90
MISSING = {4, 6, 11, 17, 24, 38, 47}

WAITS = [0, 0.5, 1.5, 3, 6]
TOFS = [0.5, 1, 2, 3.5, 5, 7, 10, 14]

with open(SOURCE) as f:
    record = json.load(f)

best = (
    record["path"],
    float(record["arrival_day"]),
    int(record["exceptions"]),
    record["transfers"]
)

def save(state):
    path, arrival, used, transfers = state
    with open(OUTPUT, "w") as f:
        json.dump({
            "path": path,
            "arrival_day": arrival,
            "exceptions": used,
            "transfers": transfers,
            "verified": False
        }, f, indent=2)

def search(prefix_length):
    global best

    path = record["path"][:prefix_length]
    transfers = record["transfers"][:prefix_length - 1]

    arrival = transfers[-1][2] + transfers[-1][3]
    used = sum(t[4] > 100 for t in transfers)

    beam = [(path, arrival, used, transfers)]
    start = time.monotonic()

    print("\nNISJA:", prefix_length, "domate", flush=True)

    while beam and time.monotonic() - start < LIMIT:

        expanded = []

        for state in beam:
            path, arrival, used, transfers = state
            visited = set(path)

            for target in range(len(tomatoes)):
                if target in visited:
                    continue

                choices = []

                for wait in WAITS:
                    dep = arrival + wait

                    for tof in TOFS:
                        finish = dep + tof
                        if finish > 200:
                            continue

                        dv = transfer_dv_cached(
                            path[-1],
                            target,
                            round(dep, 6),
                            float(tof)
                        )

                        if dv > 600:
                            continue

                        extra = int(dv > 100)

                        if used + extra > 5:
                            continue

                        choices.append(
                            (finish, target, dep, tof, dv, extra)
                        )

                # Ruajme kohe te ndryshme
                choices.sort()

                selected = []
                for c in choices:
                    if all(abs(c[0] - x[0]) >= 2 for x in selected):
                        selected.append(c)
                    if len(selected) >= 3:
                        break

                for finish, target, dep, tof, dv, extra in selected:
                    new_path = path + [target]

                    expanded.append((
                        new_path,
                        finish,
                        used + extra,
                        transfers + [
                            (path[-1], target, dep, tof, dv)
                        ]
                    ))

        if not expanded:
            break

        for state in expanded:
            if len(state[0]) > len(best[0]):
                best = state
                save(best)
                print(
                    "REKORD I RI:",
                    len(best[0]),
                    "/49",
                    flush=True
                )

        if len(best[0]) >= 45:
            return

        # Diversitet sipas domates, kohes dhe
        # domateve qe mungonin
        buckets = {}

        for state in expanded:
            path, arrival, used, transfers = state

            key = (
                path[-1],
                used,
                int(arrival // 4),
                frozenset(MISSING.intersection(path))
            )

            if key not in buckets or arrival < buckets[key][1]:
                buckets[key] = state

        candidates = list(buckets.values())

        # Ruajme rruge me me shume domate te
        # mungura dhe me pak perjashtime
        candidates.sort(
            key=lambda s: (
                -len(MISSING.intersection(s[0])),
                s[2],
                s[1]
            )
        )

        beam = candidates[:WIDTH]

        print(
            "Domate:", len(beam[0][0]),
            "| Aktive:", len(beam),
            "| Rekordi:", len(best[0]),
            "| Koha:", round(time.monotonic()-start, 1),
            flush=True
        )

for prefix in [30, 24, 18]:
    search(prefix)
    if len(best[0]) >= 45:
        break

print("\nPERFUNDOI")
print("Rezultati:", len(best[0]), "/49")
print("Dita:", round(best[1], 3))
print("Perjashtime:", best[2])

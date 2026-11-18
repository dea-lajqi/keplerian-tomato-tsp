
import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pykep as pk

from read_data import read_instance
from orbital import orbital_state_corrected as orbital_state_at_time, MU_MOON


# =========================
# TE DHENAT
# =========================

tomatoes = read_instance("data/easy.kttsp")[0]

t0 = 0.0
t_max = 200.0
dt_min = 0.001
dv_max = 100.0
dv_exception = 600.0
E = 5

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

SOURCE = RESULTS / "KTTSP_42_verified.json"
OUTPUT = RESULTS / "KTTSP_NEW_BEST.json"

TIME_LIMIT = 120
BEAM_WIDTH = 35
BACKTRACK = 15


# =========================
# LLOGARITJA E DELTA-V
# =========================

@lru_cache(maxsize=150000)
def transfer_dv_cached(i, j, start_day, tof_days):

    if tof_days < dt_min:
        return float("inf")

    r1, v1 = orbital_state_at_time(tomatoes[i], start_day)

    r2, v2 = orbital_state_at_time(
        tomatoes[j],
        start_day + tof_days
    )

    best_dv = float("inf")

    for cw in (False, True):
        try:
            lp = pk.lambert_problem(
                np.asarray(r1).tolist(),
                np.asarray(r2).tolist(),
                float(tof_days) * 86400,
                MU_MOON,
                cw,
                20
            )

            for dep_v, arr_v in zip(lp.v0, lp.v1):
                dv = (
                    np.linalg.norm(np.asarray(dep_v) - v1)
                    + np.linalg.norm(np.asarray(arr_v) - v2)
                )
                best_dv = min(best_dv, float(dv))

        except (ValueError, RuntimeError, OverflowError):
            continue

    return best_dv


# =========================
# KANDIDATET
# =========================

def get_candidates(current, visited, arrival, used):

    candidates = []

    remaining = t_max - arrival

    if remaining <= dt_min:
        return candidates

    wait_options = [
        0.0, 0.5, 1.5, 3.0, 6.0
    ]

    tof_options = [
        0.5, 1.0, 2.0, 3.5,
        5.0, 7.0, 10.0, 14.0
    ]

    for target in range(len(tomatoes)):

        if target in visited:
            continue

        normal = []
        exceptional = []

        for wait in wait_options:

            dep = arrival + wait

            if dep >= t_max:
                continue

            for tof in tof_options:

                finish = dep + tof

                if finish > t_max:
                    continue

                dv = transfer_dv_cached(
                    current,
                    target,
                    round(dep, 6),
                    float(tof)
                )

                if not np.isfinite(dv):
                    continue

                extra = int(dv > dv_max)

                if dv <= dv_max:
                    normal.append(
                        (target, dv, dep, tof, finish, extra)
                    )

                elif dv <= dv_exception and used < E:
                    exceptional.append(
                        (target, dv, dep, tof, finish, extra)
                    )

        for group in (normal, exceptional):

            group.sort(key=lambda x: x[4])

            selected = []

            for candidate in group:

                if all(
                    abs(candidate[4] - old[4]) >= 1.0
                    for old in selected
                ):
                    selected.append(candidate)

                if len(selected) >= 2:
                    break

            candidates.extend(selected)

    return candidates


# =========================
# BEAM SEARCH
# =========================

def beam_search(start_state, initial_best):

    start_time = time.monotonic()

    beams = [start_state]
    best = initial_best

    def save_best(state):

        path, arrival, used, transfers = state

        output = {
            "version": "IMPROVED_BEAM",
            "path": path,
            "arrival_day": float(arrival),
            "exceptions": int(used),
            "transfers": transfers,
            "verified": False
        }

        with open(OUTPUT, "w") as f:
            json.dump(output, f, indent=2)

    print("BEAM SEARCH START", flush=True)
    print("Rekordi fillestar:", len(best[0]), "/49", flush=True)

    for step in range(len(start_state[0]), len(tomatoes)):

        if time.monotonic() - start_time >= TIME_LIMIT:
            break

        next_states = []

        for state in beams:

            if time.monotonic() - start_time >= TIME_LIMIT:
                break

            path, arrival, used, transfers = state

            candidates = get_candidates(
                path[-1],
                set(path),
                arrival,
                used
            )

            for target, dv, dep, tof, finish, extra in candidates:

                new_used = used + extra

                if new_used > E:
                    continue

                new_state = (
                    path + [target],
                    finish,
                    new_used,
                    transfers + [
                        (path[-1], target, dep, tof, dv)
                    ]
                )

                next_states.append(new_state)

                if len(new_state[0]) > len(best[0]):

                    best = new_state
                    save_best(best)

                    print(
                        "REKORD I RI:",
                        len(best[0]),
                        "/49",
                        flush=True
                    )

        if not next_states:
            break

        # Ruajme larmi ne kohe dhe ne domaten e fundit
        buckets = {}

        for state in next_states:

            path, arrival, used, transfers = state

            key = (
                path[-1],
                frozenset(path),
                used,
                int(arrival // 5)
            )

            if key not in buckets or arrival < buckets[key][1]:
                buckets[key] = state

        grouped = {}

        for state in buckets.values():

            bucket = int(state[1] // 10)

            grouped.setdefault(bucket, []).append(state)

        for group in grouped.values():
            group.sort(key=lambda s: (s[2], s[1]))

        selected = []

        while len(selected) < BEAM_WIDTH:

            added = False

            for bucket in sorted(grouped):

                if grouped[bucket]:
                    selected.append(grouped[bucket].pop(0))
                    added = True

                    if len(selected) >= BEAM_WIDTH:
                        break

            if not added:
                break

        beams = selected

        print(
            "Domate:",
            step + 1,
            "| Aktive:",
            len(beams),
            "| Rekordi:",
            len(best[0]),
            "/49",
            "| Koha:",
            round(time.monotonic() - start_time, 1),
            flush=True
        )

    return best


# =========================
# NISJA NGA REKORDI 41
# =========================

if __name__ == "__main__":

    with open(SOURCE) as f:
        record = json.load(f)

    full_path = record["path"]
    full_transfers = record["transfers"]

    prefix_length = max(2, len(full_path) - BACKTRACK)

    path = full_path[:prefix_length]
    transfers = full_transfers[:prefix_length - 1]

    arrival = (
        float(transfers[-1][2])
        + float(transfers[-1][3])
    )

    used = sum(
        float(t[4]) > dv_max
        for t in transfers
    )

    start_state = (
        path,
        arrival,
        used,
        transfers
    )

    initial_best = (
        full_path,
        float(record["arrival_day"]),
        int(record["exceptions"]),
        full_transfers
    )

    print("Fillojme nga:", len(path), "domate")
    print("Dita:", round(arrival, 3))
    print("Perjashtime:", used)
    print("Synimi: mbi 41/49")

    best = beam_search(start_state, initial_best)

    print("\nPERFUNDOI")
    print("Rezultati:", len(best[0]), "/49")
    print("Dita:", round(best[1], 3))
    print("Perjashtime:", best[2])

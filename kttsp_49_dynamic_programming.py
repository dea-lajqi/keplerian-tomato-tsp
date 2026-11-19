"""KTTSP Easy: a reproducible *search* for the 49-tomato schedule.

HOW IT WORKS
  1) Read the user's previously PyKEP-validated 46-tomato checkpoint.
  2) Construct a candidate 49-tomato visiting order by two explicit
     route-neighborhood changes (the *order* is heuristic, not a 49! search).
  3) Solve departure times and flight durations FROM SCRATCH via dynamic
     programming (DP) and actual PyKEP Lambert transfers. No saved 49/49
     route, transfer times or delta-V values are read or used as a guide.
  4) Recompute all 48 transfer delta-V values independently and write a NEW
     result only if all dataset constraints pass.

Run from your project directory with PyKEP installed:
    python kttsp_49_dynamic_programming.py

Required local files:
    data/easy.kttsp
    read_data.py
    orbital.py
    results/KTTSP_46_DAY183_VALIDATED.json

The original 46 checkpoint and 49 checkpoint are never overwritten.
The official ESA evaluator is not run by this program.
"""

import json
import math
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pykep as pk

from read_data import read_instance
from orbital import orbital_state_corrected, MU_MOON

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results" / "KTTSP_46_DAY183_VALIDATED.json"
DATA = ROOT / "data" / "easy.kttsp"
DESTINATION = ROOT / "results" / "KTTSP_49_DP_SEARCH_VALIDATED.json"

# Search discretization in days, chosen for this KTTSP Easy instance.
TIME_STEP = 0.25
WAIT_LIMIT_DAYS = 14.0
FLIGHT_TIMES = (
    0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0,
    2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0,
    8.0, 9.0, 10.0, 12.0,
)


def build_candidate_order(base_path):
    """Guided ROUTE-ORDER search: alter local neighborhood and insert cluster.

    We are intentionally explicit about these two heuristics; this step is
    NOT a claim of discovering the optimal global permutation automatically.
    The DP below DOES independently search every leg's departure and TOF.
    """
    path = list(base_path)
    k = path.index(7)
    expected = [7, 6, 35, 26, 21, 24]
    if path[k:k + 6] != expected:
        raise ValueError(f"46 checkpoint differs near tomato 7: {path[k:k + 6]}")

    # Reordering this portion produces a feasible earlier encounter with 45.
    path[k:k + 6] = [7, 21, 26, 35, 6, 24]

    # The three previously unvisited tomatoes form a natural small cluster.
    k = path.index(45)
    path[k + 1:k + 1] = [4, 17, 11]
    return path


def main():
    started = time.monotonic()
    if not SOURCE.exists():
        raise SystemExit(f"Cannot find verified 46 checkpoint: {SOURCE}")
    if not DATA.exists():
        raise SystemExit(f"Cannot find input dataset: {DATA}")

    tomatoes, t0, dt_min, t_max, normal_limit, exception_limit, max_exceptions = read_instance(str(DATA))
    base = json.loads(SOURCE.read_text(encoding="utf-8"))
    base_path = [int(x) for x in base["path"]]
    prefix_legs = [list(x) for x in base["transfers"][:3]]
    if len(base_path) != 46 or len(set(base_path)) != 46:
        raise ValueError("Expected 46 different tomato indices in the source")
    if base_path[:4] != [32, 27, 16, 14]:
        raise ValueError("Unexpected source prefix; this search needs a new prefix")

    path = build_candidate_order(base_path)
    if len(path) != len(tomatoes) or sorted(path) != list(range(len(tomatoes))):
        raise ValueError("Constructed candidate does not contain all tomato indices")

    # Cache orbital positions to avoid repeated Kepler solving for every Lambert arc.
    @lru_cache(maxsize=250000)
    def state(i, t):
        return orbital_state_corrected(tomatoes[i], float(t))

    def actual_dv(i, j, departure, tof):
        """True physics objective: sum of the two impulsive corrections."""
        try:
            r1, v1 = state(i, departure)
            r2, v2 = state(j, round(departure + tof, 6))
        except (ValueError, RuntimeError, OverflowError, FloatingPointError):
            return math.inf
        best = math.inf
        for cw in (False, True):
            try:
                lambert = pk.lambert_problem(
                    np.asarray(r1, float).tolist(),
                    np.asarray(r2, float).tolist(),
                    float(tof) * 86400.0,
                    MU_MOON, cw, 20,
                )
                for vi, vf in zip(lambert.v0, lambert.v1):
                    dv = (np.linalg.norm(np.asarray(vi) - v1)
                          + np.linalg.norm(np.asarray(vf) - v2))
                    best = min(best, float(dv))
            except (ValueError, RuntimeError, OverflowError):
                continue
        return best

    @lru_cache(maxsize=200000)
    def dv(i, j, departure, tof):
        return actual_dv(i, j, departure, tof)

    # State: number of exceptions already used ->
    #        (earliest reached day, list of selected transfers)
    # DP stores one earliest arrival per exception count; this is a heuristic
    # pruning choice for time-dependent orbits, not a mathematical optimality proof.
    previous_arrival = float(t0)
    prefix_exceptions = 0
    for idx, leg in enumerate(prefix_legs):
        a, b, dep, tof, _ = leg
        if (a, b) != (path[idx], path[idx + 1]):
            raise ValueError("46 checkpoint prefix does not match candidate order")
        if dep < previous_arrival - 1e-8 or tof < dt_min:
            raise ValueError("Invalid timing in 46 checkpoint prefix")
        real = actual_dv(a, b, dep, tof)
        if not math.isfinite(real) or real > exception_limit + 1e-7:
            raise ValueError(f"Invalid first-3-leg transfer {a}->{b} DeltaV={real}")
        prefix_exceptions += int(real > normal_limit + 1e-7)
        prefix_legs[idx] = [int(a), int(b), float(dep), float(tof), real]
        previous_arrival = dep + tof

    if prefix_exceptions > max_exceptions:
        raise ValueError("Too many exceptions in prefix")

    print(f"START | 46 -> 49 | missing [4, 11, 17] | starting DP at day {previous_arrival:.2f}", flush=True)
    print("ORDER |", " -> ".join(map(str, path)), flush=True)
    print("ALGORITHM | DP over departure wait, time-of-flight and exceptions", flush=True)
    print("No saved 49-tomato schedule is read.", flush=True)

    @lru_cache(maxsize=50000)
    def best_successors(i, j, arrival):
        """Search possible waits and TOFs; one earliest edge per delta-V class."""
        best = [None, None]  # [normal, exceptional]
        first_departure = math.ceil((arrival - 1e-7) / TIME_STEP) * TIME_STEP
        if first_departure < arrival - 1e-9:
            first_departure += TIME_STEP
        shell_change = abs(tomatoes[i][0] / tomatoes[j][0] - 1.0) > 0.30
        n_waits = int(round(WAIT_LIMIT_DAYS / TIME_STEP))

        for k in range(n_waits + 1):
            departure = round(first_departure + k * TIME_STEP, 6)
            if departure > t_max:
                break
            for tof in FLIGHT_TIMES:
                if tof < dt_min:
                    continue
                end = round(departure + tof, 6)
                if end > t_max:
                    break
                if best[0] is not None and best[1] is not None:
                    if end >= max(best[0][0], best[1][0]):
                        continue
                if shell_change and best[1] is not None and end >= best[1][0]:
                    continue

                value = dv(i, j, departure, tof)
                if not math.isfinite(value) or value > exception_limit + 1e-7:
                    continue
                extra = int(value > normal_limit + 1e-7)
                if best[extra] is None or end < best[extra][0]:
                    best[extra] = (end, [int(i), int(j), departure, tof, value])

            # A later departure cannot improve the earliest feasible arrival.
            if (best[0] is not None or shell_change) and best[1] is not None:
                bound = max(best[0][0] if best[0] is not None else 0.0,
                            best[1][0])
                if departure + TIME_STEP >= bound:
                    break
            if best[0] is not None and best[1] is None:
                if departure + TIME_STEP >= best[0][0] + 10.0:
                    break
        return best

    states = {prefix_exceptions: (previous_arrival, prefix_legs)}
    stages = []
    # We fixed the first 3 edges from the verified 46 checkpoint.
    for pos in range(3, len(path) - 1):
        i, j = path[pos], path[pos + 1]
        next_states = {}
        for exc_used, (arrival, selected) in states.items():
            for extra, candidate in enumerate(best_successors(i, j, round(arrival, 6))):
                if candidate is None or exc_used + extra > max_exceptions:
                    continue
                finish, transfer = candidate
                new_exc = exc_used + extra
                if new_exc not in next_states or finish < next_states[new_exc][0]:
                    next_states[new_exc] = (finish, selected + [transfer])
        if not next_states:
            raise RuntimeError(
                f"Search ended at step {pos + 1}: no feasible {i}->{j}; "
                "the grid / heuristic might need widening."
            )
        states = next_states
        progress = {e: round(s[0], 2) for e, s in sorted(states.items())}
        stages.append({"edge": [i, j], "arrivals_by_exceptions": progress})
        print(f"HAPI {pos + 1:02d}/48 | {i:2d} -> {j:2d} | "
              f"DP arrivals {progress} | {time.monotonic()-started:.1f}s", flush=True)

    used, (finish_day, transfers) = min(
        states.items(), key=lambda item: (item[1][0], item[0]))
    if len(transfers) != 48:
        raise RuntimeError(f"Search generated {len(transfers)} transfers, expected 48")

    # Independent final computation; no cached dv reuse.
    checked = []
    arrival = float(t0)
    verified_exceptions = 0
    for k, (i, j, dep, tof, _old) in enumerate(transfers, 1):
        if (i, j) != (path[k - 1], path[k]):
            raise RuntimeError(f"Wrong order at transfer {k}")
        if dep < arrival - 1e-8 or tof < dt_min:
            raise RuntimeError(f"Timing invalid at transfer {k}")
        arrival = dep + tof
        if arrival > t_max + 1e-8:
            raise RuntimeError(f"Mission > {t_max} days at transfer {k}")
        value = actual_dv(i, j, dep, tof)
        if not math.isfinite(value) or value > exception_limit + 1e-7:
            raise RuntimeError(f"Invalid physics at transfer {k}: {i}->{j} dv={value}")
        verified_exceptions += int(value > normal_limit + 1e-7)
        if verified_exceptions > max_exceptions:
            raise RuntimeError(f"Too many exceptions at transfer {k}")
        checked.append([i, j, dep, tof, value])

    result = {
        "path": path,
        "transfers": checked,
        "arrival": arrival,
        "exceptions": verified_exceptions,
        "verified_pykep": True,
        "verified_official": False,
        "method": "guided 46->49 order construction + dynamic-programming time search + independent PyKEP validation",
        "search_log": stages,
    }
    DESTINATION.parent.mkdir(exist_ok=True)
    DESTINATION.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\n---------- FINAL ROUTE ----------", flush=True)
    print(f"START: tomato {path[0]} at day {t0:.2f}")
    for k, (i, j, dep, tof, value) in enumerate(checked, 1):
        special = " EXCEPTION" if value > normal_limit + 1e-7 else ""
        print(f"HAPI {k:02d}/48 | {i:2d} -> {j:2d} | "
              f"departure {dep:6.2f} | flight {tof:5.2f} | "
              f"arrival {dep+tof:6.2f} | DV {value:7.2f}{special}")
    print(f"\nSUCCESS: {len(path)}/{len(tomatoes)} | day {arrival:.2f}/{t_max:.2f} "
          f"| exceptions {verified_exceptions}/{max_exceptions}", flush=True)
    print(f"SAVED: {DESTINATION}", flush=True)
    print("PyKEP checked locally; official evaluator still required.", flush=True)


if __name__ == "__main__":
    main()


import json
import math
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from scipy.optimize import minimize

from kttsp_esa_solver import load_evaluator, validate

ROOT = Path(__file__).resolve().parent
SEED = ROOT / "results/esa/easy_refined_20261121T091447Z.json"

udp = load_evaluator(ROOT / "esa_evaluation", "small", False)
saved = json.loads(SEED.read_text(encoding="utf-8"))

path = saved["path"]
best_legs = [list(leg) for leg in saved["transfers"]]

baseline = validate(udp, path, best_legs)
if not baseline or not baseline["valid"]:
    raise RuntimeError("Seed nuk eshte valid.")

best_days = baseline["fitness"][0]
print(f"Fillimi: {best_days:.6f} dite", flush=True)

n = len(best_legs)
start_time = time.monotonic()
deadline = start_time + 120
improvements = 0
attempts = 0

# Mbajme buxhetin ekzistues te perjashtimeve.
exception_indices = {
    i for i, leg in enumerate(best_legs)
    if leg[4] > udp.dv_threshold
}

while time.monotonic() < deadline:
    improved_this_round = False

    # Provojme segmente me 3, 4 dhe 5 transferime.
    for size in (3, 4, 5):
        for start in range(n - size + 1):
            if time.monotonic() >= deadline:
                break

            attempts += 1
            indices = list(range(start, start + size))
            segment = [best_legs[i] for i in indices]

            x0 = np.array(
                [leg[2] for leg in segment]
                + [leg[3] for leg in segment],
                dtype=float,
            )

            def objective(x):
                # Segmentet e brendshme minimizojne
                # kohen e perfundimit te segmentit.
                return float(x[size - 1] + x[-1])

            def constraints(x):
                departures = x[:size]
                flights = x[size:]
                values = []

                # Kronologjia brenda segmentit.
                values.extend(
                    departures[1:] - departures[:-1] - flights[:-1]
                )

                # Kufiri me transferimin paraprak.
                if start > 0:
                    previous = best_legs[start - 1]
                    previous_end = previous[2] + previous[3]
                    values.append(departures[0] - previous_end)

                # Kufiri me transferimin pasues.
                if start + size < n:
                    next_departure = best_legs[start + size][2]
                    values.append(
                        next_departure - departures[-1] - flights[-1]
                    )

                # Delta-V per secilin transferim.
                for j, leg in enumerate(segment):
                    idx = indices[j]
                    limit = (
                        udp.dv_threshold_exception
                        if idx in exception_indices
                        else udp.dv_threshold
                    )
                    try:
                        dv = udp.compute_transfer(
                            leg[0], leg[1],
                            float(departures[j]),
                            float(flights[j]),
                        )
                        values.append(limit - dv)
                    except Exception:
                        values.append(-1e9)

                return np.asarray(values, dtype=float)

            bounds = (
                [(0.0, udp.max_time)] * size
                + [(udp.min_tof, udp.max_time)] * size
            )

            result = minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints={"type": "ineq", "fun": constraints},
                options={"maxiter": 20, "ftol": 1e-7},
            )

            candidate = [leg.copy() for leg in best_legs]

            for j, idx in enumerate(indices):
                departure = float(result.x[j])
                flight = float(result.x[size + j])
                leg = candidate[idx]
                dv = float(udp.compute_transfer(
                    leg[0], leg[1], departure, flight
                ))
                candidate[idx] = [
                    leg[0], leg[1], departure, flight, dv
                ]

            checked = validate(udp, path, candidate)

            if checked and checked["valid"]:
                days = checked["fitness"][0]

                if days < best_days - 1e-8:
                    best_days = days
                    best_legs = candidate
                    improvements += 1
                    improved_this_round = True

                    stamp = datetime.now(timezone.utc).strftime(
                        "%Y%m%dT%H%M%SZ"
                    )
                    output = ROOT / "results/esa" / (
                        f"easy_segment_{stamp}_{attempts}.json"
                    )
                    output.write_text(
                        json.dumps({
                            "path": path,
                            "transfers": best_legs,
                            "arrival_day": best_days,
                            "validation": checked,
                        }, indent=2),
                        encoding="utf-8",
                    )

                    print(
                        f"PERMIRESIM! {best_days:.6f} dite "
                        f"| Segmenti {start}-{start+size-1}",
                        flush=True,
                    )

    if not improved_this_round:
        break

print(
    f"PERFUNDIM: {best_days:.6f} dite | "
    f"{attempts} prova | {improvements} permiresime | "
    f"{time.monotonic()-start_time:.1f}s",
    flush=True,
)

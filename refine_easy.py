
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
legs = saved["transfers"]
n = len(legs)

baseline = validate(udp, path, legs)
if not baseline or not baseline["valid"]:
    raise RuntimeError("Seed nuk eshte valid.")

best_days = baseline["fitness"][0]
print(f"Fillimi: {best_days:.6f} dite", flush=True)

# Kohet e nisjes dhe fluturimit
x0 = np.array(
    [leg[2] for leg in legs] + [leg[3] for leg in legs],
    dtype=float,
)

# Ruajme te njejtin buxhet perjashtimesh Delta-V
exception_indices = {
    i for i, leg in enumerate(legs)
    if leg[4] > udp.dv_threshold
}

limits = np.array([
    udp.dv_threshold_exception if i in exception_indices
    else udp.dv_threshold
    for i in range(n)
])

def objective(x):
    return x[n - 1] + x[-1]

def constraints(x):
    departures = x[:n]
    flights = x[n:]
    values = []

    # Transferimet duhet te jene kronologjike
    values.extend(
        departures[1:] - departures[:-1] - flights[:-1]
    )

    # Kufizimet e Delta-V
    for i, (a, b, *_rest) in enumerate(legs):
        try:
            dv = udp.compute_transfer(
                a, b, float(departures[i]), float(flights[i])
            )
            values.append(limits[i] - dv)
        except Exception:
            values.append(-1e9)

    return np.asarray(values, dtype=float)

bounds = (
    [(0.0, udp.max_time)] * n
    + [(udp.min_tof, udp.max_time)] * n
)

def save_solution(x, checked):
    updated = []
    for i, leg in enumerate(legs):
        departure = float(x[i])
        flight = float(x[n + i])
        dv = float(udp.compute_transfer(
            leg[0], leg[1], departure, flight
        ))
        updated.append([
            leg[0], leg[1], departure, flight, dv
        ])

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = ROOT / "results/esa" / f"easy_refined_{stamp}.json"
    destination.write_text(
        json.dumps({
            "path": path,
            "transfers": updated,
            "arrival_day": checked["fitness"][0],
            "validation": checked
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Ruajtur: {destination}", flush=True)

print("Duke optimizuar...", flush=True)
start = time.monotonic()

result = minimize(
    objective,
    x0,
    method="SLSQP",
    bounds=bounds,
    constraints={"type": "ineq", "fun": constraints},
    options={"maxiter": 30, "ftol": 1e-6, "disp": True},
)

candidate_legs = []
for i, leg in enumerate(legs):
    departure = float(result.x[i])
    flight = float(result.x[n + i])
    dv = float(udp.compute_transfer(
        leg[0], leg[1], departure, flight
    ))
    candidate_legs.append([
        leg[0], leg[1], departure, flight, dv
    ])

checked = validate(udp, path, candidate_legs)
print(f"Koha: {time.monotonic() - start:.1f} sekonda")
print(f"Rezultati: {checked}", flush=True)

if checked and checked["valid"] and checked["fitness"][0] < best_days:
    save_solution(result.x, checked)
    print("PERMIRESIM!", flush=True)
else:
    print("Nuk ka permiresim te verifikuar.", flush=True)

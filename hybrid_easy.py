
import json
import random
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from scipy.optimize import minimize

from kttsp_esa_solver import load_evaluator, validate

ROOT = Path(__file__).resolve().parent
SEED = ROOT / "results/esa/easy_refined_20261121T091447Z.json"
OUT = ROOT / "results/esa"
OUT.mkdir(parents=True, exist_ok=True)

udp = load_evaluator(ROOT / "esa_evaluation", "small", False)
saved = json.loads(SEED.read_text())

best_path = list(saved["path"])
best_legs = [list(x) for x in saved["transfers"]]

baseline = validate(udp, best_path, best_legs)
assert baseline and baseline["valid"]

best_days = float(baseline["fitness"][0])
rng = random.Random(2026)

started = time.monotonic()
deadline = started + 180

attempts = 0
valid_count = 0
improvements = 0

NORMAL = float(udp.dv_threshold)
SPECIAL = float(udp.dv_threshold_exception)
MAX_EXCEPTIONS = int(udp.n_dv_exception)


def save(path, legs, checked):
    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    filename = OUT / f"easy_hybrid_{stamp}.json"
    filename.write_text(
        json.dumps({
            "path": path,
            "transfers": legs,
            "arrival_day": checked["fitness"][0],
            "validation": checked,
        }, indent=2)
    )
    print("RUAJTUR:", filename.name, flush=True)


def optimize_segment(path, original, left, right):
    first = max(0, left - 1)
    last = min(len(original) - 1, right)

    prefix = original[:first]
    suffix = original[last + 1:]

    earliest = (
        0.0 if not prefix
        else prefix[-1][2] + prefix[-1][3]
    )

    latest = (
        float(udp.max_time) if not suffix
        else suffix[0][2]
    )

    count = last - first + 1
    pairs = [
        (path[i], path[i + 1])
        for i in range(first, last + 1)
    ]

    outside_exceptions = sum(
        leg[4] > NORMAL
        for leg in prefix + suffix
    )
    available = MAX_EXCEPTIONS - outside_exceptions

    if available < 0 or latest <= earliest:
        return None

    old = original[first:last + 1]
    dep0 = np.array([x[2] for x in old])
    tof0 = np.array([x[3] for x in old])

    x0 = np.concatenate([dep0, tof0])

    # Per transferimet e reja provojme fillimisht
    # kufirin normal; perjashtimet i ndajme
    # sipas transferimeve te vjetra.
    limits = np.array([
        SPECIAL if old[i][4] > NORMAL else NORMAL
        for i in range(count)
    ])

    # Nuk lejojme me shume perjashtime
    # sesa buxheti i mbetur.
    if sum(limits > NORMAL) > available:
        return None

    def objective(x):
        return float(x[count - 1] + x[-1])

    def constraints(x):
        dep = x[:count]
        tof = x[count:]
        values = [dep[0] - earliest]

        for i in range(count - 1):
            values.append(
                dep[i + 1] - dep[i] - tof[i]
            )

        values.append(
            latest - dep[-1] - tof[-1]
        )

        for i, (a, b) in enumerate(pairs):
            try:
                dv = float(
                    udp.compute_transfer(
                        a, b,
                        float(dep[i]),
                        float(tof[i]),
                    )
                )
                values.append(limits[i] - dv)
            except Exception:
                values.append(-1e9)

        return np.asarray(values)

    bounds = (
        [(earliest, latest)] * count
        + [(udp.min_tof, max(
            udp.min_tof, latest - earliest
        ))] * count
    )

    try:
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints={
                "type": "ineq",
                "fun": constraints,
            },
            options={
                "maxiter": 35,
                "ftol": 1e-5,
                "disp": False,
            },
        )
    except Exception:
        return None

    x = result.x
    replacement = []

    for i, (a, b) in enumerate(pairs):
        dep = float(x[i])
        tof = float(x[count + i])

        try:
            dv = float(
                udp.compute_transfer(a, b, dep, tof)
            )
        except Exception:
            return None

        replacement.append([a, b, dep, tof, dv])

    candidate = prefix + replacement + suffix
    checked = validate(udp, path, candidate)

    if checked and checked["valid"]:
        return candidate, checked

    return None


print(f"START: {best_days:.6f} dite", flush=True)

while time.monotonic() < deadline:
    attempts += 1

    path = best_path[:]

    left = rng.randint(1, len(path) - 5)
    right = min(
        len(path) - 2,
        left + rng.randint(1, 3)
    )

    if rng.random() < 0.5:
        path[left:right + 1] = reversed(
            path[left:right + 1]
        )
    else:
        block = path[left:right + 1]
        rng.shuffle(block)
        path[left:right + 1] = block

    if path == best_path:
        continue

    result = optimize_segment(
        path, best_legs, left, right
    )

    if result is not None:
        valid_count += 1
        legs, checked = result
        days = float(checked["fitness"][0])

        print(
            f"VALID: {days:.6f} dite",
            flush=True,
        )

        if days < best_days - 1e-8:
            best_days = days
            best_path = path[:]
            best_legs = [list(x) for x in legs]
            improvements += 1

            save(best_path, best_legs, checked)

            print(
                f"PERMIRESIM: {best_days:.6f}",
                flush=True,
            )

    if attempts % 10 == 0:
        print(
            f"Prova: {attempts} | "
            f"Valide: {valid_count} | "
            f"BEST: {best_days:.6f}",
            flush=True,
        )

print("\n========== REZULTATI ==========")
print("Prova:", attempts)
print("Valide:", valid_count)
print("Permiresime:", improvements)
print(f"BEST: {best_days:.6f} dite")
print(f"Koha: {time.monotonic() - started:.1f}s")


import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from kttsp_esa_solver import load_evaluator, validate

ROOT = Path(__file__).resolve().parent
SEED = ROOT / "results/esa/easy_refined_20261121T091447Z.json"
OUT = ROOT / "results/esa"
OUT.mkdir(parents=True, exist_ok=True)

udp = load_evaluator(ROOT / "esa_evaluation", "small", False)
saved = json.loads(SEED.read_text(encoding="utf-8"))

best_path = list(saved["path"])
best_legs = [list(x) for x in saved["transfers"]]

check = validate(udp, best_path, best_legs)
if not check or not check["valid"]:
    raise RuntimeError("Seed nuk eshte valid!")

best_days = float(check["fitness"][0])

rng = random.Random(42)
start_time = time.monotonic()
deadline = start_time + 180

attempts = 0
valid_count = 0
improvements = 0

print(f"START: {best_days:.6f} dite", flush=True)


def save_solution(path, legs, checked):
    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    filename = OUT / f"easy_global_{stamp}.json"

    filename.write_text(
        json.dumps(
            {
                "path": path,
                "transfers": legs,
                "arrival_day": checked["fitness"][0],
                "validation": checked,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"RUAJTUR: {filename.name}", flush=True)


def mutate_path(path):
    result = path[:]
    n = len(result)

    i = rng.randrange(1, n - 3)
    j = min(n - 2, i + rng.randint(2, 6))

    if rng.random() < 0.5:
        result[i], result[j] = result[j], result[i]
    else:
        result[i:j + 1] = reversed(result[i:j + 1])

    return result, i


def rebuild_schedule(path, old_legs, changed_at):
    legs = [list(x) for x in old_legs[:changed_at - 1]]

    current_day = (
        float(udp.t0.mjd2000)
        if not legs
        else legs[-1][2] + legs[-1][3]
    )

    exceptions = sum(
        leg[4] > udp.dv_threshold for leg in legs
    )

    for index in range(len(legs), len(path) - 1):
        if time.monotonic() >= deadline:
            return None

        a = path[index]
        b = path[index + 1]

        best_leg = None
        best_arrival = float("inf")

        # Kerkojme transferime te reja.
        for wait in (0, 0.125, 0.25, 0.5, 1.0):
            departure = current_day + wait

            for tof in (
                0.0625, 0.125, 0.25, 0.5,
                0.75, 1.0, 1.5, 2.0,
                3.0, 4.0, 6.0,
            ):
                arrival = departure + tof

                if arrival >= best_arrival:
                    continue

                if arrival > udp.max_time:
                    continue

                try:
                    dv = float(
                        udp.compute_transfer(
                            a, b, departure, tof
                        )
                    )
                except Exception:
                    continue

                if not math.isfinite(dv):
                    continue

                allowed = (
                    udp.dv_threshold_exception
                    if exceptions < udp.n_dv_exception
                    else udp.dv_threshold
                )

                if dv > allowed:
                    continue

                if (
                    dv > udp.dv_threshold
                    and exceptions >= udp.n_dv_exception
                ):
                    continue

                best_leg = [
                    a, b, departure, tof, dv
                ]
                best_arrival = arrival

        if best_leg is None:
            return None

        legs.append(best_leg)
        current_day = best_arrival

        if best_leg[4] > udp.dv_threshold:
            exceptions += 1

    return legs


def optimize_times(path, legs):
    # Optimizojme kohen e nisjes dhe fluturimit
    # ne segmentin e fundit te rruges.
    count = min(5, len(legs))
    offset = len(legs) - count

    prefix = [list(x) for x in legs[:offset]]
    suffix = [list(x) for x in legs[offset:]]

    earliest = (
        float(udp.t0.mjd2000)
        if not prefix
        else prefix[-1][2] + prefix[-1][3]
    )

    x0 = []

    for leg in suffix:
        x0.extend([leg[2], leg[3]])

    def objective(x):
        penalty = 0.0
        previous = earliest
        used = sum(
            leg[4] > udp.dv_threshold
            for leg in prefix
        )

        for k in range(count):
            dep = float(x[2 * k])
            tof = float(x[2 * k + 1])

            if dep < previous:
                penalty += 10000 * (previous - dep)

            if tof < udp.min_tof:
                penalty += 10000 * (udp.min_tof - tof)

            try:
                dv = float(
                    udp.compute_transfer(
                        suffix[k][0],
                        suffix[k][1],
                        dep,
                        tof,
                    )
                )
            except Exception:
                return 1e9

            if not math.isfinite(dv):
                return 1e9

            if dv > udp.dv_threshold_exception:
                penalty += (
                    100 * (dv - udp.dv_threshold_exception)
                )
            elif dv > udp.dv_threshold:
                used += 1

            previous = dep + tof

        if used > udp.n_dv_exception:
            penalty += 10000 * (
                used - udp.n_dv_exception
            )

        return previous + penalty

    bounds = []

    for leg in suffix:
        bounds.append(
            (earliest, udp.max_time)
        )
        bounds.append(
            (udp.min_tof, 10.0)
        )

    result = minimize(
        objective,
        np.asarray(x0, dtype=float),
        method="Powell",
        bounds=bounds,
        options={
            "maxiter": 25,
            "maxfev": 400,
            "xtol": 1e-3,
            "ftol": 1e-4,
        },
    )

    x = result.x
    new_legs = prefix[:]

    for k, old in enumerate(suffix):
        dep = float(x[2 * k])
        tof = float(x[2 * k + 1])

        try:
            dv = float(
                udp.compute_transfer(
                    old[0], old[1], dep, tof
                )
            )
        except Exception:
            return None

        new_legs.append(
            [old[0], old[1], dep, tof, dv]
        )

    return new_legs


while time.monotonic() < deadline:
    attempts += 1

    path, changed_at = mutate_path(best_path)

    legs = rebuild_schedule(
        path,
        best_legs,
        changed_at,
    )

    if legs is None:
        continue

    checked = validate(udp, path, legs)

    if not checked or not checked["valid"]:
        continue

    valid_count += 1
    days = checked["fitness"][0]

    print(
        f"VALID: {days:.6f} dite",
        flush=True,
    )

    # Provojme optimizimin e koheve.
    if time.monotonic() < deadline:
        refined = optimize_times(path, legs)

        if refined is not None:
            refined_check = validate(
                udp, path, refined
            )

            if (
                refined_check
                and refined_check["valid"]
                and refined_check["fitness"][0] < days
            ):
                legs = refined
                checked = refined_check
                days = checked["fitness"][0]

    if days < best_days - 1e-8:
        best_days = days
        best_path = path[:]
        best_legs = [list(x) for x in legs]
        improvements += 1

        print(
            f"PERMIRESIM: {best_days:.6f} dite!",
            flush=True,
        )

        save_solution(
            best_path, best_legs, checked
        )

    if attempts % 10 == 0:
        print(
            f"Prova: {attempts} | "
            f"Valide: {valid_count} | "
            f"Best: {best_days:.6f}",
            flush=True,
        )

print()
print("========== REZULTATI ==========")
print(f"Prova: {attempts}")
print(f"Valide: {valid_count}")
print(f"Permiresime: {improvements}")
print(f"BEST: {best_days:.6f} dite")
print(
    f"Koha: {time.monotonic() - start_time:.1f}s"
)


import argparse
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
DEFAULT_SEED = ROOT / "results/esa/easy_refined_20261121T091447Z.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--seed-file", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds duhet te jete pozitiv.")

    rng = random.Random(args.random_seed)
    udp = load_evaluator(ROOT / "esa_evaluation", "small", False)

    saved = json.loads(args.seed_file.read_text(encoding="utf-8"))
    path = saved["path"]
    legs = saved["transfers"]
    n = len(legs)

    baseline = validate(udp, path, legs)
    if not baseline or not baseline["valid"]:
        raise RuntimeError("Seed nuk eshte valid.")

    best_days = baseline["fitness"][0]
    best_legs = legs
    print(f"Fillimi: {best_days:.6f} dite", flush=True)

    output = ROOT / "results/esa"
    output.mkdir(parents=True, exist_ok=True)

    # Mbajme te njejtat 5 transferime me perjashtim.
    exception_indices = {
        i for i, leg in enumerate(legs)
        if leg[4] > udp.dv_threshold
    }

    limits = np.array([
        udp.dv_threshold_exception if i in exception_indices
        else udp.dv_threshold
        for i in range(n)
    ])

    bounds = (
        [(0.0, udp.max_time)] * n
        + [(udp.min_tof, udp.max_time)] * n
    )

    def objective(x):
        return float(x[n - 1] + x[-1])

    def constraints(x):
        departures = x[:n]
        flights = x[n:]

        chronological = (
            departures[1:] - departures[:-1] - flights[:-1]
        )

        dv_constraints = []
        for i, leg in enumerate(legs):
            try:
                dv = udp.compute_transfer(
                    leg[0],
                    leg[1],
                    float(departures[i]),
                    float(flights[i]),
                )
                dv_constraints.append(limits[i] - dv)
            except Exception:
                dv_constraints.append(-1e9)

        return np.concatenate([
            chronological,
            np.array(dv_constraints, dtype=float),
        ])

    def vector_from_legs(items):
        return np.array(
            [leg[2] for leg in items]
            + [leg[3] for leg in items],
            dtype=float,
        )

    def perturb(items):
        """Ndryshon pak oraret, pa ndryshuar renditjen."""
        x = vector_from_legs(items)
        departures = x[:n]
        flights = x[n:]

        # Ndryshojme nje segment te shkurter.
        start = rng.randrange(n)
        stop = min(n, start + rng.randint(1, 5))

        shift = rng.uniform(-0.25, 0.25)
        departures[start:stop] += shift

        for i in range(start, stop):
            flights[i] += rng.uniform(-0.10, 0.10)

        return np.clip(
            x,
            [b[0] for b in bounds],
            [b[1] for b in bounds],
        )

    started = time.monotonic()
    deadline = started + args.seconds
    attempts = 0
    improvements = 0

    while time.monotonic() < deadline:
        attempts += 1

        if attempts == 1:
            x0 = vector_from_legs(best_legs)
        else:
            x0 = perturb(best_legs)

        # SLSQP optimizon kohen me kufizimet e ESA-s.
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints={"type": "ineq", "fun": constraints},
            options={
                "maxiter": 30,
                "ftol": 1e-7,
                "disp": False,
            },
        )

        candidate = []
        for i, leg in enumerate(legs):
            departure = float(result.x[i])
            flight = float(result.x[n + i])
            dv = float(udp.compute_transfer(
                leg[0], leg[1], departure, flight
            ))
            candidate.append([
                leg[0], leg[1], departure, flight, dv
            ])

        checked = validate(udp, path, candidate)

        if checked and checked["valid"]:
            days = checked["fitness"][0]

            if days < best_days - 1e-8:
                best_days = days
                best_legs = candidate
                improvements += 1

                stamp = datetime.now(timezone.utc).strftime(
                    "%Y%m%dT%H%M%SZ"
                )
                filename = (
                    output
                    / f"easy_multistart_{stamp}_{attempts}.json"
                )

                filename.write_text(
                    json.dumps({
                        "path": path,
                        "transfers": best_legs,
                        "arrival_day": best_days,
                        "validation": checked,
                        "attempt": attempts,
                    }, indent=2),
                    encoding="utf-8",
                )

                print(
                    f"PERMIRESIM! {best_days:.6f} dite "
                    f"| Prova {attempts}",
                    flush=True,
                )

        print(
            f"Prova {attempts} | "
            f"Me i miri: {best_days:.6f} dite | "
            f"Koha: {time.monotonic() - started:.1f}s",
            flush=True,
        )

    print(
        f"\nPERFUNDIM: {best_days:.6f} dite | "
        f"{attempts} prova | {improvements} permiresime",
        flush=True,
    )


if __name__ == "__main__":
    main()

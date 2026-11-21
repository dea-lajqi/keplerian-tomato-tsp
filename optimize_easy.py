
import argparse
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from kttsp_esa_solver import load_evaluator, validate, retime

ROOT = Path(__file__).resolve().parent
SEED = ROOT / "results/esa/easy_20261120T080457Z_b0dd1006_8_verified.json"


def make_legs(path, original_legs):
    """Krijon transferimet fillestare për një renditje të re."""
    old = {(leg[0], leg[1]): leg for leg in original_legs}
    result = []

    for a, b in zip(path, path[1:]):
        if (a, b) in old:
            result.append(old[(a, b)])
        else:
            result.append((a, b, 0.0, 1.0, 0.0))

    return result


def mutate(path, rng):
    """Ndryshon renditjen e domateve."""
    candidate = path.copy()
    i, j = sorted(rng.sample(range(len(path)), 2))

    if rng.random() < 0.5:
        candidate[i], candidate[j] = candidate[j], candidate[i]
    else:
        candidate[i:j + 1] = reversed(candidate[i:j + 1])

    return candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds duhet te jete pozitiv.")

    rng = random.Random(args.seed)
    udp = load_evaluator(ROOT / "esa_evaluation", "small", False)

    saved = json.loads(SEED.read_text(encoding="utf-8"))
    best_path = saved["path"]
    best_legs = saved["transfers"]

    baseline = validate(udp, best_path, best_legs)
    if not baseline or not baseline["valid"]:
        raise RuntimeError("Zgjidhja fillestare nuk eshte valide.")

    best_days = baseline["fitness"][0]
    print(f"Fillimi: {best_days} dite", flush=True)

    output = ROOT / "results/esa"
    output.mkdir(parents=True, exist_ok=True)

    deadline = time.monotonic() + args.seconds
    attempts = 0

    while time.monotonic() < deadline:
        attempts += 1

        candidate_path = mutate(best_path, rng)
        initial_legs = make_legs(candidate_path, best_legs)

        attempt_deadline = min(deadline, time.monotonic() + 2.0)

        candidate_legs = retime(
            udp,
            initial_legs,
            0.25,
            attempt_deadline,
            False,
        )

        if len(candidate_legs) != len(candidate_path) - 1:
            continue

        checked = validate(udp, candidate_path, candidate_legs)

        if checked and checked["valid"]:
            days = checked["fitness"][0]

            if days < best_days:
                best_days = days
                best_path = candidate_path
                best_legs = candidate_legs

                stamp = datetime.now(timezone.utc).strftime(
                    "%Y%m%dT%H%M%SZ"
                )
                filename = output / f"easy_optimized_{stamp}_{attempts}.json"

                filename.write_text(
                    json.dumps(
                        {
                            "path": best_path,
                            "transfers": best_legs,
                            "arrival_day": best_days,
                            "validation": checked,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                print(
                    f"PERMIRESIM! {best_days:.4f} dite | "
                    f"Prova {attempts}",
                    flush=True,
                )

        if attempts % 10 == 0:
            print(
                f"Prova {attempts} | Me i miri: {best_days:.4f} dite",
                flush=True,
            )

    print(f"PERFUNDIM: {best_days:.4f} dite | {attempts} prova")


if __name__ == "__main__":
    main()

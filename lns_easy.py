
import argparse
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from kttsp_esa_solver import load_evaluator, validate, try_leg

ROOT = Path(__file__).resolve().parent
SEED = ROOT / "results/esa/easy_refined_20261121T091447Z.json"


def change_order(path, rng):
    """Shkemben ose zhvendos nje domate."""
    new = path.copy()
    i, j = rng.sample(range(len(new)), 2)

    if rng.random() < 0.5:
        new[i], new[j] = new[j], new[i]
    else:
        tomato = new.pop(i)
        new.insert(j, tomato)

    return new


def rebuild(udp, path, old_legs, deadline, step=0.25):
    """Rinderton orarin duke perdorur transferimet ekzistuese."""
    old = {(a, b): leg for a, b, *_, in [] for leg in []}
    old = {(leg[0], leg[1]): leg for leg in old_legs}

    result = []
    day = float(udp.t0.mjd2000)
    exceptions = 0

    for a, b in zip(path, path[1:]):
        if time.monotonic() >= deadline:
            return None

        previous = old.get((a, b), (a, b, day, 1.0, 0.0))

        # Fillimisht provojme kufirin normal.
        leg = try_leg(
            udp, previous, day,
            udp.dv_threshold,
            step, deadline, True
        )

        # Perjashtim vetem nese eshte i nevojshem.
        if leg is None and exceptions < udp.n_dv_exception:
            leg = try_leg(
                udp, previous, day,
                udp.dv_threshold_exception,
                step, deadline, True
            )

        if leg is None:
            return None

        exceptions += leg[4] > udp.dv_threshold
        result.append(leg)
        day = leg[2] + leg[3]

        if day >= udp.max_time:
            return None

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--seed-file", type=Path, default=SEED)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds duhet te jete pozitiv.")

    rng = random.Random(args.random_seed)
    udp = load_evaluator(ROOT / "esa_evaluation", "small", False)

    saved = json.loads(args.seed_file.read_text(encoding="utf-8"))
    best_path = saved["path"]
    best_legs = saved["transfers"]

    baseline = validate(udp, best_path, best_legs)
    if not baseline or not baseline["valid"]:
        raise RuntimeError("Seed nuk eshte valid.")

    best_days = baseline["fitness"][0]
    print(f"Fillimi: {best_days:.6f} dite", flush=True)

    output = ROOT / "results/esa"
    output.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    deadline = started + args.seconds
    attempts = 0
    valid_count = 0

    while time.monotonic() < deadline:
        attempts += 1
        candidate_path = change_order(best_path, rng)

        attempt_deadline = min(
            deadline, time.monotonic() + 3.0
        )

        candidate_legs = rebuild(
            udp, candidate_path, best_legs,
            attempt_deadline
        )

        if candidate_legs is not None:
            checked = validate(udp, candidate_path, candidate_legs)

            if checked and checked["valid"]:
                valid_count += 1
                days = checked["fitness"][0]

                if days < best_days - 1e-8:
                    best_days = days
                    best_path = candidate_path
                    best_legs = candidate_legs

                    stamp = datetime.now(timezone.utc).strftime(
                        "%Y%m%dT%H%M%SZ"
                    )
                    filename = output / (
                        f"easy_lns_{stamp}_{attempts}.json"
                    )

                    filename.write_text(
                        json.dumps({
                            "path": best_path,
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

        if attempts % 10 == 0:
            print(
                f"Prova {attempts} | "
                f"Valide: {valid_count} | "
                f"Me i miri: {best_days:.6f} dite | "
                f"Koha: {time.monotonic()-started:.1f}s",
                flush=True,
            )

    print(
        f"\nPERFUNDIM: {best_days:.6f} dite | "
        f"{attempts} prova | {valid_count} valide",
        flush=True,
    )


if __name__ == "__main__":
    main()

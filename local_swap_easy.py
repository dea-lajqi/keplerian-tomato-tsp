
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from kttsp_esa_solver import load_evaluator, validate

ROOT = Path(__file__).resolve().parent
SEED = ROOT / "results/esa/easy_refined_20261121T091447Z.json"
OUTPUT = ROOT / "results/esa"
VALID_DIR = OUTPUT / "valid_candidates"

OUTPUT.mkdir(parents=True, exist_ok=True)
VALID_DIR.mkdir(parents=True, exist_ok=True)

udp = load_evaluator(ROOT / "esa_evaluation", "small", False)
saved = json.loads(SEED.read_text(encoding="utf-8"))

best_path = saved["path"][:]
best_legs = [list(leg) for leg in saved["transfers"]]

baseline = validate(udp, best_path, best_legs)

if not baseline or not baseline["valid"]:
    raise RuntimeError("Zgjidhja fillestare nuk eshte valide!")

best_days = baseline["fitness"][0]

current_path = best_path[:]
current_legs = [leg.copy() for leg in best_legs]
current_days = best_days

rng = random.Random()
started = time.monotonic()
deadline = started + 120

attempts = 0
valid_count = 0
improvements = 0

print(f"Fillimi: {best_days:.6f} dite", flush=True)


def save_candidate(path, legs, checked, attempt):
    days = checked["fitness"][0]

    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    filename = VALID_DIR / (
        f"easy_valid_{stamp}_{attempt}.json"
    )

    filename.write_text(
        json.dumps(
            {
                "path": path,
                "transfers": legs,
                "arrival_day": days,
                "validation": checked,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"VALID! {days:.6f} dite | "
        f"Ruajtur: {filename.name}",
        flush=True,
    )


def possible_legs(a, b, earliest, latest, attempt_deadline):
    options = []
    seen = set()

    for step in (0.25, 0.0625, 0.015625):
        start = math.ceil(max(0.0, earliest) / step)
        end = math.floor(latest / step)

        for tick in range(start, end + 1):
            if time.monotonic() >= attempt_deadline:
                break

            departure = tick * step

            for flight_ticks in range(1, int(3 / step) + 1):
                tof = flight_ticks * step

                if departure + tof > latest + 1e-9:
                    break

                if tof < udp.min_tof:
                    continue

                key = (
                    round(departure, 7),
                    round(tof, 7),
                )

                if key in seen:
                    continue

                seen.add(key)

                try:
                    dv = float(
                        udp.compute_transfer(a, b, departure, tof)
                    )
                except Exception:
                    continue

                if (
                    math.isfinite(dv)
                    and dv <= udp.dv_threshold_exception
                ):
                    options.append(
                        [a, b, departure, tof, dv]
                    )

        if time.monotonic() >= attempt_deadline:
            break

    options.sort(
        key=lambda leg: (
            leg[2] + leg[3],
            leg[4] > udp.dv_threshold,
        )
    )

    return options


def repair_swap(path, legs, position, attempt_deadline):
    new_path = path[:]

    new_path[position], new_path[position + 1] = (
        new_path[position + 1],
        new_path[position],
    )

    left = max(0, position - 1)
    right = min(len(legs) - 1, position + 1)

    earliest = (
        float(udp.t0.mjd2000)
        if left == 0
        else legs[left - 1][2] + legs[left - 1][3]
    )

    latest = (
        udp.max_time
        if right == len(legs) - 1
        else legs[right + 1][2]
    )

    if latest <= earliest:
        return None

    outside_exceptions = sum(
        leg[4] > udp.dv_threshold
        for i, leg in enumerate(legs)
        if i < left or i > right
    )

    budget = udp.n_dv_exception - outside_exceptions

    if budget < 0:
        return None

    beam = [(earliest, 0, [])]

    for i in range(left, right + 1):
        if time.monotonic() >= attempt_deadline:
            return None

        a = new_path[i]
        b = new_path[i + 1]
        candidates = []

        for current_day, used, sequence in beam:
            options = possible_legs(
                a,
                b,
                current_day,
                latest,
                attempt_deadline,
            )

            for leg in options:
                new_used = used + int(
                    leg[4] > udp.dv_threshold
                )

                if new_used > budget:
                    continue

                arrival = leg[2] + leg[3]

                candidates.append(
                    (
                        arrival,
                        new_used,
                        sequence + [leg],
                    )
                )

        if not candidates:
            return None

        candidates.sort(
            key=lambda x: (x[0], x[1])
        )

        beam = []
        seen = set()

        for item in candidates:
            key = (
                round(item[0], 2),
                item[1],
            )

            if key not in seen:
                beam.append(item)
                seen.add(key)

            if len(beam) >= 12:
                break

    # Kontrollojme te gjitha alternativat
    # e mbetura dhe zgjedhim me te shpejten valide.
    best_result = None
    best_candidate_days = float("inf")

    for arrival, used, replacement in beam:
        candidate_legs = (
            legs[:left]
            + replacement
            + legs[right + 1:]
        )

        checked = validate(
            udp,
            new_path,
            candidate_legs,
        )

        if checked and checked["valid"]:
            days = checked["fitness"][0]

            if days < best_candidate_days:
                best_candidate_days = days
                best_result = (
                    new_path,
                    candidate_legs,
                    checked,
                )

    return best_result


while time.monotonic() < deadline:
    attempts += 1

    # Provojme fillimisht pozicionet prane
    # transferimeve me Delta-V te larte.
    ranked_positions = sorted(
        range(len(current_path) - 1),
        key=lambda i: max(
            current_legs[j][4]
            for j in range(
                max(0, i - 1),
                min(len(current_legs), i + 2),
            )
        ),
        reverse=True,
    )

    position = ranked_positions[
        (attempts - 1) % len(ranked_positions)
    ]

    attempt_deadline = min(
        deadline,
        time.monotonic() + 5.0,
    )

    result = repair_swap(
        current_path,
        current_legs,
        position,
        attempt_deadline,
    )

    if result is not None:
        valid_count += 1

        path, legs, checked = result
        days = checked["fitness"][0]

        # Ruajme CDO kandidat valid.
        save_candidate(
            path,
            legs,
            checked,
            attempts,
        )

        # Simulated annealing:
        # mund te vazhdojme edhe nga nje
        # kandidat pak me i ngadalshem.
        temperature = 2.0

        probability = math.exp(
            min(
                0.0,
                (current_days - days) / temperature,
            )
        )

        if (
            days <= current_days
            or rng.random() < probability
        ):
            current_path = path[:]
            current_legs = [
                leg.copy() for leg in legs
            ]
            current_days = days

        # Ruajme vetem rekordin me te mire.
        if days < best_days - 1e-8:
            best_days = days
            best_path = path[:]
            best_legs = [
                leg.copy() for leg in legs
            ]
            improvements += 1

            stamp = datetime.now(
                timezone.utc
            ).strftime("%Y%m%dT%H%M%S%fZ")

            filename = OUTPUT / (
                f"easy_improved_{stamp}.json"
            )

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
                f"PERMIRESIM! "
                f"{best_days:.6f} dite",
                flush=True,
            )

    if attempts % 10 == 0:
        print(
            f"Prova {attempts} | "
            f"Valide: {valid_count} | "
            f"Me i miri: {best_days:.6f}",
            flush=True,
        )

print(
    f"PERFUNDIM: "
    f"{best_days:.6f} dite | "
    f"{attempts} prova | "
    f"{valid_count} valide | "
    f"{improvements} permiresime | "
    f"{time.monotonic() - started:.1f}s",
    flush=True,
)

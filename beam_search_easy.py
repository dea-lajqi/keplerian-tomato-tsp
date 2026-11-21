
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from kttsp_esa_solver import load_evaluator, validate

ROOT = Path(__file__).resolve().parent
SEED = ROOT / "results/esa/easy_refined_20261121T091447Z.json"
OUT = ROOT / "results/esa"
OUT.mkdir(parents=True, exist_ok=True)

udp = load_evaluator(ROOT / "esa_evaluation", "small", False)
data = json.loads(SEED.read_text(encoding="utf-8"))

best_path = list(data["path"])
best_legs = [list(x) for x in data["transfers"]]
check = validate(udp, best_path, best_legs)

if not check or not check["valid"]:
    raise RuntimeError("Seed nuk eshte valid!")

best_days = float(check["fitness"][0])

rng = random.Random(2026)
deadline = time.monotonic() + 180
start_time = time.monotonic()

attempts = 0
valid_count = 0
improvements = 0

DV_NORMAL = float(udp.dv_threshold)
DV_SPECIAL = float(udp.dv_threshold_exception)
MAX_SPECIAL = int(udp.n_dv_exception)
MIN_TOF = float(udp.min_tof)


def save(path, legs, checked):
    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    file = OUT / f"easy_beam_{stamp}.json"
    file.write_text(
        json.dumps({
            "path": path,
            "transfers": legs,
            "arrival_day": checked["fitness"][0],
            "validation": checked,
        }, indent=2),
        encoding="utf-8",
    )
    print("RUAJTUR:", file.name, flush=True)


def mutate(path):
    new = path[:]
    n = len(new)

    left = rng.randint(1, n - 7)
    length = rng.randint(3, 6)
    right = left + length - 1

    operation = rng.choice(["reverse", "rotate", "shuffle"])

    if operation == "reverse":
        new[left:right + 1] = reversed(new[left:right + 1])

    elif operation == "rotate":
        block = new[left:right + 1]
        new[left:right + 1] = block[1:] + block[:1]

    else:
        block = new[left:right + 1]
        rng.shuffle(block)
        new[left:right + 1] = block

    return new, left, right


def leg_options(a, b, earliest, latest, max_options=8):
    options = []
    seen = set()

    # Nisje dhe fluturime me hapa te ndryshem.
    waits = [
        0.0, 0.0625, 0.125, 0.25,
        0.5, 0.75, 1.0, 1.5,
    ]
    flights = [
        0.0625, 0.125, 0.25, 0.5,
        0.75, 1.0, 1.5, 2.0,
        2.5, 3.0, 4.0, 5.0,
        6.0, 8.0,
    ]

    for wait in waits:
        dep = earliest + wait

        for tof in flights:
            if tof < MIN_TOF:
                continue

            arrival = dep + tof
            if arrival > latest:
                continue

            key = (round(dep, 6), round(tof, 6))
            if key in seen:
                continue
            seen.add(key)

            try:
                dv = float(udp.compute_transfer(a, b, dep, tof))
            except Exception:
                continue

            if not math.isfinite(dv) or dv > DV_SPECIAL:
                continue

            options.append([
                a, b, dep, tof, dv
            ])

    options.sort(
        key=lambda x: (
            x[2] + x[3],
            x[4] > DV_NORMAL,
        )
    )

    # Mbajme edhe alternativa me Delta-V normal.
    normal = [x for x in options if x[4] <= DV_NORMAL]
    special = [x for x in options if x[4] > DV_NORMAL]

    return normal[:max_options] + special[:3]


def repair(path, original_legs, left, right):
    # Ndryshojne lidhjet nga left-1 deri te right.
    first = left - 1
    last = right

    prefix = [
        list(x) for x in original_legs[:first]
    ]
    suffix = [
        list(x) for x in original_legs[last + 1:]
    ]

    earliest = (
        float(udp.t0.mjd2000)
        if not prefix
        else prefix[-1][2] + prefix[-1][3]
    )

    # Duhet te lidhemi me pjesen e pandryshuar.
    latest = (
        float(udp.max_time)
        if not suffix
        else float(suffix[0][2])
    )

    outside_special = sum(
        x[4] > DV_NORMAL
        for x in prefix + suffix
    )
    special_budget = MAX_SPECIAL - outside_special

    if special_budget < 0:
        return None

    beam = [(earliest, 0, [])]

    for i in range(first, last + 1):
        if time.monotonic() >= deadline:
            return None

        a = path[i]
        b = path[i + 1]
        candidates = []

        for day, used, sequence in beam:
            options = leg_options(
                a, b, day, latest
            )

            for leg in options:
                extra = int(leg[4] > DV_NORMAL)
                if used + extra > special_budget:
                    continue

                arrival = leg[2] + leg[3]

                candidates.append((
                    arrival,
                    used + extra,
                    sequence + [leg],
                ))

        if not candidates:
            return None

        candidates.sort(key=lambda x: (x[0], x[1]))

        beam = []
        seen = set()

        for candidate in candidates:
            key = (
                round(candidate[0], 2),
                candidate[1],
            )

            if key in seen:
                continue

            seen.add(key)
            beam.append(candidate)

            if len(beam) >= 24:
                break

    best_result = None
    best_time = float("inf")

    for day, used, replacement in beam:
        legs = prefix + replacement + suffix
        checked = validate(udp, path, legs)

        if checked and checked["valid"]:
            total = checked["fitness"][0]
            if total < best_time:
                best_time = total
                best_result = (legs, checked)

    return best_result


print(f"START: {best_days:.6f} dite", flush=True)

while time.monotonic() < deadline:
    attempts += 1

    new_path, left, right = mutate(best_path)

    result = repair(
        new_path, best_legs, left, right
    )

    if result is not None:
        valid_count += 1
        legs, checked = result
        days = float(checked["fitness"][0])

        if days < best_days - 1e-8:
            best_days = days
            best_path = new_path[:]
            best_legs = [list(x) for x in legs]
            improvements += 1

            print(
                f"PERMIRESIM: {best_days:.6f} dite",
                flush=True,
            )
            save(best_path, best_legs, checked)

    if attempts % 10 == 0:
        print(
            f"Prova: {attempts} | "
            f"Valide: {valid_count} | "
            f"Permiresime: {improvements} | "
            f"BEST: {best_days:.6f}",
            flush=True,
        )

print("\n========== REZULTATI ==========")
print("Prova:", attempts)
print("Valide:", valid_count)
print("Permiresime:", improvements)
print(f"BEST: {best_days:.6f} dite")
print(f"Koha: {time.monotonic() - start_time:.1f}s")


import json
import random
import time
import math
from pathlib import Path
from datetime import datetime, timezone

from kttsp_esa_solver import load_evaluator, validate

ROOT = Path(__file__).resolve().parent
SEED = ROOT / "results/esa/easy_refined_20261121T091447Z.json"
OUT = ROOT / "results/esa"
OUT.mkdir(parents=True, exist_ok=True)

# Ndryshoje ne 7200 per 2 ore
HOURS = 2
TIME_LIMIT = HOURS * 3600

# Sa alternativa mbajme ne cdo hap
BEAM_WIDTH = 16

# Sa domate lejojme te nderrojne pozicion
WINDOW = 5

rng = random.Random(20261121)

udp = load_evaluator(ROOT / "esa_evaluation", "small", False)
saved = json.loads(SEED.read_text())

seed_path = list(saved["path"])
seed_legs = saved["transfers"]

check = validate(udp, seed_path, seed_legs)
assert check and check["valid"]

best = float(check["fitness"][0])

NORMAL = float(udp.dv_threshold)
SPECIAL = float(udp.dv_threshold_exception)
EXCEPTIONS = int(udp.n_dv_exception)
MIN_TOF = float(udp.min_tof)

start = time.monotonic()
deadline = start + TIME_LIMIT

attempts = 0
complete = 0
valid_count = 0
improvements = 0
transfer_calls = 0

# Cache per llogaritjet e transferimeve
cache = {}


def dv_value(a, b, departure, tof):
    global transfer_calls

    key = (
        a, b,
        round(departure, 5),
        round(tof, 5),
    )

    if key in cache:
        return cache[key]

    transfer_calls += 1

    try:
        value = float(
            udp.compute_transfer(
                a, b, departure, tof
            )
        )
        if not math.isfinite(value):
            value = float("inf")
    except Exception:
        value = float("inf")

    if len(cache) > 500000:
        cache.clear()

    cache[key] = value
    return value


def save(path, legs, checked):
    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    destination = OUT / f"easy_long_{stamp}.json"

    destination.write_text(
        json.dumps({
            "path": path,
            "transfers": legs,
            "arrival_day": checked["fitness"][0],
            "validation": checked,
        }, indent=2),
        encoding="utf-8",
    )

    print("RUAJTUR:", destination, flush=True)


def possible_transfers(a, b, arrival, target_tof):
    """
    Kerkojme nisje dhe kohezgjatje te ndryshme.
    Nuk kerkojme te perputhen me orarin e vjeter.
    """

    waits = [
        0.0, 0.125, 0.25, 0.5,
        0.75, 1.0, 1.5, 2.0,
    ]

    tofs = [
        0.5, 1.0, 1.5, 2.0,
        2.5, 3.0, 3.5, 4.0,
        4.5, 5.0, 6.0, 7.0,
        8.0,
    ]

    if target_tof is not None:
        for delta in [
            -1.0, -0.5, -0.25,
            0.0, 0.25, 0.5, 1.0,
        ]:
            t = target_tof + delta
            if t >= MIN_TOF:
                tofs.append(t)

    results = []

    for wait in waits:
        departure = arrival + wait

        if departure >= best:
            continue

        for tof in tofs:
            if tof < MIN_TOF:
                continue

            finish = departure + tof

            if finish >= best:
                continue

            dv = dv_value(
                a, b, departure, tof
            )

            if dv <= NORMAL:
                results.append((
                    finish,
                    departure,
                    tof,
                    dv,
                    0,
                ))

            elif dv <= SPECIAL:
                results.append((
                    finish,
                    departure,
                    tof,
                    dv,
                    1,
                ))

    results.sort(key=lambda x: (x[0], x[4]))

    # Mbajme alternativa me kohe dhe buxhet te ndryshem
    normal = [x for x in results if x[4] == 0][:3]
    special = [x for x in results if x[4] == 1][:2]

    return normal + special


def search_once():
    """
    Ndertojme rrugen nga fillimi.
    Nje beam-state permban:
    (koha, perjashtimet, path, transfers, remaining)
    """

    states = [(
        0.0,
        0,
        [seed_path[0]],
        [],
        tuple(seed_path[1:]),
    )]

    for depth in range(1, len(seed_path)):

        if time.monotonic() >= deadline:
            return None

        new_states = []

        for arrival, used, path, legs, remaining in states:

            # Zgjedhim nga nje dritare e rruges ekzistuese
            choices = list(remaining[:WINDOW])

            rng.shuffle(choices)

            for b in choices:
                a = path[-1]

                target = (
                    float(seed_legs[depth - 1][3])
                    if depth - 1 < len(seed_legs)
                    else None
                )

                options = possible_transfers(
                    a, b, arrival, target
                )

                for finish, dep, tof, dv, extra in options:
                    if used + extra > EXCEPTIONS:
                        continue

                    new_remaining = tuple(
                        x for x in remaining if x != b
                    )

                    new_states.append((
                        finish,
                        used + extra,
                        path + [b],
                        legs + [[a, b, dep, tof, dv]],
                        new_remaining,
                    ))

        if not new_states:
            return None

        # Ruajme gjendje te ndryshme, jo vetem
        # ato me kohen me te vogel.
        new_states.sort(
            key=lambda s: (
                s[0] + 0.4 * s[1],
                s[1],
            )
        )

        selected = []
        signatures = set()

        for state in new_states:
            signature = (
                state[2][-1],
                state[1],
                round(state[0] * 2),
                state[4][:3],
            )

            if signature in signatures:
                continue

            signatures.add(signature)
            selected.append(state)

            if len(selected) >= BEAM_WIDTH:
                break

        states = selected

    return states


print(f"START BEST: {best:.6f} dite")
print(f"KOHA MAKSIMALE: {HOURS} ore")
print("Kerkohet nje rruge e re valide...", flush=True)

last_report = start

while time.monotonic() < deadline:
    attempts += 1

    states = search_once()

    if states:
        for arrival, used, path, legs, remaining in states:
            if remaining:
                continue

            complete += 1

            checked = validate(udp, path, legs)

            if not checked or not checked["valid"]:
                continue

            valid_count += 1
            days = float(checked["fitness"][0])

            if days < best - 1e-8:
                best = days
                improvements += 1

                save(path, legs, checked)

                print(
                    f"\nPERMIRESIM! {best:.6f} dite",
                    flush=True,
                )

    now = time.monotonic()

    if now - last_report >= 60:
        print(
            f"Koha: {(now-start)/60:.1f} min | "
            f"Prova: {attempts} | "
            f"Rruge te plota: {complete} | "
            f"Valide: {valid_count} | "
            f"BEST: {best:.6f}",
            flush=True,
        )
        last_report = now

print("\n========== REZULTATI ==========")
print("Prova:", attempts)
print("Rruge te plota:", complete)
print("Valide:", valid_count)
print("Permiresime:", improvements)
print(f"BEST: {best:.6f} dite")
print("Transfer calculations:", transfer_calls)
print(f"Koha: {(time.monotonic()-start)/60:.1f} min")

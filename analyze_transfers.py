
import json
import math
import time
from pathlib import Path

from kttsp_esa_solver import load_evaluator

ROOT = Path(__file__).resolve().parent
SEED = ROOT / "results/esa/easy_refined_20261121T091447Z.json"

udp = load_evaluator(ROOT / "esa_evaluation", "small", False)
data = json.loads(SEED.read_text(encoding="utf-8"))

legs = data["transfers"]

# Analizojme 5 transferime te gjata.
TARGETS = [
    (24, 6),
    (46, 41),
    (31, 37),
    (5, 47),
    (40, 44),
]

DV_LIMIT = float(udp.dv_threshold)

print("=" * 65)
print("ANALIZA E TRANSFERIMEVE")
print("=" * 65)

start = time.monotonic()

for a, b in TARGETS:
    original = next(
        (leg for leg in legs if leg[0] == a and leg[1] == b),
        None,
    )

    if original is None:
        print(f"\n{a} -> {b}: nuk u gjet")
        continue

    departure = float(original[2])
    tof = float(original[3])
    original_dv = float(original[4])
    original_arrival = departure + tof

    print(f"\nTRANSFERIMI {a} -> {b}")
    print(f"Fluturimi aktual: {tof:.4f} dite")
    print(f"Delta-V aktual: {original_dv:.2f} m/s")

    # Provojme nisje nga 1 dite me heret
    # deri ne 1 dite me vone.
    departure_offsets = [
        -1.0, -0.75, -0.5, -0.25,
        -0.125, 0.0, 0.125,
        0.25, 0.5, 0.75, 1.0,
    ]

    # Provojme fluturime me te shkurtra.
    reductions = [
        0.125, 0.25, 0.5, 0.75,
        1.0, 1.5, 2.0, 2.5, 3.0,
    ]

    candidates = []

    for offset in departure_offsets:
        new_departure = departure + offset

        if new_departure < float(udp.t0.mjd2000):
            continue

        for reduction in reductions:
            new_tof = tof - reduction

            if new_tof < udp.min_tof:
                continue

            try:
                dv = float(
                    udp.compute_transfer(
                        a,
                        b,
                        new_departure,
                        new_tof,
                    )
                )
            except Exception:
                continue

            if not math.isfinite(dv):
                continue

            if dv > DV_LIMIT:
                continue

            new_arrival = new_departure + new_tof
            arrival_gain = original_arrival - new_arrival

            if arrival_gain <= 0:
                continue

            candidates.append(
                (
                    arrival_gain,
                    reduction,
                    offset,
                    new_departure,
                    new_tof,
                    dv,
                )
            )

    candidates.sort(reverse=True)

    if not candidates:
        print("Nuk u gjet alternative me ardhje me te hershme.")
        continue

    print(f"Alternativa te gjetura: {len(candidates)}")
    print("5 alternativat me te mira:")

    for candidate in candidates[:5]:
        gain, reduction, offset, dep, flight, dv = candidate

        print(
            f"  Fitim ardhjeje: {gain:.3f} dite | "
            f"Fluturim: {flight:.3f}d | "
            f"Nisja: {offset:+.3f}d | "
            f"DV: {dv:.2f} m/s"
        )

print("\n" + "=" * 65)
print(f"Koha e analizes: {time.monotonic() - start:.1f}s")
print("Asnje skedar nuk eshte ndryshuar.")


import json
import time
import random
import numpy as np
from beam_search import transfer_dv_cached

SOURCE = "results/KTTSP_41_verified.json"
OUTPUT = "results/KTTSP_LNS_best.json"

TIME_LIMIT = 180
SEED = 42

random.seed(SEED)

with open(SOURCE) as f:
    record = json.load(f)

original = record["transfers"]
missing = sorted(set(range(49)) - set(record["path"]))

start = time.monotonic()
attempts = 0
best_count = len(record["path"])


def evaluate_route(transfers):
    if not transfers:
        return None

    path = [int(transfers[0][0])]
    previous_arrival = 0.0
    exceptions = 0
    checked = []

    for a, b, dep, tof, _ in transfers:
        a, b = int(a), int(b)
        dep, tof = float(dep), float(tof)

        if path[-1] != a or b in path:
            return None

        if dep < previous_arrival - 1e-8:
            return None

        if tof < 0.001 or dep + tof > 200 + 1e-8:
            return None

        dv = transfer_dv_cached(a, b, dep, tof)

        if not np.isfinite(dv) or dv > 600:
            return None

        exceptions += int(dv > 100)

        if exceptions > 5:
            return None

        checked.append([a, b, dep, tof, float(dv)])
        path.append(b)
        previous_arrival = dep + tof

    return {
        "path": path,
        "transfers": checked,
        "arrival_day": previous_arrival,
        "exceptions": exceptions,
        "verified": True
    }


def try_segment(k, x):
    # Rindertojme 3 transferime ekzistuese
    width = 3

    if k + width > len(original):
        return None

    segment = original[k:k + width]

    nodes = [int(segment[0][0])]
    nodes += [int(t[1]) for t in segment]

    earliest = (
        0.0 if k == 0
        else original[k - 1][2] + original[k - 1][3]
    )

    latest = (
        200.0 if k + width == len(original)
        else original[k + width][2]
    )

    if latest - earliest < 0.5:
        return None

    # Fusim domaten X ne pozita te ndryshme
    position = random.randint(1, len(nodes) - 1)
    nodes.insert(position, int(x))

    available = (
        5 - record["exceptions"]
        + sum(t[4] > 100 for t in segment)
    )

    # Provojme kombinime te ndryshme te koheve
    for trial in range(12):
        current_time = earliest
        new_segment = []
        used = 0
        success = True

        for a, b in zip(nodes[:-1], nodes[1:]):
            remaining = len(nodes) - 1 - len(new_segment)

            time_left = latest - current_time

            if time_left < remaining * 0.001:
                success = False
                break

            # Zgjedhim kohe fluturimi
            max_tof = min(
                15.0,
                time_left - (remaining - 1) * 0.001
            )

            if max_tof < 0.001:
                success = False
                break

            tof = random.uniform(
                min(0.5, max_tof),
                max_tof
            )

            dep = current_time

            dv = transfer_dv_cached(
                int(a), int(b),
                float(dep), float(tof)
            )

            attempts_valid = np.isfinite(dv) and dv <= 600

            if not attempts_valid:
                success = False
                break

            used += int(dv > 100)

            if used > available:
                success = False
                break

            new_segment.append([
                int(a), int(b),
                float(dep), float(tof),
                float(dv)
            ])

            current_time = dep + tof

        if success and current_time <= latest + 1e-8:
            candidate = (
                original[:k]
                + new_segment
                + original[k + width:]
            )

            result = evaluate_route(candidate)

            if result is not None:
                return result

    return None


print("LNS START")
print("Rekordi:", best_count, "/49")
print("Domatet qe mungojne:", missing)
print("Kufiri:", TIME_LIMIT, "sekonda", flush=True)

while time.monotonic() - start < TIME_LIMIT:
    k = random.randrange(len(original) - 2)
    x = random.choice(missing)

    attempts += 1
    result = try_segment(k, x)

    if result is not None:
        count = len(result["path"])

        if count > best_count:
            best_count = count

            with open(OUTPUT, "w") as f:
                json.dump(result, f, indent=2)

            print(
                "REKORD I RI:",
                count, "/49",
                "| Dita:",
                round(result["arrival_day"], 3),
                "| Perjashtime:",
                result["exceptions"],
                flush=True
            )

            break

print("LNS PERFUNDOI")
print("Prova:", attempts)
print("Koha:", round(time.monotonic() - start, 1), "sekonda")
print("Rezultati me i mire:", best_count, "/49")

if best_count == 41:
    print("Nuk u gjet 42/49 ne kete ekzekutim.")


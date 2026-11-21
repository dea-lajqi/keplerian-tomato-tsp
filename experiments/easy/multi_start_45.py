
import json
import time
from pathlib import Path

import beam_search as bs

RESULTS = Path("results")
SOURCE = RESULTS / "KTTSP_42_verified.json"
OUTPUT = RESULTS / "KTTSP_MULTI_BEST.json"

with open(SOURCE) as f:
    record = json.load(f)

full_path = record["path"]
full_transfers = record["transfers"]

best = (
    full_path,
    float(record["arrival_day"]),
    int(record["exceptions"]),
    full_transfers
)

# Pika te ndryshme nisjeje
prefix_lengths = [34, 28, 22, 16]

bs.TIME_LIMIT = 55
bs.BEAM_WIDTH = 60
bs.OUTPUT = OUTPUT

start = time.monotonic()

for prefix_length in prefix_lengths:
    path = full_path[:prefix_length]
    transfers = full_transfers[:prefix_length - 1]

    arrival = (
        float(transfers[-1][2])
        + float(transfers[-1][3])
    )

    used = sum(t[4] > 100 for t in transfers)

    state = (path, arrival, used, transfers)

    print(
        "\nNISJA NGA:",
        prefix_length,
        "DOMATE",
        flush=True
    )

    candidate = bs.beam_search(state, best)

    if len(candidate[0]) > len(best[0]):
        best = candidate

        print(
            "PERMIRËSIM:",
            len(best[0]),
            "/49",
            flush=True
        )

    if len(best[0]) >= 45:
        break

print("\nPERFUNDOI")
print("Koha:", round(time.monotonic()-start, 1))
print("Rezultati me i mire:", len(best[0]), "/49")
print("Kandidati:", OUTPUT)

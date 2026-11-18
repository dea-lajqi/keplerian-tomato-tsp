
import json
import numpy as np
from beam_search import transfer_dv_cached

SOURCE = "results/KTTSP_V44_best.json"
OUTPUT = "results/KTTSP_41_repaired.json"

with open(SOURCE) as f:
    record = json.load(f)

transfers = record["transfers"]
index = 31  # Transferimi nr. 32

i, j, old_dep, old_tof, old_dv = transfers[index]

previous = transfers[index - 1]
next_transfer = transfers[index + 1]

earliest = previous[2] + previous[3]
latest = next_transfer[2]

best = None
checked = 0

# Provojme kohe alternative nisjeje dhe mberritjeje
for departure in np.linspace(earliest, latest - 0.001, 65):
    for arrival in np.linspace(departure + 0.001, latest, 65):
        tof = arrival - departure

        dv = transfer_dv_cached(
            int(i), int(j),
            float(departure), float(tof)
        )
        checked += 1

        if np.isfinite(dv) and dv <= 100:
            candidate = (dv, departure, tof, arrival)

            if best is None or dv < best[0]:
                best = candidate

print("Transferime te testuara:", checked)

if best is None:
    print("Nuk u gjet riparim nen 100 m/s.")
else:
    dv, departure, tof, arrival = best

    record["transfers"][index] = [
        int(i), int(j),
        float(departure),
        float(tof),
        float(dv)
    ]

    # Rillogarisim perjashtimet me modelin e korrigjuar
    exceptions = 0

    for a, b, dep, flight, _ in record["transfers"]:
        actual = transfer_dv_cached(
            int(a), int(b),
            float(dep), float(flight)
        )
        if actual > 100:
            exceptions += 1

    record["exceptions"] = exceptions
    record["verified"] = exceptions <= 5

    with open(OUTPUT, "w") as f:
        json.dump(record, f, indent=2)

    print("DV i ri:", dv)
    print("Nisja:", departure)
    print("Koha e fluturimit:", tof)
    print("Mberritja:", arrival)
    print("Perjashtime:", exceptions)
    print("U ruajt:", OUTPUT)


import json
from beam_search import transfer_dv_cached

source = "results/KTTSP_41_repaired.json"
output = "results/KTTSP_41_verified.json"

with open(source) as f:
    record = json.load(f)

new_transfers = []
exceptions = 0

for i, j, dep, tof, old_dv in record["transfers"]:
    dv = transfer_dv_cached(
        int(i), int(j), float(dep), float(tof)
    )

    new_transfers.append([
        int(i), int(j), float(dep),
        float(tof), float(dv)
    ])

    if dv > 100:
        exceptions += 1

record["transfers"] = new_transfers
record["exceptions"] = exceptions
record["verified"] = False  # Derisa ta kontrollojme perseri

with open(output, "w") as f:
    json.dump(record, f, indent=2)

print("Domate:", len(record["path"]))
print("Perjashtime:", exceptions)
print("U ruajt:", output)

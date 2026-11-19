
import json
import math
from pathlib import Path

from beam_search import transfer_dv_cached

SOURCE = Path("results/KTTSP_42_verified.json")
OUTPUT = Path("results/KTTSP_LAST_OPTIMIZED.json")

with open(SOURCE) as f:
    record = json.load(f)

transfers = record["transfers"]

# Transferimi 39 (indeksi 38)
start = float(transfers[38][2])

# Provojme kohe te ndryshme fluturimi
# per 46 -> 34 -> 18 -> 23
options = [
    0.25, 0.5, 0.75, 1.0,
    1.5, 2.0, 2.5, 3.0,
    4.0, 5.0
]

best = None
tested = 0

for t1 in options:
    for t2 in options:
        for t3 in options:

            finish = start + t1 + t2 + t3

            if finish > 200:
                continue

            dv1 = transfer_dv_cached(46, 34, round(start, 6), t1)
            dv2 = transfer_dv_cached(34, 18, round(start+t1, 6), t2)
            dv3 = transfer_dv_cached(18, 23, round(start+t1+t2, 6), t3)

            if not all(math.isfinite(x) for x in [dv1,dv2,dv3]):
                continue

            tested += 1

            if any(x > 600 for x in [dv1,dv2,dv3]):
                continue

            exceptions = sum(x > 100 for x in [dv1,dv2,dv3])

            # Synojme asnje perjashtim ne segment
            if exceptions == 0:
                best = (t1,t2,t3,dv1,dv2,dv3,finish)
                break

        if best:
            break
    if best:
        break

print("Kombinime te testuara:", tested)

if best is None:
    print("Nuk u gjet segment pa perjashtime.")
else:
    t1,t2,t3,dv1,dv2,dv3,finish = best

    new_record = dict(record)
    new_transfers = [list(x) for x in transfers]

    new_transfers[38] = [46,34,start,t1,dv1]
    new_transfers[39] = [34,18,start+t1,t2,dv2]
    new_transfers[40] = [18,23,start+t1+t2,t3,dv3]

    new_record["transfers"] = new_transfers
    new_record["arrival_day"] = finish
    new_record["exceptions"] = sum(x[4] > 100 for x in new_transfers)
    new_record["verified"] = False

    with open(OUTPUT,"w") as f:
        json.dump(new_record,f,indent=2)

    print("U GJET!")
    print("Kohet:",t1,t2,t3)
    print("Delta-V:",round(dv1,2),round(dv2,2),round(dv3,2))
    print("Dita:",round(finish,3))
    print("Perjashtime gjithsej:",new_record["exceptions"])

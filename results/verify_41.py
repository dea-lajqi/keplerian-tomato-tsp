
import json
from beam_search import transfer_dv_cached

with open("results/KTTSP_41_verified.json") as f:
    record = json.load(f)

errors = []
exceptions = 0
previous_arrival = 0.0

for index, transfer in enumerate(record["transfers"], 1):
    i, j, departure, tof, saved_dv = transfer

    actual_dv = transfer_dv_cached(
        int(i), int(j), float(departure), float(tof)
    )

    arrival = departure + tof

    if departure < previous_arrival - 1e-7:
        errors.append((index, "Koha nuk perputhet"))

    if tof < 0.001:
        errors.append((index, "Koha e fluturimit shume e shkurter"))

    if arrival > 200 + 1e-7:
        errors.append((index, "Kalon 200 dite"))

    if actual_dv > 600 + 1e-7:
        errors.append((index, "DV mbi 600"))

    if actual_dv > 100 + 1e-7:
        exceptions += 1

    if abs(actual_dv - saved_dv) > 0.01:
        errors.append((index, "DV ndryshe", saved_dv, actual_dv))

    previous_arrival = arrival

if exceptions > 5:
    errors.append(("Perjashtime", exceptions))

print("Domate:", len(record["path"]))
print("Transferime:", len(record["transfers"]))
print("Perjashtime te llogaritura:", exceptions)
print("Dita perfundimtare:", previous_arrival)
print("Gabime:", errors)

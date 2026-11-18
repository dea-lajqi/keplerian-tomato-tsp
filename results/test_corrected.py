
from orbital import orbital_state_corrected, MU_MOON
from read_data import read_instance
import pykep as pk
import numpy as np
import json

tomatoes = read_instance("data/easy.kttsp")[0]

with open("results/BACKUP_EASY_40.json") as f:
    record = json.load(f)

transfer = next(
    tr for tr in record["transfers"]
    if int(tr[0]) == 32 and int(tr[1]) == 27
)

a, b, dep, tof, saved_dv = transfer

r1, v1 = orbital_state_corrected(tomatoes[a], dep)
r2, v2 = orbital_state_corrected(tomatoes[b], dep + tof)

best_dv = float("inf")

for cw in (False, True):
    lp = pk.lambert_problem(
        r1.tolist(),
        r2.tolist(),
        float(tof) * 86400,
        MU_MOON,
        cw,
        20
    )

    for departure_v, arrival_v in zip(lp.v0, lp.v1):
        dv = (
            np.linalg.norm(np.array(departure_v) - v1)
            + np.linalg.norm(np.array(arrival_v) - v2)
        )
        best_dv = min(best_dv, dv)

print("DV i ruajtur:", saved_dv)
print("DV i korrigjuar:", best_dv)
print("Diferenca:", abs(saved_dv - best_dv))


import json
import numpy as np
import pykep as pk

from read_data import read_instance
from lambert import compute_transfer_pykep
from orbital import (
    orbital_state_at_time,
    MU_MOON,
    DAY2SEC
)


# 1. LEXOJME REKORDIN 40/49

with open("results/BACKUP_EASY_40.json") as f:
    record = json.load(f)

tomatoes = read_instance("data/easy.kttsp")[0]


# 2. KONTROLLOJME TRANSFERIMIN E PARE

a, b, dep, tof, saved_dv = record["transfers"][0]

dep = float(dep)
tof = float(tof)
saved_dv = float(saved_dv)

calculated_dv = compute_transfer_pykep(
    tomatoes[a],
    tomatoes[b],
    dep,
    tof
)

print("KRAHASIMI I DELTA-V")
print("Transferimi:", a, "->", b)
print("DV i ruajtur:", round(saved_dv, 6), "m/s")
print("DV i llogaritur:", round(calculated_dv, 6), "m/s")
print("Diferenca:", abs(saved_dv - calculated_dv), "m/s")


# 3. KRAHASOJME GJENDJET ORBITALE

def compare_state(elements, day):

    a, e, inc, raan, argp, M = map(float, elements)

    # Modeli yne orbital
    r_own, v_own = orbital_state_at_time(
        elements, day
    )

    # Gjendja fillestare me PyKEP
    E_anomaly = pk.m2e(M, e)
    true_anomaly = pk.e2f(E_anomaly, e)

    r0, v0 = pk.par2ic(
        [a, e, inc, raan, argp, true_anomaly],
        MU_MOON
    )

    # Propagimi me PyKEP 3.1.0
    r_pk, v_pk = pk.propagate_lagrangian(
        [r0, v0],
        day * DAY2SEC,
        MU_MOON
    )

    # Diferencat
    position_difference = np.linalg.norm(
        np.asarray(r_own) - np.asarray(r_pk)
    )

    velocity_difference = np.linalg.norm(
        np.asarray(v_own) - np.asarray(v_pk)
    )

    print(
        "Diferenca e pozicionit:",
        position_difference,
        "m"
    )

    print(
        "Diferenca e shpejtesise:",
        velocity_difference,
        "m/s"
    )


# 4. TESTOJME TE DY DOMATET

print("\nDOMATJA E NISJES")
compare_state(tomatoes[a], dep)

print("\nDOMATJA E MBERRITJES")
compare_state(tomatoes[b], dep + tof)



print("\nKRAHASIMI I KONSTANTAVE")
print("MU_MOON ne orbital.py:", MU_MOON)
print("MU_MOON ne PyKEP:", pk.MU_MOON)
print("Diferenca:", abs(MU_MOON - pk.MU_MOON))

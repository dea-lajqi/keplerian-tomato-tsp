
from read_data import read_instance
from lambert import compute_transfer_pykep
import numpy as np

data = read_instance("data/easy.kttsp")
tomatoes = data[0]

dv = compute_transfer_pykep(
    tomatoes[0],
    tomatoes[1],
    start_day=0.0,
    tof_days=1.0
)

print("Delta-V:", dv, "m/s")
print("Rezultati eshte numer i vlefshem:", np.isfinite(dv))




import pykep as pk
from orbital import orbital_state_at_time, MU_MOON

elements = tomatoes[0]
a, e, inc, raan, argp, M = elements

# Llogaritja jone
r_own, v_own = orbital_state_at_time(elements, 0.0)

# Llogaritja me PyKEP
E_anomaly = pk.m2e(float(M), float(e))
true_anomaly = pk.e2f(E_anomaly, float(e))

r_pk, v_pk = pk.par2ic(
    [float(a), float(e), float(inc),
     float(raan), float(argp), float(true_anomaly)],
    MU_MOON
)

print("\nKRAHASIMI I MODELEVE")
print("Diferenca e pozicionit:",
      np.linalg.norm(np.asarray(r_own) - np.asarray(r_pk)), "m")
print("Diferenca e shpejtesise:",
      np.linalg.norm(np.asarray(v_own) - np.asarray(v_pk)), "m/s")

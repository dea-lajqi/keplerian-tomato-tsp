"""Diagnose bridges into missing tomatoes 4, 11, 17. Read-only."""
from functools import lru_cache
import numpy as np
import pykep as pk
from read_data import read_instance
from orbital import orbital_state_corrected, MU_MOON

T = read_instance('data/easy.kttsp')[0]
TARGETS = (4, 11, 17)
ANCHORS = (32, 27, 16, 14, 42, 23, 34)
# Broad initial-phase scan. These are individual edges, NOT a valid route.
DEPARTURES = tuple(float(x) for x in np.arange(0, 30.01, 0.5))
TOFS = (0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 7, 10, 12)

@lru_cache(maxsize=20000)
def state(i, day):
    return orbital_state_corrected(T[i], day)

def dv(i, j, dep, tof):
    try:
        r1, v1 = state(i, dep)
        r2, v2 = state(j, dep + tof)
        best = float('inf')
        for cw in (False, True):
            try:
                lp = pk.lambert_problem(np.asarray(r1).tolist(), np.asarray(r2).tolist(), tof * 86400., MU_MOON, cw, 20)
                for a, b in zip(lp.v0, lp.v1):
                    best = min(best, float(np.linalg.norm(np.asarray(a) - v1) + np.linalg.norm(np.asarray(b) - v2)))
            except (ValueError, RuntimeError, OverflowError):
                pass
        return best
    except (ValueError, RuntimeError, OverflowError):
        return float('inf')

def scan(i, j):
    best = None
    normal = None
    exception = None
    for dep in DEPARTURES:
        for tof in TOFS:
            value = dv(i, j, dep, tof)
            item = (value, dep, tof)
            if np.isfinite(value) and (best is None or value < best[0]):
                best = item
            if value <= 100 and (normal is None or dep + tof < normal[1] + normal[2]):
                normal = item
            if value <= 600 and (exception is None or dep + tof < exception[1] + exception[2]):
                exception = item
    return best, normal, exception

def fmt(item):
    if item is None: return 'NONE in sampled grid'
    value, dep, tof = item
    return f'dep={dep:g}, tof={tof:g}, finish={dep+tof:g}, dv={value:.2f}'

if __name__ == '__main__':
    print('READ-ONLY | sampled departure days 0..30 step 0.5 | 12 flight times', flush=True)
    print('Edges are independent; feasibility of a complete timed route is NOT established.', flush=True)
    for target in TARGETS:
        print(f'\n=== TARGET {target} ===', flush=True)
        for anchor in ANCHORS:
            for i, j in ((anchor, target), (target, anchor)):
                best, normal, exception = scan(i, j)
                print(f'{i:2d}->{j:2d} | earliest normal: {fmt(normal)} | earliest <=600: {fmt(exception)} | lowest dv: {fmt(best)}', flush=True)
    print('\nDONE. Existing solution files unchanged.', flush=True)

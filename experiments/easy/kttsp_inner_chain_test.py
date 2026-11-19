"""Diagnostic only: find a timed 16 -> 11 -> 17 -> 4 -> 14 chain.
Does not alter saved solutions or claim a valid 47/49 route.
Run from the project root with the (kttsp) environment activated.
"""
import numpy as np
import pykep as pk
from functools import lru_cache
from read_data import read_instance
from orbital import orbital_state_corrected, MU_MOON

TOMATOES = read_instance('data/easy.kttsp')[0]
CHAIN = [16, 11, 17, 4, 14]
START_DAY = 3.5  # arrival at 16 in the validated route
MAX_FINISH_DAY = 30.0
WAIT_OPTIONS = (0.0, 0.5, 1.0, 2.0, 3.0)
TOF_OPTIONS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0)
BEAM_WIDTH = 30

@lru_cache(maxsize=100000)
def dv_for(i, j, departure, tof):
    r1, v1 = orbital_state_corrected(TOMATOES[i], departure)
    r2, v2 = orbital_state_corrected(TOMATOES[j], departure + tof)
    best = float('inf')
    for cw in (False, True):
        try:
            lp = pk.lambert_problem(np.asarray(r1).tolist(), np.asarray(r2).tolist(),
                                    tof * 86400.0, MU_MOON, cw, 20)
            for vdep, varr in zip(lp.v0, lp.v1):
                dv = (np.linalg.norm(np.asarray(vdep) - v1) +
                      np.linalg.norm(np.asarray(varr) - v2))
                best = min(best, float(dv))
        except (ValueError, RuntimeError, OverflowError):
            pass
    return best


def main():
    # state = (arrival, exceptions_used, total_dv, transfers)
    states = [(START_DAY, 0, 0.0, [])]
    print('Testing chain:', ' -> '.join(map(str, CHAIN)), flush=True)
    for i, j in zip(CHAIN, CHAIN[1:]):
        next_states = []
        for arrival, exceptions, total_dv, transfers in states:
            for wait in WAIT_OPTIONS:
                dep = round(arrival + wait, 6)
                for tof in TOF_OPTIONS:
                    finish = round(dep + tof, 6)
                    if finish > MAX_FINISH_DAY:
                        continue
                    dv = dv_for(i, j, dep, tof)
                    if not np.isfinite(dv) or dv > 600:
                        continue
                    extra = int(dv > 100)
                    if exceptions + extra > 1:
                        continue
                    next_states.append((finish, exceptions + extra, total_dv + dv,
                                        transfers + [(i, j, dep, tof, round(dv, 3))]))
        if not next_states:
            print(f'No feasible tested link {i} -> {j} under these search settings.')
            return
        # Retain diversity by arrival-time bucket and exception count.
        next_states.sort(key=lambda s: (s[1], s[0], s[2]))
        selected = []
        seen = set()
        for state in next_states:
            key = (state[1], int(state[0] * 2))
            if key in seen:
                continue
            seen.add(key)
            selected.append(state)
            if len(selected) >= BEAM_WIDTH:
                break
        states = selected
        print(f'{i} -> {j}: {len(next_states)} feasible candidates; '
              f'{len(states)} retained; earliest arrival {min(s[0] for s in states):.2f}', flush=True)
    best = min(states, key=lambda s: (s[1], s[0], s[2]))
    print('\nCHAIN FOUND (not a full-route validation)')
    print('Finish day:', best[0], '| exceptions:', best[1])
    for step in best[3]:
        print('  %d -> %d | depart %.2f | flight %.2f | dv %.3f' % step)
    print('\nIMPORTANT: Existing transfers after tomato 14 would need to be rescheduled and revalidated.')

if __name__ == '__main__':
    main()

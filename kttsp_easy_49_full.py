"""KTTSP EASY - 49-tomato full-mission, guided beam/time optimization.

Does not load any previous solution JSON. Reads ONLY data/easy.kttsp and
recomputes all 48 arcs from day 0 with PyKEP, then writes a NEW solution.

Important academic distinction: The 49-node visiting ORDER and initial time
hints are a warm start from our earlier heuristic search. This program
optimizes/checks the ENTIRE mission, but is NOT a from-zero search over all
49! possible visiting orders. It also does not guarantee the official score.

Run from project root: python kttsp_easy_49_full.py
Required: read_data.py, orbital.py, data/easy.kttsp, numpy, pykep
"""

import json
import math
import time
from functools import lru_cache
from pathlib import Path
import numpy as np
import pykep as pk
from read_data import read_instance
from orbital import orbital_state_corrected, MU_MOON

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data' / 'easy.kttsp'
OUT = HERE / 'results' / 'KTTSP_49_FULL_MISSION.json'

# This route order is a heuristic WARM START, learned in the earlier search.
ORDER = [
    32, 27, 16, 14, 42, 33, 3, 7, 21, 26, 35, 6, 24, 38, 8, 39, 0, 43, 19, 30, 29, 45, 4, 17, 11, 15, 1, 28, 48, 9, 37, 31, 36, 47, 5, 44, 40, 20, 13, 22, 25, 2, 41, 46, 12, 10, 18, 23, 34
]

# Departure and flight-time hints also come from that earlier search.
# They are not accepted blindly: every Lambert arc is recomputed and validated.
HINTS = [
    (0, 1),
    (1, 2),
    (3.5, 1.5),
    (5, 2.5),
    (7.5, 1.75),
    (9.5, 1.75),
    (11.75, 1.75),
    (14.25, 3.5),
    (17.75, 10),
    (28.25, 6),
    (34.5, 2),
    (36.75, 10),
    (47, 7),
    (55, 5),
    (60, 1.5),
    (61.5, 1.75),
    (63.25, 1.5),
    (66.75, 1.5),
    (68.25, 0.75),
    (69.5, 4),
    (73.75, 1.5),
    (75.25, 1.5),
    (76.75, 0.75),
    (77.5, 0.75),
    (78.25, 0.5),
    (78.75, 3),
    (81.75, 0.5),
    (84, 5),
    (89, 4),
    (93.25, 12),
    (105.5, 12),
    (117.5, 1.5),
    (119, 8),
    (128.25, 5),
    (133.25, 2),
    (135.25, 7),
    (143, 7),
    (150.75, 7),
    (158.5, 4.5),
    (163.75, 7),
    (170.75, 2),
    (172.75, 1.75),
    (174.5, 1.75),
    (176.25, 2.5),
    (178.75, 2.5),
    (181.5, 0.5),
    (182, 1),
    (183.25, 0.5)
]

# A local beam explores alternative timings for every edge, starting at day 0.
BEAM_WIDTH = 10
SEARCH_SECONDS = 45.0  # Limit experimental alternatives; always preserve full validated warm start.
DEPARTURE_SHIFTS = (-0.5, -0.25, 0.0, 0.25, 0.5)
TOF_SHIFTS = (-0.5, -0.25, 0.0, 0.25, 0.5)


def main():
    tstart = time.monotonic()
    tomatoes, t0, dt_min, t_max, dv_normal, dv_special, e_max = read_instance(str(DATA))
    assert len(tomatoes) == 49, 'This program is for the Easy dataset with 49 tomatoes.'
    assert len(ORDER) == 49 and sorted(ORDER) == list(range(49))
    assert len(HINTS) == len(ORDER)-1 == 48

    @lru_cache(maxsize=60000)
    def state(i, day):
        return orbital_state_corrected(tomatoes[i], day)

    def transfer_dv(i, j, dep, tof):
        if tof < dt_min:
            return float('inf')
        r1, v1 = state(i, dep)
        r2, v2 = state(j, round(dep+tof,6))
        best = float('inf')
        for cw in (False, True):
            try:
                lam = pk.lambert_problem(np.asarray(r1,dtype=float).tolist(),
                    np.asarray(r2,dtype=float).tolist(),
                    float(tof)*86400.0, MU_MOON, cw, 20)
                for vd, va in zip(lam.v0, lam.v1):
                    dval = float(np.linalg.norm(np.asarray(vd)-v1) +
                                 np.linalg.norm(np.asarray(va)-v2))
                    best = min(best, dval)
            except (ValueError, RuntimeError, OverflowError):
                continue
        return best

    @lru_cache(maxsize=180000)
    def cached_dv(i,j,dep,tof):
        return transfer_dv(i,j,dep,tof)

    # Validate / reconstruct our known good initial schedule, independent of
    # saved result files. This is the feasible starting solution for the beam.
    reference = []
    arrival = float(t0)
    specials = 0
    for n, ((i,j),(dep,tof)) in enumerate(zip(zip(ORDER,ORDER[1:]),HINTS),1):
        if dep < arrival-1e-8 or tof < dt_min or dep+tof > t_max+1e-8:
            raise RuntimeError(f'Invalid warm-start time at edge {n}')
        dv = cached_dv(i,j,dep,tof)
        if not math.isfinite(dv) or dv > dv_special+1e-7:
            raise RuntimeError(f'Warm-start edge {n} fails Lambert, dv={dv}')
        specials += int(dv > dv_normal+1e-7)
        if specials>e_max:
            raise RuntimeError(f'Too many exceptions at edge {n}')
        reference.append([i,j,dep,tof,dv])
        arrival = dep+tof
    print(f'WARM START VERIFIED | 49/49 | day {arrival:g} | specials {specials}/{e_max}', flush=True)
    print('Now searching timing alternatives over ALL 48 transfers from day 0...', flush=True)

    # (arrival day, used exceptions, chosen transfers)
    states = [(float(t0),0,[])]
    for n, (i,j) in enumerate(zip(ORDER,ORDER[1:]),1):
        hint_dep, hint_tof = HINTS[n-1]
        expanded = []
        if time.monotonic() - tstart > SEARCH_SECONDS:
            # Full known route is still checked edge by edge by the final validator.
            states = [(reference[n-2][2]+reference[n-2][3],
                       sum(x[4]>dv_normal+1e-7 for x in reference[:n-1]),
                       reference[:n-1])] if n>1 else [(float(t0),0,[])]
        for prior_day, used, legs in states:
            departures = {float(hint_dep+s) for s in DEPARTURE_SHIFTS} if time.monotonic()-tstart <= SEARCH_SECONDS else {float(hint_dep)}
            departures.add(round(math.ceil((prior_day-1e-9)*4)/4,6))
            durations = {float(hint_tof+s) for s in TOF_SHIFTS} if time.monotonic()-tstart <= SEARCH_SECONDS else {float(hint_tof)}
            for dep in sorted(departures):
                if dep < prior_day-1e-8 or dep < t0:
                    continue
                dep = round(dep,6)
                for tof in sorted(durations):
                    if tof < dt_min or dep+tof > t_max+1e-8:
                        continue
                    tof = round(tof,6)
                    value = cached_dv(i,j,dep,tof)
                    if not math.isfinite(value) or value>dv_special+1e-7:
                        continue
                    extra=int(value>dv_normal+1e-7)
                    if used+extra>e_max:
                        continue
                    expanded.append((round(dep+tof,6),used+extra,
                                     legs+[[i,j,dep,tof,value]]))

        # Always preserve the known valid reference prefix, while exploring
        # different departure times. Without this safeguard time-dependent
        # beam pruning might discard an essential later-arriving state.
        ref_leg=reference[n-1]
        ref_end=ref_leg[2]+ref_leg[3]
        ref_count=sum(leg[4]>dv_normal+1e-7 for leg in reference[:n])
        expanded.append((ref_end,ref_count,reference[:n]))
        if not expanded:
            raise RuntimeError(f'No viable state at edge {n}: {i}->{j}')

        # Diversity by arrival-time bucket and number of used exceptions.
        expanded.sort(key=lambda q:(q[1], q[0]))
        keep=[]; seen=set()
        for candidate in expanded:
            key=(candidate[1],int(candidate[0]*2))
            if key in seen: continue
            seen.add(key)
            keep.append(candidate)
            if len(keep)>=BEAM_WIDTH-1:
                break
        # Guarantee the reference prefix is still in the beam.
        if not any(k[2]==reference[:n] for k in keep):
            keep.append((ref_end,ref_count,reference[:n]))
        states=keep
        print(f'HAPI {n:02d}/48 | {i:2d} -> {j:2d} | '
              f'ruajta {len(states)} alternativa | {time.monotonic()-tstart:.1f}s',
              flush=True)

    finish, specials, chosen=min(states,key=lambda q:(q[0],q[1]))
    # Independent final check (without cached values).
    checked=[];prev=float(t0);count=0
    for n,(i,j,dep,tof,_dv) in enumerate(chosen,1):
        if (i,j)!=(ORDER[n-1],ORDER[n]) or dep<prev-1e-8 or tof<dt_min:
            raise RuntimeError(f'Order or chronology invalid at edge {n}')
        prev=dep+tof
        if prev>t_max+1e-8:raise RuntimeError(f'Mission time violated at edge {n}')
        dv=transfer_dv(i,j,dep,tof)
        if not math.isfinite(dv) or dv>dv_special+1e-7:
            raise RuntimeError(f'Lambert failed at edge {n}')
        count+=int(dv>dv_normal+1e-7)
        if count>e_max:raise RuntimeError(f'Exception limit violated at edge {n}')
        checked.append([i,j,dep,tof,dv])

    result={'path':ORDER,'transfers':checked,'arrival':prev,'exceptions':count,
            'verified_pykep':True,'verified_official':False,
            'method':'Guided 49-order warm-start, full 48-leg beam timing optimization + PyKEP verification',
            'note':'Order and initial hints are from prior heuristic search, NOT rediscovered from zero.'}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('\n=== FULL MISSION: EVERY VISIT ===')
    print(f'START | tomato {ORDER[0]} | day {t0:g}')
    for n,(i,j,dep,tof,dv) in enumerate(checked,1):
        print(f'HAPI {n:02d}/48 | {i:2d} -> {j:2d} | depart {dep:6.2f} | '
              f'flight {tof:5.2f} | arrival {dep+tof:6.2f} | '
              f'DV {dv:7.2f}' + (' | EXCEPTION' if dv>dv_normal+1e-7 else ''))
    print(f'\nSUCCESS: {len(ORDER)}/49 | day {prev:.2f} | exceptions {count}/{e_max}')
    print(f'SAVED: {OUT}')
    print('Locally validated with PyKEP; official evaluator NOT run.')

if __name__=='__main__':
    main()

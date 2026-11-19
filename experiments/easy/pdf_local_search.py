"""KTTSP: windowed local search inspired by the supplied solver PDF.
Run from project root: python pdf_local_search.py
Requires existing beam_search.py and results/KTTSP_42_verified.json.
Never overwrites the verified record.
"""
import json
import math
import random
import time
from pathlib import Path

from beam_search import transfer_dv_cached, tomatoes

ROOT = Path('results')
SOURCE = ROOT / 'KTTSP_42_verified.json'
OUTPUT = ROOT / 'KTTSP_PDF_LOCAL_CANDIDATE.json'
TIME_LIMIT = 180.0
SEED = 26
rng = random.Random(SEED)

with SOURCE.open() as f:
    baseline = json.load(f)
original_path = list(baseline['path'])
original_legs = [list(t) for t in baseline['transfers']]
N = len(tomatoes)
MAX_TIME = 200.0
MAX_EXC = 5
NORMAL = 100.0
EXC = 600.0
MIN_TOF = 0.001

# In PDF: adaptive timing, windowed local search, swap / relocate / reversal.
# A small beam keeps multiple arrival/exception possibilities in a window.
WAITS = (0., 0.15, 0.5, 1.0, 2.0)
TOFS = (0.15, 0.3, 0.5, 0.75, 1., 1.5, 2., 3., 3.5, 5., 7., 10.)
WINDOW_BEAM = 7
start_clock = time.monotonic()


def count_exc(legs):
    return sum(float(x[4]) > NORMAL for x in legs)


def dv(i, j, dep, tof):
    try:
        return float(transfer_dv_cached(int(i), int(j), round(float(dep), 6), round(float(tof), 6)))
    except (ValueError, RuntimeError, OverflowError):
        return math.inf


def leg_choices(i, j, available, deadline, exceptions_left, old_leg=None):
    """Find diverse feasible arrival times; reuses old leg if it fits."""
    options = []
    candidates = [(w, tof) for w in WAITS for tof in TOFS]
    # Fine local timing around the old leg and fractional times.
    if old_leg is not None:
        for shift in (-0.5, -0.2, 0., 0.2, 0.5):
            for dt in (-0.5, -0.2, 0., 0.2, 0.5):
                dep = max(available, float(old_leg[2]) + shift)
                tof = float(old_leg[3]) + dt
                if tof >= MIN_TOF:
                    candidates.append((dep - available, tof))
    for wait, tof in candidates:
        dep = available + wait
        arr = dep + tof
        if tof < MIN_TOF or arr > deadline + 1e-8:
            continue
        val = dv(i, j, dep, tof)
        extra = int(val > NORMAL)
        if not math.isfinite(val) or val > EXC or extra > exceptions_left:
            continue
        options.append((arr, extra, [int(i), int(j), dep, tof, val]))
    # Preserve different arrival-time buckets and exception counts.
    options.sort(key=lambda x: (x[1], x[0]))
    unique = {}
    for arr, extra, leg in options:
        key = (extra, int(arr * 2))
        if key not in unique or arr < unique[key][0]:
            unique[key] = (arr, extra, leg)
    return sorted(unique.values(), key=lambda x: (x[1], x[0]))[:8]


def solve_window(new_nodes, a, b):
    """Replace original path[a:b+1] while retaining its endpoints and end-time."""
    prefix = original_legs[:a]
    suffix = original_legs[b:]
    start_time = 0.0 if a == 0 else float(original_legs[a-1][2]) + float(original_legs[a-1][3])
    end_time = float(original_legs[b-1][2]) + float(original_legs[b-1][3])
    # Keep old suffix exactly; it is valid if the window finishes by end_time.
    exc_budget = MAX_EXC - count_exc(prefix) - count_exc(suffix)
    if exc_budget < 0:
        return None
    states = [(start_time, 0, [])]
    for k, (i, j) in enumerate(zip(new_nodes, new_nodes[1:])):
        if time.monotonic() - start_clock >= TIME_LIMIT:
            return None
        next_states = []
        for arrival, used, legs in states:
            old_leg = original_legs[a+k] if a+k < b and k < b-a else None
            for arr, extra, leg in leg_choices(i, j, arrival, end_time,
                                               exc_budget-used, old_leg):
                next_states.append((arr, used+extra, legs+[leg]))
        if not next_states:
            return None
        # Pareto-ish diversity: distinct exception counts and arrival bins.
        next_states.sort(key=lambda x: (x[1], x[0]))
        kept = {}
        for state in next_states:
            key = (state[1], int(state[0] * 2))
            if key not in kept:
                kept[key] = state
        states = sorted(kept.values(), key=lambda x: (x[1], x[0]))[:WINDOW_BEAM]
    if not states:
        return None
    arrival, used, replacement = min(states, key=lambda s: (s[1], s[0]))
    new_path = original_path[:a] + list(new_nodes) + original_path[b+1:]
    new_legs = prefix + replacement + suffix
    if len(new_legs) != len(new_path)-1:
        return None
    if len(set(new_path)) != len(new_path) or count_exc(new_legs) > MAX_EXC:
        return None
    return new_path, new_legs


def validate(path, legs):
    if len(path) != len(legs)+1 or len(path) != len(set(path)):
        return False
    prev = 0.
    exc = 0
    for k, leg in enumerate(legs):
        i,j,dep,tof,_ = leg
        if i != path[k] or j != path[k+1] or dep < prev-1e-7 or tof < MIN_TOF-1e-9:
            return False
        value = dv(i,j,dep,tof)
        if not math.isfinite(value) or value > EXC+1e-7:
            return False
        exc += value > NORMAL
        prev = dep+tof
    return exc <= MAX_EXC and prev <= MAX_TIME+1e-7


missing = sorted(set(range(N)) - set(original_path))
print('PDF local search | baseline:', len(original_path), '/', N, flush=True)
print('Missing:', missing, '| limit:', TIME_LIMIT, 'seconds', flush=True)
print('Original verified JSON stays unchanged.', flush=True)

# Prioritize windows around exceptions, but also explore other windows.
windows = []
for width in (2, 3, 4):
    for a in range(0, len(original_path)-width):
        b = a+width
        if b < len(original_path):
            weight = 3 if any(original_legs[k][4] > NORMAL for k in range(a,b)) else 1
            windows.extend([(a,b)]*weight)
rng.shuffle(windows)
seen = set()
trials = 0
best_count = len(original_path)
for a,b in windows:
    if time.monotonic()-start_clock >= TIME_LIMIT:
        break
    if (a,b) in seen:
        continue
    seen.add((a,b))
    inner = original_path[a+1:b]
    # Try insertion and alternative ordering of the existing inner nodes.
    moves = []
    for target in missing:
        for position in range(len(inner)+1):
            cand = inner[:position]+[target]+inner[position:]
            moves.append(cand)
            if len(inner) >= 2:
                moves.append(list(reversed(cand)))
    rng.shuffle(moves)
    for inner_order in moves:
        if time.monotonic()-start_clock >= TIME_LIMIT:
            break
        trials += 1
        nodes = [original_path[a]] + inner_order + [original_path[b]]
        result = solve_window(nodes, a, b)
        if result is None:
            continue
        path, legs = result
        if len(path) <= best_count or not validate(path, legs):
            continue
        arrival = legs[-1][2]+legs[-1][3]
        out = {'version':'PDF_INSPIRED_WINDOW_LOCAL_SEARCH', 'path':path,
               'transfers':legs, 'arrival_day':arrival,
               'exceptions':count_exc(legs), 'verified':False}
        with OUTPUT.open('w') as f:
            json.dump(out,f,indent=2)
        best_count = len(path)
        print('NEW CANDIDATE:', best_count, '/', N, '| saved:', OUTPUT, flush=True)
        break
    if best_count > len(original_path):
        break
print('Finished. Trials:', trials, '| elapsed:', round(time.monotonic()-start_clock,1), 's')
if best_count == len(original_path):
    print('No new candidate. Verified 42/49 is unchanged.')
else:
    print('Candidate must be independently verified before accepting.')

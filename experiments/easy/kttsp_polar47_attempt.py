"""KTTSP: directed 47/49 attempt around compatible polar orbit tomatoes.

Run in the project directory: python kttsp_polar47_attempt.py
Requires: beam_search.py, orbital.py, read_data.py, data/easy.kttsp and
results/KTTSP_46_DAY183_VALIDATED.json. Never overwrites a validated file.

Plan: try the missing 4/11/17 in the high-orbit polar section of the
46-tomato itinerary; also remove up to 2 exceptional-edge tomatoes in order
to free exception slots. Re-time the changed suffix with a small, time-aware
beam search. Every new complete route is rechecked with PyKEP before saving.
"""
from __future__ import annotations

import itertools
import json
import math
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from beam_search import tomatoes, transfer_dv_cached

ROOT = Path('results')
SOURCE = ROOT / 'KTTSP_46_DAY183_VALIDATED.json'
OUTPUT = ROOT / 'KTTSP_47_POLAR_CANDIDATE.json'
T_MAX, DV_NORMAL, DV_MAX, EXCEPTIONS = 200.0, 100.0, 600.0, 5
TIME_LIMIT = 120.0  # seconds; raise to 240 only if you want a longer run
START = time.monotonic()
LAST_UPDATE = START


def heartbeat():
    global LAST_UPDATE
    now = time.monotonic()
    if now - LAST_UPDATE > 15:
        print("Searching guided routes... |", round(now - START), "seconds", flush=True)
        LAST_UPDATE = now


def elapsed():
    return time.monotonic() - START


def remaining():
    return TIME_LIMIT - elapsed()


@lru_cache(maxsize=200000)
def dv(i, j, departure, tof):
    try:
        z = float(transfer_dv_cached(int(i), int(j), round(float(departure), 6), round(float(tof), 6)))
        return z if math.isfinite(z) else math.inf
    except (ValueError, RuntimeError, OverflowError, ZeroDivisionError):
        return math.inf


def verify(s):
    route, legs = s['path'], s['transfers']
    if (len(route) != len(set(route)) or len(legs) != len(route) - 1 or
            any(x < 0 or x >= len(tomatoes) for x in route)):
        return False
    previous, used = 0.0, 0
    for k, leg in enumerate(legs):
        i, j, dep, tof, reported = leg
        if i != route[k] or j != route[k + 1] or dep < previous - 1e-7 or tof < .001:
            return False
        previous = dep + tof
        if previous > T_MAX + 1e-7:
            return False
        actual = dv(i, j, dep, tof)
        if actual > DV_MAX + 1e-7 or abs(actual - reported) > .001:
            return False
        used += actual > DV_NORMAL + 1e-7
    return used <= EXCEPTIONS and used == s['exceptions'] and abs(previous - s['arrival']) < 1e-6


# Probe dense launch phases; inner low orbits complete one revolution in ~0.24 days.
# The earlier 0.5-day grids were too coarse for this subproblem.
WAITS = (0, .0625, .125, .1875, .25, .375, .5, .75, 1, 1.5, 2, 3, 5)
TOFS = (.0625, .125, .1875, .25, .375, .5, .75, 1, 1.5, 2, 2.5, 3, 4, 5, 7, 9, 12)


@lru_cache(maxsize=18000)
def new_legs(i, j, departure_bucket):
    """An edge menu from the rounded-up departure time (1/16 d increments).

    Keep arrival-phase diversity; an earliest-only DP can miss a feasible next leg.
    """
    dep0 = departure_bucket / 16
    groups = {}
    for wait in WAITS:
        dep = round(dep0 + wait, 6)
        if dep >= T_MAX:
            continue
        for tof in TOFS:
            if remaining() <= 0:
                return tuple()
            heartbeat()
            end = dep + tof
            if end > T_MAX:
                continue
            v = dv(i, j, dep, tof)
            if v > DV_MAX + 1e-7:
                continue
            extra = int(v > DV_NORMAL + 1e-7)
            # 0.5-day arrival bins: several distinct phases per edge.
            bucket = (extra, round(end * 2))
            prior = groups.get(bucket)
            if prior is None or end < prior[0] - 1e-8 or (abs(end - prior[0]) < 1e-8 and v < prior[1][4]):
                groups[bucket] = (end, [i, j, dep, tof, v])
    by_type = {0: [], 1: []}
    for (extra, _), value in groups.items():
        by_type[extra].append(value)
    options = []
    for e in (0, 1):
        # Prefer early arrivals but retain later orbital phases.
        arr = sorted(by_type[e], key=lambda x: x[0])
        options.extend(arr[:6])
        if len(arr) > 6:
            options.extend(arr[6::max(1, len(arr) // 3)][:3])
    return tuple((float(t), tuple(leg)) for t, leg in options)


def attempt(route, original):
    """Retimes changed itinerary; reuses existing exact legs where possible."""
    old_path, old_legs = original['path'], original['transfers']
    first = next((k for k in range(min(len(route), len(old_path))) if route[k] != old_path[k]),
                 min(len(route), len(old_path)))
    # Prefix means all nodes through index first-1 unchanged.
    # Preserve transfers through first-2.
    fixed = max(0, first - 1)
    prefix = [list(x) for x in old_legs[:fixed]]
    t = prefix[-1][2] + prefix[-1][3] if prefix else 0.0
    ex = sum(x[4] > DV_NORMAL + 1e-7 for x in prefix)
    if ex > EXCEPTIONS:
        return None
    states = [(t, ex, prefix)]
    old_edges = {(int(x[0]), int(x[1])): x for x in old_legs}
    for k in range(fixed, len(route) - 1):
        if remaining() <= 0:
            return None
        i, j = route[k], route[k + 1]
        old = old_edges.get((i, j))
        next_states = {}
        for arrival, used, transfers in states:
            options = []
            # It is always legal to wait at a tomato and reuse an original transfer.
            if old is not None and old[2] >= arrival - 1e-7:
                options.append((float(old[2] + old[3]), tuple(old)))
            # Re-time modified edges and old exceptional edges (to free exceptions).
            # If old dep is too early, old plan cannot be reused at this phase.
            if old is None or old[4] > DV_NORMAL or old[2] < arrival - 1e-7:
                bucket = math.ceil((arrival - 1e-9) * 16)
                options.extend(new_legs(i, j, bucket))
            for end, leg in options:
                new_used = used + int(leg[4] > DV_NORMAL + 1e-7)
                if new_used > EXCEPTIONS or end > T_MAX + 1e-7:
                    continue
                # Need enough time for remaining legs; use only a safe optimistic bound.
                if end + .001 * (len(route) - k - 2) > T_MAX + 1e-7:
                    continue
                key = (new_used, round(end * 2))
                prior = next_states.get(key)
                if prior is None or end < prior[0] - 1e-7:
                    next_states[key] = (end, new_used, transfers + [list(leg)])
        if not next_states:
            return None
        # Retain a variety of arrival phases for each exception budget.
        by_ex = defaultdict(list)
        for st in next_states.values():
            by_ex[st[1]].append(st)
        states = []
        for e, subset in by_ex.items():
            subset.sort(key=lambda s: s[0])
            states.extend(subset[:7])
            if len(subset) > 7:
                states.extend(subset[7::max(1, len(subset) // 3)][:3])
        states.sort(key=lambda z: (z[1], z[0]))
        states = states[:34]
    if not states:
        return None
    arr, ex, legs = min(states, key=lambda z: (z[0], z[1]))
    result = {'path': route, 'transfers': legs, 'arrival': arr, 'exceptions': ex,
              'verified': False, 'method': 'polar_cluster_exception_repair'}
    return result if verify(result) else None


def get_candidate_routes(path):
    missing = sorted(set(range(len(tomatoes))) - set(path))
    if missing != [4, 11, 17]:
        raise SystemExit(f'Unexpected missing tomatoes {missing}; expected [4, 11, 17].')
    # Polar high-orbit anchors; include orbit plane alignment, not only altitude.
    anchors = [x for x in path if tomatoes[x][0] > 1e7 and abs(math.cos(tomatoes[x][2])) < .35]
    # Existing exceptional-edge nodes that may be replaced to free exception slots.
    removals = [20, 24, 21, 7, 6, 40, 18, 10, 14, 16, 13, 22]
    removal_pairs = list(itertools.combinations(removals, 2))
    # Higher priority when old exceptional edge vanishes on removing the node.
    primary = [20, 24, 7, 6, 21, 40, 18, 10]
    removal_pairs.sort(key=lambda p: (sum(x not in primary for x in p),
                                       abs(path.index(p[0]) - path.index(p[1]))))
    # First attempt full triple (3 new - 2 removed = 47) with compatible polar anchors.
    # Pair 11 <-> 17 <-> 4 has demonstrated low-dv individual edges.
    for left, right in removal_pairs:
        if left not in path or right not in path:
            continue
        clean = [x for x in path if x not in (left, right)]
        for anchor in anchors:
            if anchor not in clean:
                continue
            idx = clean.index(anchor)
            for offset in (1, 0):
                at = idx + offset
                for trio in ((4, 17, 11), (11, 17, 4)):
                    yield clean[:at] + list(trio) + clean[at:]
    # Also try replacing just one troublesome tomato with two missing ones.
    for removed in primary:
        if removed not in path:
            continue
        clean = [x for x in path if x != removed]
        for anchor in anchors:
            if anchor not in clean:
                continue
            at = clean.index(anchor) + 1
            for duo in ((4, 17), (17, 4), (11, 17), (17, 11)):
                yield clean[:at] + list(duo) + clean[at:]
    # As a final option, one missing tomato without removing any original one.
    for anchor in anchors:
        at = path.index(anchor) + 1
        for target in missing:
            yield path[:at] + [target] + path[at:]


def main():
    if not SOURCE.exists():
        raise SystemExit(f'File missing: {SOURCE}. No files changed.')
    seed = json.loads(SOURCE.read_text(encoding='utf8'))
    if not verify(seed) or len(seed['path']) != 46:
        raise SystemExit('Original 46 solution failed validation. No files changed.')
    print('START | VALIDATED 46/49 | day %.2f | 5/5 exceptions' % seed['arrival'], flush=True)
    print('NEW STRATEGY: polar outer anchors, missing trio, remove 1-2 costly tomatoes', flush=True)
    anchors = [x for x in seed['path'] if tomatoes[x][0] > 1e7 and abs(math.cos(tomatoes[x][2])) < .35]
    print('Compatible outer anchors:', anchors, '| max time', int(TIME_LIMIT), 's', flush=True)
    trials = 0
    saved = False
    last_progress = elapsed()
    try:
        for route in get_candidate_routes(seed['path']):
            if remaining() <= 0:
                break
            trials += 1
            attempt_result = attempt(route, seed)
            if attempt_result and len(attempt_result['path']) >= 47:
                # Do not overwrite existing candidate from another run either.
                target = OUTPUT
                serial = 2
                while target.exists():
                    target = ROOT / f'KTTSP_47_POLAR_CANDIDATE_{serial}.json'
                    serial += 1
                target.write_text(json.dumps(attempt_result, indent=2), encoding='utf8')
                print('\nFOUND %d/49 candidate | day %.3f | exceptions %d' % (
                    len(route), attempt_result['arrival'], attempt_result['exceptions']), flush=True)
                print('SAVED:', target, flush=True)
                print('All edges checked with PyKEP; separate official checker still needed.', flush=True)
                saved = True
                break
            if trials % 10 == 0 or elapsed() - last_progress > 15:
                print('Trials', trials, '| best 46/49 |', round(elapsed()), 'sec', flush=True)
                last_progress = elapsed()
    except KeyboardInterrupt:
        print('Stopped by user.', flush=True)
    if not saved:
        print('FINISHED | no 47 in tested guided routes | trials', trials,
              '| seconds', round(elapsed()), flush=True)
    print('Original 46 JSON unchanged.', flush=True)


if __name__ == '__main__':
    main()

"""Attempt to insert one missing tomato inside every existing transfer.

Run from the project root: python insert_all_edges.py
Keeps the original 42-tomato schedule and all later arrivals unchanged.
Does not overwrite the verified baseline. Any result is a candidate until independently verified.
"""
import json
import math
import time
from pathlib import Path

from scipy.optimize import minimize_scalar
from beam_search import transfer_dv_cached

BASE = Path('results/KTTSP_42_verified.json')
OUT = Path('results/KTTSP_43_INSERT_CANDIDATE.json')
DV_NORMAL = 100.0
DV_EXCEPTION = 600.0
MAX_EXCEPTIONS = 5
DT_MIN = 0.001


def dv(i, j, dep, tof):
    if tof < DT_MIN:
        return math.inf
    try:
        return float(transfer_dv_cached(int(i), int(j), round(float(dep), 6), round(float(tof), 6)))
    except (ValueError, RuntimeError, OverflowError):
        return math.inf


def best_split(i, middle, j, dep, total_tof, exception_allowance):
    """Find a feasible two-leg replacement for one original transfer."""
    if total_tof <= 2 * DT_MIN:
        return None
    # Broad grid plus local refinement around the most promising splits.
    fractions = [0.02, 0.04, 0.06, 0.1, 0.15, 0.2, 0.3, 0.4,
                 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.94, 0.96, 0.98]
    lower = DT_MIN / total_tof
    samples = []

    def evaluate(frac):
        t1 = total_tof * float(frac)
        t2 = total_tof - t1
        if t1 < DT_MIN or t2 < DT_MIN:
            return None
        v1 = dv(i, middle, dep, t1)
        v2 = dv(middle, j, dep + t1, t2)
        if not (math.isfinite(v1) and math.isfinite(v2)):
            return None
        count = int(v1 > DV_NORMAL) + int(v2 > DV_NORMAL)
        return (count, max(v1, v2), v1 + v2, t1, t2, v1, v2)

    def valid(result):
        return result is not None and result[0] <= exception_allowance and result[1] <= DV_EXCEPTION

    for f in fractions:
        if lower < f < 1 - lower:
            res = evaluate(f)
            if res is not None:
                samples.append((f, res))
    feasible = [res for _, res in samples if valid(res)]
    if feasible:
        return min(feasible)

    # Minimize the worst of the two delta-Vs. This is a continuous-time
    # local search, rather than only the original beam-search time grid.
    for f, _ in sorted(samples, key=lambda item: (item[1][0], item[1][1]))[:4]:
        a = max(lower + 1e-8, f - 0.09)
        b = min(1 - lower - 1e-8, f + 0.09)
        if a >= b:
            continue
        opt = minimize_scalar(
            lambda x: (lambda r: r[1] if r else 1e12)(evaluate(x)),
            bounds=(a, b), method='bounded',
            options={'xatol': 1e-5, 'maxiter': 24}
        )
        for f2 in [opt.x, max(a, opt.x - 0.003), min(b, opt.x + 0.003)]:
            res = evaluate(f2)
            if valid(res):
                feasible.append(res)
    return min(feasible) if feasible else None


def main():
    record = json.loads(BASE.read_text(encoding='utf-8'))
    path = list(record['path'])
    transfers = list(record['transfers'])
    missing = sorted(set(range(49)) - set(path))
    print('Baseline:', len(path), '/49; missing:', missing, flush=True)
    assert len(transfers) == len(path) - 1
    assert len(path) == len(set(path))
    start = time.monotonic()
    checked = 0
    for k, old in enumerate(transfers):
        i, j, dep, tof, old_dv = old
        # With all five exceptions already used, replacement may use
        # at most as many exceptions as the original leg.
        allowed = int(float(old_dv) > DV_NORMAL)
        for middle in missing:
            checked += 1
            split = best_split(i, middle, j, float(dep), float(tof), allowed)
            if split is None:
                continue
            count, _, _, t1, t2, v1, v2 = split
            new_transfers = [list(x) for x in transfers]
            new_transfers[k:k+1] = [
                [i, middle, float(dep), t1, v1],
                [middle, j, float(dep) + t1, t2, v2],
            ]
            new_path = path[:k+1] + [middle] + path[k+1:]
            exceptions = sum(float(x[4]) > DV_NORMAL for x in new_transfers)
            assert exceptions <= MAX_EXCEPTIONS
            assert len(new_transfers) == len(new_path) - 1
            assert len(set(new_path)) == len(new_path)
            candidate = {
                'version': 'SPLIT_EDGE_INSERTION',
                'path': new_path,
                'arrival_day': float(record['arrival_day']),
                'exceptions': exceptions,
                'transfers': new_transfers,
                'verified': False,
                'source': str(BASE),
            }
            OUT.write_text(json.dumps(candidate, indent=2), encoding='utf-8')
            print('FOUND:', len(new_path), '/49; inserted:', middle,
                  'between', i, 'and', j, '; exceptions:', exceptions,
                  '; file:', OUT, flush=True)
            return
        print('Edge', k+1, '/', len(transfers), 'done;',
              'elapsed', round(time.monotonic()-start, 1), 's', flush=True)
    print('No insertion found among', checked, 'edge/tomato pairs.', flush=True)
    print('The original verified 42/49 remains unchanged.', flush=True)


if __name__ == '__main__':
    main()

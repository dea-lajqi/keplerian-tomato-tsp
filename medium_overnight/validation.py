"""Independent ESA Medium validation. This module NEVER writes original files."""
from __future__ import annotations
import json
import math
import runpy
from pathlib import Path

EVALUATOR_REL = 'esa_evaluation/spoc-4-keplerian-tomato-traveling-salesperson-medium-1782203984713.py'
BASELINE_REL = 'results/esa/medium_20261120T080509Z_bc1f8d8f_4_verified.json'


def official_udp(root):
    root = Path(root)
    script = root / EVALUATOR_REL
    if not script.is_file():
        raise FileNotFoundError(f'Cannot find original ESA Medium evaluator: {script}')
    import pykep as pk
    if not hasattr(pk.planet, 'keplerian'):
        raise RuntimeError('The ESA evaluator requires PyKEP 2.x with pk.planet.keplerian')
    # The official evaluator loads its instance by searching from __file__.
    udp = runpy.run_path(str(script))['udp']
    if udp.n_tomatoes != 181:
        raise ValueError(f'Expected 181 Medium tomatoes, got {udp.n_tomatoes}')
    return udp


def to_solution(data):
    path = list(map(int, data['path']))
    ts = data['transfers']
    departures = [float(row[2]) for row in ts]
    flights = [float(row[3]) for row in ts]
    return {'path': path, 'dep': departures, 'tof': flights}


def as_chromosome(sol):
    return list(sol['dep']) + list(sol['tof']) + list(sol['path'])


def verify(udp, sol, *, strict=True):
    """Both manual bounds and the exact original ESA fitness are required."""
    n = udp.n_tomatoes
    path, dep, tof = sol['path'], sol['dep'], sol['tof']
    if len(path) != n or len(dep) != n-1 or len(tof) != n-1:
        raise ValueError('Malformed Medium chromosome lengths')
    if len(set(path)) != n or sorted(path) != list(range(n)):
        raise ValueError('Tomatoes are missing or repeated')
    if any(not isinstance(x, int) for x in path):
        raise ValueError('Tomato indices must be integers')
    for i, (d, f) in enumerate(zip(dep, tof)):
        if not math.isfinite(d) or not math.isfinite(f):
            raise ValueError(f'Non-finite timing at transfer {i}')
        if d < 0 or f < udp.min_tof - 1e-10 or d+f > udp.max_time + 1e-9:
            raise ValueError(f'ESA time bounds violated at transfer {i}')
        if i > 0 and d+1e-9 < dep[i-1]+tof[i-1]:
            raise ValueError(f'Chronology violated at transfer {i}')
    fvals = [float(v) for v in udp.fitness(as_chromosome(sol))]
    if len(fvals) != 5 or any(not math.isfinite(v) for v in fvals):
        raise ValueError('Unexpected ESA fitness vector')
    if any(abs(fvals[i]) > 1e-9 for i in (1,2,3)) or fvals[4] > 0:
        raise ValueError(f'Original ESA evaluator rejected schedule: {fvals}')
    if abs(fvals[0] - (dep[-1] + tof[-1])) > 1e-8:
        raise ValueError('Mission arrival mismatch')
    return fvals


def baseline(root, udp):
    p = Path(root) / BASELINE_REL
    data = json.loads(p.read_text(encoding='utf-8'))
    sol = to_solution(data)
    f = verify(udp, sol)
    if abs(f[0] - 202.8125) > 1e-5:
        raise ValueError(f'Expected baseline 202.8125, found {f[0]}')
    return sol, f


def solution_payload(sol, f, *, provenance=None, udp=None):
    rows = []
    for i, (d,t) in enumerate(zip(sol['dep'],sol['tof'])):
        if udp is None:
            dv = None
        else:
            dv = float(udp.compute_transfer(sol['path'][i],sol['path'][i+1],d,t))
        rows.append([sol['path'][i],sol['path'][i+1], d, t, dv])
    return {'path': sol['path'], 'transfers': rows, 'arrival_day': f[0],
            'exceptions': sum(1 for row in rows if row[4] is not None and row[4] > udp.dv_threshold),
            'validation': {'fitness': f, 'valid': True, 'source': 'original ESA Medium fitness'},
            'provenance': provenance or {}}

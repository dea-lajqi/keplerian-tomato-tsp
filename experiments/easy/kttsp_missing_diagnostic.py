"""Diagnose missing KTTSP tomatoes with the project's exact PyKEP Lambert evaluator.
Run next to beam_search.py. Read-only: never modifies results.
"""
import json
import math
import time
from pathlib import Path
from beam_search import transfer_dv_cached

P = Path('results/KTTSP_46_DAY183_VALIDATED.json')
d = json.loads(P.read_text(encoding='utf-8'))
path = d['path']
missing = sorted(set(range(49)) - set(path))
inner = [i for i in range(49) if i in [4, 11, 16, 17, 23, 27, 32, 34]]
# Includes the early inner-orbit sequence and the final inner-orbit pair.
anchors = sorted(set(inner + [14, 18, 42]))
TOFS = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 12.0)
TIMES = (0, 1, 2, 3, 4, 5, 7, 10, 15, 25, 50, 90, 130, 160, 175, 185)

print('BASE:', len(path), '/49, arrival:', d['arrival'], 'missing:', missing, flush=True)
print('INNER ANCHORS:', anchors, flush=True)
start = time.monotonic()

for target in missing:
    print('\nTARGET', target, flush=True)
    for direction in ('IN', 'OUT'):
        records = []
        count_normal = 0
        count_exception = 0
        for neighbor in anchors:
            if neighbor == target:
                continue
            for t in TIMES:
                for tof in TOFS:
                    if t + tof > 200:
                        continue
                    i, j = (neighbor, target) if direction == 'IN' else (target, neighbor)
                    dv = transfer_dv_cached(i, j, float(t), float(tof))
                    if not math.isfinite(dv):
                        continue
                    if dv <= 100:
                        count_normal += 1
                        records.append((0, dv, t + tof, neighbor, t, tof))
                    elif dv <= 600:
                        count_exception += 1
                        records.append((1, dv, t + tof, neighbor, t, tof))
        records.sort(key=lambda x: (x[0], x[1], x[2]))
        print(direction, '| normal:', count_normal, '| exception:', count_exception, flush=True)
        for extra, dv, arrival, neighbor, dep, tof in records[:6]:
            print(' ', ('EXCEPTION' if extra else 'NORMAL'), 'neighbor', neighbor,
                  'dep', dep, 'tof', tof, 'dv', round(dv, 2), flush=True)
    print('elapsed', round(time.monotonic()-start, 1), 's', flush=True)

print('\nFINISHED. Diagnostic only; these are individual edges, not a valid full route.')

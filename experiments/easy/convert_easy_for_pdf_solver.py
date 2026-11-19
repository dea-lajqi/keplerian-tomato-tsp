import json
from pathlib import Path
from read_data import read_instance

src = Path('data/easy.kttsp')
tomatoes, t0, dt_min, t_max, dv_max, dv_exception, E = read_instance(str(src))
# Data is in metres, m^3/s^2 and radians. PDF solver expects internally consistent units.
# Preserve SI units and supply Moon GM in m^3/s^2.
output = {
    'elements': tomatoes,
    'mu': 4.9048695e12,
    'time_scale': 86400.0,
    't_ref': 0.0,
    't_start': t0,
    'dv_cap': dv_max,
    'dv_exc_cap': dv_exception,
    'n_exc': E,
    'tof_min': dt_min,
    't_max': t_max,
    'max_revs': 20,
}
Path('data/easy_for_pdf_solver.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
print('Created data/easy_for_pdf_solver.json with', len(tomatoes), 'targets')

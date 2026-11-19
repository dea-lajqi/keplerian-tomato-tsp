import json, math
from pathlib import Path
from beam_search import transfer_dv_cached
src=Path('results/KTTSP_46_PDF_ESCAPE_CANDIDATE.json')
d=json.loads(src.read_text(encoding='utf-8'))
p=d['path']; ts=d['transfers']; missing=sorted(set(range(49))-set(p))
issues=[]; prev=0.; ex=0
if len(p)!=46 or len(set(p))!=46: issues.append('path length/duplicates')
if len(ts)!=len(p)-1: issues.append('transfer count')
for k,(i,j,t,dt,v) in enumerate(ts):
    if (i,j)!=(p[k],p[k+1]):issues.append(f'leg {k} endpoints')
    if t+1e-8<prev or t<0 or dt<.001 or t+dt>200+1e-8:issues.append(f'leg {k} time')
    actual=float(transfer_dv_cached(int(i),int(j),float(t),float(dt)))
    if not math.isfinite(actual) or actual>600+1e-6 or abs(actual-v)>.01:issues.append(f'leg {k} DV: stored {v:.3f}, recomputed {actual:.3f}')
    ex+=actual>100+1e-6
    prev=t+dt
if ex>5:issues.append(f'exceptions {ex}')
if abs(prev-d['arrival'])>1e-6:issues.append('arrival mismatch')
print('MISSING:',missing,'| ARRIVAL:',prev,'| EXCEPTIONS:',ex)
print('VALID:',not issues)
for issue in issues[:12]:print('ISSUE:',issue)
if not issues:
    out=Path('results/KTTSP_46_DAY183_VALIDATED.json')
    d['verified']=True;out.write_text(json.dumps(d,indent=2),encoding='utf-8')
    print('SAVED:',out)

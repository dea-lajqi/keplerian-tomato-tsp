"""Export the best independently validated Medium chromosome to ESA .sol format."""
import json
import os
from pathlib import Path

from validation import as_chromosome, baseline, official_udp, verify

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / 'results/esa/medium_overnight/run_20261122'


def main():
    udp=official_udp(ROOT)
    if (RUN/'best.json').is_file():
        saved=json.loads((RUN/'best.json').read_text(encoding='utf-8'))
        solution=saved['solution']
    else:
        solution,_=baseline(ROOT,udp)
    f=verify(udp,solution)
    out=RUN/'best_medium_esa_validated.sol'
    out.parent.mkdir(parents=True,exist_ok=True)
    lines=[format(float(x),'.16g') for x in as_chromosome(solution)]
    if len(lines)!=541:
        raise ValueError(f'Expected 541 values, got {len(lines)}')
    temp=out.with_name('.'+out.name+'.tmp')
    temp.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    os.replace(temp,out)
    print('Original ESA Medium fitness:',f)
    print('Official .sol chromosome:',out)
    print('Values:',len(lines))

if __name__=='__main__':main()

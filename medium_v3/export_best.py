#!/usr/bin/env python3
"""ESA recheck + export the current best to a submission-compatible .sol."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from medium_overnight.validation import official_udp,verify,as_chromosome
from medium_v3.run import ROOT,DEFAULT,seed_from_files

def main():
    udp=official_udp(ROOT)
    sol,fit,source=seed_from_files(udp,DEFAULT)
    # Never overwrite original .sol, evaluator, or existing submission.
    out=DEFAULT/'exported_esa_verified_medium.sol'
    if out.exists():
        print('Already exists, not overwritten:',out)
        print('Current ESA fitness:',fit,'source:',source)
        return
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as fp:
        fp.write('c ESA Medium 181 tomatoes; original ESA evaluator fitness '+str(fit)+'\n')
        for x in as_chromosome(sol):fp.write(f'{x:.14g}\n')
    print('Wrote:',out)
    print('ESA fitness:',fit,'original source:',source)

if __name__=='__main__':
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
    main()

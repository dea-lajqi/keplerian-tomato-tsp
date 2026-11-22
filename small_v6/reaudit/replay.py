"""Replay V4 or V5 with the reverse directional filter corrected locally."""
import argparse
from pathlib import Path
import signal
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from global_dp_easy import Grid
import small_v4_windows as v4
import small_v5_excursions as v5


class CorrectedGrid(Grid):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.finish_day is not None and self.directional:
            self.z_speed=-self.z_speed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("version",choices=("v4","v5"))
    p.add_argument("--atlas",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seconds",type=float,default=115.)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--four-only",action="store_true")
    p.add_argument("--legacy",action="store_true",help="Keep the old screening for paired controls")
    p.add_argument("--phase",type=float)
    p.add_argument("--relaxed-source",action="store_true")
    a=p.parse_args()
    if a.version=="v4":
        if not a.legacy:
            v4.Grid=CorrectedGrid
        ns=argparse.Namespace(mode="join",seconds=a.seconds,power=6,joins=12,
            seed=a.seed,output=a.output,resume=a.atlas,topology=False,rounds=1,
            topology_per_group=1,widths=[8,10],phase=a.phase if a.phase is not None else .019,beam=110)
        search=v4.Search(ns)
    else:
        if not a.legacy:
            v5.Grid=CorrectedGrid
        ns=argparse.Namespace(atlas=a.atlas,output=a.output,resume=None,seconds=a.seconds,
            rounds=25,sources=5,seed=a.seed,phase=a.phase if a.phase is not None else .017,
            rotate=a.relaxed_source,static_seeds=0,four_only=a.four_only,method="meet")
        search=v5.Search(ns)
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    search.run()


if __name__=="__main__":
    main()

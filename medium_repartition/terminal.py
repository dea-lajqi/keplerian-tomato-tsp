"""Enrich the terminal phase atlas while enforcing hard ΔV caps on submissions."""
import argparse
from pathlib import Path

from medium_coupled.atlas_enrich import Enrich
from medium_repartition.caps import strict_schedule


class Terminal(Enrich):
    def submit(self, sol, method, plan):
        tightened = strict_schedule(self.udp,sol)
        if tightened is None:
            self.count["strict_rejections"] += 1
            return False
        return super().submit(tightened, method, plan)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for x in ("seed-record","directory","atlas"):
        p.add_argument("--"+x,type=Path,required=True)
    p.add_argument("--profiles",type=Path,nargs="+",required=True)
    p.add_argument("--seconds",type=float,default=175.)
    p.add_argument("--enrich-seconds",type=float,default=90.)
    p.add_argument("--epoch-bin",type=float,default=.16)
    p.add_argument("--polish-slack",type=float,default=.38)
    p.add_argument("--goal",type=float,default=182.25416)
    p.set_defaults(mode="strict_terminal_enrich",anchors=(),baseline_record=None,
                   phase=None,beam=None)
    Terminal(p.parse_args()).run()

"""Ekzekuton nje prove Easy te arkivuar pa prishur importet.
Shembull: python run_easy_experiment.py kttsp_47_search.py
"""
import runpy
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
folder = root / "experiments" / "easy"
if len(sys.argv) < 2:
    print("Perdorimi: python run_easy_experiment.py EMRI_I_SKRIPTIT.py [argumente]")
    sys.exit(2)
name = Path(sys.argv[1]).name
script = folder / name
if not script.is_file():
    print("Nuk u gjet:", script)
    sys.exit(2)
sys.path.insert(0, str(root))
sys.path.insert(0, str(folder))
sys.argv = [str(script)] + sys.argv[2:]
runpy.run_path(str(script), run_name="__main__")

"""Organizim i sigurt i projektit KTTSP, pa fshire zgjidhjet.

Vendose kete file ne rrenjen e projektit (ku ndodhet read_data.py).
    python organizo_kttsp.py             # backup dhe organizim
    python organizo_kttsp.py --dry-run   # tregon cfare do ndryshohet
    python organizo_kttsp.py --undo      # rikthen skriptet e levizura

Nuk ndryshon asnje file ekzistues .py, .json, .kttsp ose README.md.
Skriptet eksperimente zhvendosen vetem pasi krijohet backup ZIP.
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import shutil
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EASY = ROOT / "experiments" / "easy"
MEDIUM = ROOT / "experiments" / "medium"
BACKUPS = ROOT / "backups"
MANIFEST = EASY / "organization_manifest.json"

# Keto skedare mbeten ne rrenjen e projektit sepse perdoren direkt
# nga programet ose jane versione kryesore te punes Easy.
KEEP = {
    "algorithm.py", "beam_search.py", "read_data.py", "orbital.py",
    "lambert.py", "utils_users.py", "organizo_kttsp.py",
    "kttsp_complete_solver.py", "kttsp_search_until_49.py",
    "kttsp_49_dynamic_programming.py", "kttsp_reach_49.py",
    "kttsp_reach_49_steps.py", "kttsp_easy_49_full.py",
    "run_easy_experiment.py",
}
EXPERIMENT_PATTERNS = (
    "kttsp_*.py", "optimize_*.py", "targeted_*.py",
    "multi_start_*.py", "insert_all_edges.py",
    "pdf_local_search*.py", "convert_easy_for*.py",
    "validate_46_fast.py",
)

RUNNER_CONTENT = r'''"""Ekzekuton nje prove Easy te arkivuar pa prishur importet.
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
'''

SUMMARY = """# KTTSP - Easy | Shenime te projektit

## Rezultati i ruajtur
- `results/KTTSP_49_PYKEP_VALIDATED.json`: zgjidhja 49/49 e raportuar
  si e verifikuar lokalisht me PyKEP (kontrolli zyrtar nuk eshte konfirmuar).
- `results/KTTSP_46_DAY183_VALIDATED.json`: zgjidhja e meparshme 46/49.
- `data/easy.kttsp`: te dhenat origjinale te Easy.

## Si eshte punuar
1. Leximi i te dhenave dhe gjendjeve orbitale (`read_data.py`, `orbital.py`).
2. Beam Search dhe kerkime heuristike per nje rruge fillestare.
3. Permiresim i renditjes nga 46/49 drejt 49/49 me ndryshime te udhezuara.
4. Kerkim / optimizim i koheve te transferimeve dhe verifikim me PyKEP.

**Kujdes:** `kttsp_reach_49.py` rinderton rrugen e njohur. Skriptet
`kttsp_complete_solver.py` dhe `kttsp_search_until_49.py` kerkojne
rruge, por ende nuk eshte vertetuar qe ato arrijne 49/49 nga zero.
Mos pretendo se 49/49 gjendet automatikisht nga zero pa nje prove te tille.

## Skedaret kryesore
- `beam_search.py`: baza e kerkimit me beam.
- `kttsp_complete_solver.py`: kerkuesi i pergjithshem eksperimental.
- `kttsp_search_until_49.py`: kerkuesi pa limit kohe.
- `kttsp_49_dynamic_programming.py`: optimizim i koheve nga nje rruge baze.
- `kttsp_reach_49.py`: rindertimi dhe kontrolli me PyKEP.
- `experiments/easy/`: provat e tjera, te ruajtura per dokumentim.

Per te ekzekutuar nje skript nga eksperimente/easy, perdor:
`python run_easy_experiment.py EMRI.py`

## Medium
- `data/medium.kttsp` ruhet me vete. Mos nderro parametrat e Easy
  ne kodet ekzistuese; Medium ka limite te ndryshme.

## Siguria
- Rezultatet ekzistuese ne `results/` nuk levizen ose mbishkruhen.
- Nje backup ZIP krijohet ne `backups/` para cdo organizimi.
- Per te rikthyer skriptet ne rrenje: `python organizo_kttsp.py --undo`.
"""


def check_root():
    missing = [p for p in ("read_data.py", "orbital.py", "data/easy.kttsp")
               if not (ROOT / p).is_file()]
    if missing:
        raise RuntimeError("Nuk je ne folderin kryesor te projektit. Mungojne: "
                           + ", ".join(missing))


def root_imports(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    except (UnicodeError, SyntaxError, OSError):
        return set()
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(x.name.split(".")[0] for x in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    return found


def find_movables() -> list[Path]:
    py = list(ROOT.glob("*.py"))
    stems = {p.stem: p for p in py}
    protected = set(KEEP)
    # Mos leviz skedar qe importohet nga nje nga modulet e mbrojtura.
    for _ in range(len(py) + 1):
        old = len(protected)
        for name in tuple(protected):
            p = ROOT / name
            if p.exists():
                for module in root_imports(p):
                    if module in stems:
                        protected.add(stems[module].name)
        if len(protected) == old:
            break
    selected = []
    for p in py:
        if p.name in protected or "medium" in p.name.lower():
            continue
        if any(fnmatch.fnmatch(p.name, pattern) for pattern in EXPERIMENT_PATTERNS):
            selected.append(p)
    return sorted(selected, key=lambda p: p.name.lower())


def backup_zip() -> Path:
    BACKUPS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = BACKUPS / f"KTTSP_BEFORE_ORGANIZE_{stamp}.zip"
    if path.exists():
        path = BACKUPS / f"KTTSP_BEFORE_ORGANIZE_{stamp}_{time.time_ns()}.zip"
    files = []
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT)
        if any(part in {".git", ".venv", "venv", "__pycache__", "backups", ".vscode"}
               for part in rel.parts):
            continue
        # Backup i skripteve, dataseteve, rezultateve dhe dokumentimit.
        if p.suffix.lower() in {".py", ".json", ".kttsp", ".md", ".txt", ".csv", ".ipynb"} or p.name == ".gitignore":
            if p.stat().st_size <= 100 * 1024 * 1024:
                files.append(p)
    if not files:
        raise RuntimeError("S'u gjet asnje skedar per backup")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, arcname=str(p.relative_to(ROOT)))
    with zipfile.ZipFile(path) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError(f"Backup ZIP i demtuar: {bad}")
    return path


def organize(dry_run: bool):
    check_root()
    if MANIFEST.exists():
        print("EASY eshte organizuar me pare. Nuk leviza asgje.")
        print("Per rikthim: python organizo_kttsp.py --undo")
        return
    moves = find_movables()
    print("PROJEKTI:", ROOT)
    print("Skripte eksperimenti per levizje:", len(moves))
    for p in moves:
        print("   ", p.name, "-> experiments/easy/")
    collisions = [p.name for p in moves if (EASY / p.name).exists()]
    if collisions:
        raise RuntimeError("Ekzistojne skedare me emra te njejte; asgje nuk u leviz: "
                           + ", ".join(collisions))
    if dry_run:
        print("DRY RUN: Nuk eshte ndryshuar asnje skedar.")
        return
    backup = backup_zip()
    print("BACKUP I KRIJUAR:", backup)
    EASY.mkdir(parents=True, exist_ok=True)
    MEDIUM.mkdir(parents=True, exist_ok=True)
    moved = []
    try:
        for p in moves:
            shutil.move(str(p), str(EASY / p.name))
            moved.append(p.name)
    except Exception:
        for name in reversed(moved):
            shutil.move(str(EASY / name), str(ROOT / name))
        raise
    manifest = {
        "backup": str(backup.relative_to(ROOT)),
        "date": datetime.now().isoformat(timespec="seconds"),
        "moved": moved,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    runner = ROOT / "run_easy_experiment.py"
    if not runner.exists():
        runner.write_text(RUNNER_CONTENT, encoding="utf-8")
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    notes = docs / "EASY_SUMMARY.md"
    if not notes.exists():
        notes.write_text(SUMMARY, encoding="utf-8")
    best = ROOT / "results" / "KTTSP_49_PYKEP_VALIDATED.json"
    if best.is_file():
        copied = ROOT / "results" / "easy" / best.name
        copied.parent.mkdir(parents=True, exist_ok=True)
        if not copied.exists():
            shutil.copy2(best, copied)
    print("\nU KRYE: u levizen", len(moved), "skripte ne experiments/easy/")
    print("Te paprekur: data/, results/ origjinale dhe kodet kryesore.")
    print("Shenimet:", notes.relative_to(ROOT))
    print("Rikthim (nese duhet): python organizo_kttsp.py --undo")


def undo():
    if not MANIFEST.exists():
        print("S'ka manifest organizimi per ta rikthyer.")
        return
    meta = json.loads(MANIFEST.read_text(encoding="utf-8"))
    moved = meta.get("moved", [])
    collisions = [name for name in moved if (ROOT / name).exists()]
    if collisions:
        raise RuntimeError("Nuk mund te bej rikthim: keto emra ekzistojne ne rrenje: "
                           + ", ".join(collisions))
    missing = [name for name in moved if not (EASY / name).is_file()]
    if missing:
        raise RuntimeError("Mungojne skriptet e arkivuara: " + ", ".join(missing))
    for name in moved:
        shutil.move(str(EASY / name), str(ROOT / name))
    MANIFEST.unlink()
    print("U RIKTHYEN", len(moved), "skripte ne rrenjen e projektit.")
    print("Backup ZIP, shenimet dhe rezultatet mbeten te ruajtura.")


def main():
    parser = argparse.ArgumentParser(description="Organizon KTTSP Easy me backup dhe Undo")
    x = parser.add_mutually_exclusive_group()
    x.add_argument("--dry-run", action="store_true", help="Shfaq ndryshimet pa levizur skedare")
    x.add_argument("--undo", action="store_true", help="Kthen skriptet ne vend")
    args = parser.parse_args()
    try:
        if args.undo:
            undo()
        else:
            organize(args.dry_run)
    except (OSError, ValueError, RuntimeError) as e:
        print("GABIM:", e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Riparon oraret e rrugëve KTTSP sipas vlerësuesve zyrtarë të ESA-s."""

import argparse
import importlib
import importlib.util
import json
import math
import re
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parent
SOURCES = {
    "easy": ("small", "KTTSP_49_FULL_MISSION.json"),
    "medium": ("medium", "KTTSP_MEDIUM_BEST_VERIFIED.json"),
}


def load_evaluator(directory, problem, native_core):
    """Ngarkon kodin e ESA-s dhe kërkon PyKEP 2.6.4."""
    if native_core and "pykep" not in sys.modules:
        # Në Windows ngarkojmë vetëm modulet orbitale; ato përdorin të
        # njëjtin kod C++ si importi i plotë i PyKEP 2.6.4.
        spec = importlib.util.find_spec("pykep")
        if spec is None or not re.search(
            r"(?m)^version\s*=\s*['\"]2\.6\.4['\"]",
            Path(spec.origin).read_text(encoding="utf-8"),
        ):
            raise RuntimeError("Kërkohet PyKEP 2.6.4.")
        package = types.ModuleType("pykep")
        package.__path__ = list(spec.submodule_search_locations)
        package.version = "2.6.4"
        sys.modules["pykep"] = package
        core = importlib.import_module("pykep.core")
        package.epoch, package.DAY2SEC = core.epoch, core.DAY2SEC
        package.lambert_problem = core.lambert_problem
        package.planet = importlib.import_module("pykep.planet")
    else:
        import pykep
        if getattr(pykep, "version", None) != "2.6.4":
            raise RuntimeError("Kërkohet PyKEP 2.6.4.")

    spec = importlib.util.spec_from_file_location("esa_kttsp", directory / f"{problem}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.udp


def chromosome(path, legs):
    """Ndërton vektorin me nisjet, fluturimet dhe rendin e vizitave."""
    return [leg[2] for leg in legs] + [leg[3] for leg in legs] + list(path)


def validate(udp, path, legs):
    """Thërret fitness zyrtar dhe kontrollon kufijtë e vektorit dhe afatin."""
    if sorted(path) != list(range(udp.n_tomatoes)):
        return None
    if [(leg[0], leg[1]) for leg in legs] != list(zip(path, path[1:])):
        return None
    x = chromosome(path, legs)
    lower, upper = udp.get_bounds()
    if not all(math.isfinite(value) and lo <= value <= hi
               for value, lo, hi in zip(x, lower, upper)):
        return None
    fitness = [float(value) for value in udp.fitness(x)]
    equalities = fitness[1:1 + udp.get_nec()]
    inequalities = fitness[1 + udp.get_nec():]
    valid = (fitness[0] <= udp.max_time and all(value == 0 for value in equalities)
             and all(value <= 0 for value in inequalities))
    return {"fitness": fitness, "valid": valid}


def try_leg(udp, old, earliest, limit, step, deadline, prefer_original):
    """Gjen një orar me Lambert zyrtar, duke provuar kohë pritjeje dhe fluturimi."""
    source, target, old_departure, old_flight, _ = old
    if prefer_original and old_departure >= earliest:
        if udp.min_tof <= old_flight and old_departure + old_flight <= udp.max_time:
            cost = float(udp.compute_transfer(source, target, old_departure, old_flight))
            if math.isfinite(cost) and cost <= limit:
                return source, target, old_departure, old_flight, cost
    start = math.ceil(earliest / step)
    horizon = min(udp.max_time, earliest + 24)
    max_flight_ticks = int(16 / step)
    for end in range(start + 1, math.floor(horizon / step) + 1):
        if time.monotonic() >= deadline:
            return None
        for tick in range(end - 1, max(start, end - max_flight_ticks) - 1, -1):
            flight = (end - tick) * step
            if flight < udp.min_tof:
                continue
            departure = tick * step
            cost = float(udp.compute_transfer(source, target, departure, flight))
            if math.isfinite(cost) and cost <= limit:
                return source, target, departure, flight, cost
    return None


def retime(udp, legs, step, deadline, prefer_original):
    """Rindërton oraret duke rezervuar përjashtimet për kalimet e vështira."""
    schedule = []
    day = float(udp.t0.mjd2000)
    exceptions = 0
    # Kostot historike shërbejnë vetëm për të rezervuar buxhetin.
    planned = sorted(range(len(legs)), key=lambda i: legs[i][4], reverse=True)
    planned = set(planned[:udp.n_dv_exception])
    for index, leg in enumerate(legs):
        if time.monotonic() >= deadline:
            break
        allow_exception = index in planned and exceptions < udp.n_dv_exception
        limit = udp.dv_threshold_exception if allow_exception else udp.dv_threshold
        replacement = try_leg(udp, leg, day, limit, step, deadline, prefer_original)
        if replacement is None:
            break
        schedule.append(replacement)
        day = replacement[2] + replacement[3]
        exceptions += replacement[4] > udp.dv_threshold
    return schedule


def run(args):
    """Provon orare të ndryshme dhe ruan veçmas vetëm rrugët e pranuara."""
    problem, filename = SOURCES[args.instance]
    udp = load_evaluator(args.official_dir, problem, args.native_core)
    saved = json.loads((args.seed_file or ROOT / "results" / filename).read_text(encoding="utf-8"))
    path, original_legs = saved["path"], saved["transfers"]
    baseline = validate(udp, path, original_legs)
    print(f"{args.instance} | hyrja sipas ESA: {baseline}", flush=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    output = ROOT / "results" / "esa"
    output.mkdir(exist_ok=True)
    attempts = []
    best = math.inf
    started = time.monotonic()
    deadline = started + args.seconds
    for direction in (False, True):
        order = list(reversed(path)) if direction else path
        legs = [(b, a, t, tof, dv) for a, b, t, tof, dv in reversed(original_legs)] if direction else original_legs
        for step, retain in ((.25, True), (.25, False), (.125, False), (.0625, False)):
            if time.monotonic() >= deadline:
                break
            # Një provë e vështirë nuk mund të harxhojë të gjithë buxhetin.
            attempt_deadline = min(deadline, time.monotonic() + args.seconds / 4)
            candidate = retime(udp, legs, step, attempt_deadline, retain and not direction)
            checked = validate(udp, order, candidate) if len(candidate) == len(order) - 1 else None
            attempt = {"reverse": direction, "step_days": step, "prefer_original": retain and not direction,
                       "scheduled_visits": len(candidate) + 1, "validation": checked,
                       "elapsed_seconds": round(time.monotonic() - started, 3)}
            attempts.append(attempt)
            if checked and checked["valid"] and checked["fitness"][0] < best:
                best = checked["fitness"][0]
                with (output / f"{args.instance}_{run_id}_{len(attempts)}_verified.json").open("x", encoding="utf-8") as file:
                    json.dump({"path": order, "transfers": candidate, "arrival_day": best,
                               "exceptions": sum(leg[4] > udp.dv_threshold for leg in candidate),
                               "validation": checked, "attempt": attempt}, file, indent=2)
            print(f"Prova {len(attempts)} | {attempt}", flush=True)
    with (output / f"{args.instance}_{run_id}_experiment.json").open("x", encoding="utf-8") as file:
        json.dump({"instance": args.instance, "budget_seconds": args.seconds,
                   "baseline_validation": baseline, "attempts": attempts,
                   "best_official_days": best if math.isfinite(best) else None}, file, indent=2)
    print(f"Rezultati zyrtar më i mirë: {best if math.isfinite(best) else 'asnjë rrugë e pranuar'}")


def main():
    """Lexon instancën, dosjen zyrtare dhe buxhetin e provës."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", choices=SOURCES)
    parser.add_argument("--official-dir", type=Path, required=True)
    parser.add_argument("--seed-file", type=Path)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--native-core", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds duhet të jetë pozitiv dhe i fundëm.")
    run(args)


if __name__ == "__main__":
    main()

"""Search time-dependent route windows and validate full missions with ESA."""

import argparse
from bisect import bisect_left
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import json
import math
from pathlib import Path
import random
import time
from uuid import uuid4

from scipy.optimize import minimize

from kttsp_esa_solver import ROOT, load_evaluator, validate

SEED = ROOT / "results/esa/easy_refined_20261121T091447Z.json"


def arrival(leg):
    return leg[2] + leg[3]


class BudgetExpired(Exception):
    pass


class Transfers:
    def __init__(self, udp, deadline):
        self.udp = udp
        self.deadline = deadline
        self.calls = 0
        self.cost = lru_cache(maxsize=500_000)(self._cost)

    def check_time(self):
        if time.monotonic() >= self.deadline:
            raise BudgetExpired

    def _cost(self, source, target, departure, flight):
        self.calls += 1
        if self.calls % 64 == 0:
            self.check_time()
        return float(self.udp.compute_transfer(source, target, departure, flight))

    def polish(self, legs, start):
        """Minimize each arrival in turn, with a feasible fallback on every leg."""
        result = []
        earliest = start
        for leg in legs:
            self.check_time()
            a, b, dep, tof, dv = leg
            limit = (self.udp.dv_threshold if dv <= self.udp.dv_threshold
                     else self.udp.dv_threshold_exception + 1e-6)
            finish = arrival(leg)
            if dep >= earliest and earliest + self.udp.min_tof < finish:
                fit = minimize(
                    lambda x: x[0] + x[1], (dep, tof), method="SLSQP",
                    bounds=[(earliest, finish - self.udp.min_tof),
                            (self.udp.min_tof, finish - earliest)],
                    constraints=[{"type": "ineq", "fun": lambda x: (
                        limit - 1e-7 - self.cost(a, b, float(x[0]), float(x[1]))
                    )}],
                    options={"maxiter": 25, "ftol": 1e-10, "eps": 1e-7},
                )
                new_dep, new_tof = map(float, fit.x)
                cost = self.cost(a, b, new_dep, new_tof)
                if (new_dep >= earliest and new_tof >= self.udp.min_tof
                        and cost <= limit and new_dep + new_tof < finish):
                    leg = (a, b, new_dep, new_tof, cost)
            result.append(leg)
            earliest = arrival(leg)
        return result


class Window:
    """Subset DP over a sampled transfer graph, with free waiting at vertices."""

    def __init__(self, transfers, path, legs, left, right, step, phase, flight_phase, cap):
        self.transfers = transfers
        self.udp = transfers.udp
        self.nodes = path[left:right]
        self.prefix = legs[:max(0, left - 1)]
        self.suffix = legs[right:] if right < len(path) else []
        self.old = {(leg[0], leg[1]): leg for leg in legs}
        self.source = path[left - 1] if left else None
        self.target = path[right] if right < len(path) else None
        self.start = arrival(self.prefix[-1]) if self.prefix else float(self.udp.t0.mjd2000)
        self.end = self.suffix[0][2] + 1e-6 if self.suffix else cap
        self.budget = self.udp.n_dv_exception - sum(
            self.exception(leg) for leg in self.prefix + self.suffix
        )
        self.step = step
        self.phase = phase
        self.flight_phase = flight_phase
        self.tables = {}

    def exception(self, leg):
        return int(leg[4] > self.udp.dv_threshold)

    def catalogue(self, a, b):
        """Index the best arrival at or after each sampled departure by class."""
        if (a, b) in self.tables:
            return self.tables[a, b]
        self.transfers.check_time()
        step = self.step
        ticks = range(math.ceil((self.start - self.phase) / step),
                      math.floor((self.end - self.phase) / step) + 1)
        departures = {tick * step + self.phase for tick in ticks}
        departures.add(self.start)
        old = self.old.get((a, b))
        if old and self.start - 1e-6 <= old[2] and arrival(old) <= self.end:
            departures.add(old[2])
        departures = sorted(departures)
        regular_flights = sorted({self.udp.min_tof, 0.01, 0.025, 0.0625} | {
            tick * step + self.flight_phase
            for tick in range(1, math.floor(min(10, self.end - self.start) / step) + 1)
        })
        flights = sorted(set(regular_flights) | {old[3]}) if old else regular_flights
        best = [None, None]
        rows = []
        max_cost = self.udp.dv_threshold_exception + 1e-6
        for dep in reversed(departures):
            # Once both classes have a feasible earlier arrival, later landings
            # from this departure cannot change the suffix minima.
            if old and dep == old[2]:
                extra = self.exception(old)
                if best[extra] is None or arrival(old) < arrival(best[extra]):
                    best[extra] = old
            for tof in flights:
                finish = dep + tof
                if finish > self.end:
                    break
                relevant = best if self.budget else best[:1]
                if all(leg is not None and arrival(leg) <= finish for leg in relevant):
                    break
                cost = self.transfers.cost(a, b, dep, tof)
                if cost > max_cost or not math.isfinite(cost):
                    continue
                extra = int(cost > self.udp.dv_threshold)
                if best[extra] is None or finish < arrival(best[extra]):
                    best[extra] = (a, b, dep, tof, cost)
            rows.append(tuple(best))
        table = (departures, rows[::-1])
        self.tables[a, b] = table
        return table

    def options(self, a, b, day):
        departures, rows = self.catalogue(a, b)
        index = bisect_left(departures, day - 1e-6)
        return rows[index] if index < len(rows) else ()

    def solve(self):
        """Return the earliest completion for each exception count and order class."""
        size = len(self.nodes)
        full = (1 << size) - 1
        # Key = visited subset, last node, exceptions used, differs from seed.
        layers = [{} for _ in range(size + 1)]
        layers[0][(0, self.source, 0, False)] = (self.start, [])
        for depth in range(size):
            self.transfers.check_time()
            for (mask, last, used, changed), (day, chain) in layers[depth].items():
                for index, node in enumerate(self.nodes):
                    bit = 1 << index
                    if mask & bit:
                        continue
                    changed_order = changed or node != self.nodes[depth]
                    options = self.options(last, node, day) if last is not None else (None,)
                    for leg in options:
                        if leg is None and last is not None:
                            continue
                        count = used + (self.exception(leg) if leg else 0)
                        finish = arrival(leg) if leg else day
                        remaining_legs = size - depth - 1 + (self.target is not None)
                        if count > self.budget or finish + remaining_legs * (self.udp.min_tof - 1e-6) > self.end:
                            continue
                        key = (mask | bit, node, count, changed_order)
                        previous = layers[depth + 1].get(key)
                        if previous is None or finish < previous[0]:
                            layers[depth + 1][key] = (finish, chain + [leg])
        solutions = {}
        for (mask, last, used, changed), (day, chain) in layers[-1].items():
            assert mask == full
            options = self.options(last, self.target, day) if self.target is not None else (None,)
            for leg in options:
                if leg is None and self.target is not None:
                    continue
                count = used + (self.exception(leg) if leg else 0)
                finish = arrival(leg) if leg else day
                if count > self.budget:
                    continue
                key = (count, changed)
                if key not in solutions or finish < solutions[key][0]:
                    patch = [item for item in chain + [leg] if item is not None]
                    solutions[key] = (finish, self.prefix + patch + self.suffix)
        return [legs for _, legs in sorted(solutions.values(), key=lambda state: state[0])]


def order(legs):
    return [legs[0][0], *(leg[1] for leg in legs)]


def write_json(path, payload):
    temporary = path.with_suffix(".tmp")
    with temporary.open("x", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, allow_nan=False)
    temporary.replace(path)


def run(args):
    udp = load_evaluator(ROOT / "esa_evaluation", "small", args.native_core)
    seed = json.loads(args.seed_file.read_text(encoding="utf-8"))
    baseline = validate(udp, seed["path"], seed["transfers"])
    if not baseline or not baseline["valid"]:
        raise ValueError("ESA rejected the seed")
    started = time.monotonic()
    transfers = Transfers(udp, started + args.seconds)
    rng = random.Random(args.random_seed)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    feasible_routes = set()
    seen = set()
    history = []
    best_days = baseline["fitness"][0]
    last_progress = started
    original = seed["path"]
    working = [
        (*leg[:4], transfers.cost(*leg[:4])) for leg in seed["transfers"]
    ]
    seen.add(tuple(leg[:4] for leg in working))
    best_changed = None
    print(f"ESA baseline {baseline['fitness']} | run {run_id}", flush=True)

    def submit(candidate, attempt):
        nonlocal best_days, best_changed, last_progress
        signature = tuple(leg[:4] for leg in candidate)
        if signature in seen:
            stats["duplicate_schedules"] += 1
            return None
        seen.add(signature)
        path = order(candidate)
        checked = validate(udp, path, candidate)
        stats["esa_checks"] += 1
        if not checked or not checked["valid"]:
            stats["esa_rejected"] += 1
            return None
        days = checked["fitness"][0]
        stats["esa_feasible"] += 1
        feasible_routes.add(tuple(path))
        if path != original:
            stats["esa_feasible_changed_order"] += 1
            if best_changed is None or days < best_changed:
                best_changed = days
        if days < best_days:
            if best_days - days > 1e-6:
                last_progress = time.monotonic()
            best_days = days
            stats["improvements"] += 1
            filename = output / f"easy_pareto_{run_id}_{stats['improvements']:04d}.json"
            write_json(filename, {"path": path, "transfers": candidate, "arrival_day": days,
                                  "validation": checked, "attempt": attempt})
            print(f"ESA improvement {days:.12f} days | {filename}", flush=True)
        return days

    def score(candidate):
        return (arrival(candidate[-1]), sum(leg[4] > udp.dv_threshold for leg in candidate),
                sum(map(arrival, candidate)))

    # Start at the open end, then alternate it with windows around long legs
    # and exceptions, and windows sampled across the full route.
    stop_reason = "error"
    try:
        stats["polish_attempts"] += 1
        polished = transfers.polish(working, float(udp.t0.mjd2000))
        days = submit(polished, {"method": "forward continuous polish"})
        history.append({"method": "forward continuous polish", "valid_days": days})
        if days is not None and score(polished) < score(working):
            working = polished
        pool = {tuple(order(working)): working}
        while True:
            transfers.check_time()
            if args.stall_seconds and time.monotonic() - last_progress >= args.stall_seconds:
                stop_reason = "stagnation"
                break
            index = stats["window_attempts"]
            size = rng.choice((4, 4, 5, 5, 6, 7, 8))
            incumbent = working if index % 3 else rng.choice(list(pool.values()))
            path = order(incumbent)
            if index % 4 == 0:
                left = len(path) - size
            elif index % 4 == 1:
                focus = rng.choice(sorted(
                    range(len(incumbent)),
                    key=lambda i: (incumbent[i][4] > udp.dv_threshold, incumbent[i][3]),
                    reverse=True,
                )[:15])
                left = min(len(path) - size, max(0, focus - size // 2))
            else:
                left = rng.randrange(len(path) - size + 1)
            right = left + size
            step = rng.choice((0.125, 0.125, 0.1875, 0.25, 0.0625))
            phase = 0.0 if index % 3 == 0 else rng.random() * step
            flight_phase = rng.random() * step if index % 3 == 1 else 0.0
            # Allow occasional slower open-end orders into the exploration pool.
            cap = best_days
            if right == len(path):
                cap = min(udp.max_time, max(cap, arrival(incumbent[-1])) + (3 if index % 8 == 0 else 0))
            attempt = {"method": "subset DP", "window": [left, right], "nodes": path[left:right],
                       "step_days": step, "departure_phase_days": phase,
                       "flight_phase_days": flight_phase, "open_end_cap_days": cap if right == len(path) else None}
            history.append(attempt)
            stats["window_attempts"] += 1
            window = Window(transfers, path, incumbent, left, right, step, phase, flight_phase, cap)
            candidates = window.solve()
            stats["windows_completed"] += 1
            attempt["candidates"] = len(candidates)
            attempt["valid_days"] = []
            for candidate in candidates:
                days = submit(candidate, attempt)
                if days is None:
                    continue
                attempt["valid_days"].append(days)
                if score(candidate) < score(working):
                    working = candidate
                if order(candidate) != path:
                    key = tuple(order(candidate))
                    if key not in pool or score(candidate) < score(pool[key]):
                        pool[key] = candidate
                    if len(pool) > 24:
                        del pool[max(pool, key=lambda route: score(pool[route]))]
                # Continuous refinement of frontier schedules, including
                # routes whose unchanged suffix still masks earlier arrivals.
                stats["polish_attempts"] += 1
                polished = transfers.polish(candidate, float(udp.t0.mjd2000))
                polished_days = submit(polished, {**attempt, "method": "DP + continuous polish"})
                if polished_days is not None and score(polished) < score(working):
                    working = polished
            if stats["windows_completed"] % 10 == 0:
                print(f"{dict(stats)} | best {best_days:.12f} | {time.monotonic()-started:.1f}s", flush=True)
    except BudgetExpired:
        stop_reason = "time budget"
    except KeyboardInterrupt:
        stop_reason = "interrupted"
    finally:
        report = {
            "baseline": baseline, "seed": str(args.seed_file), "random_seed": args.random_seed,
            "budget_seconds": args.seconds, "elapsed_seconds": time.monotonic() - started,
            "stop_reason": stop_reason,
            "counts": dict(stats), "official_transfer_calls": transfers.calls,
            "distinct_feasible_candidate_orders": len(feasible_routes),
            "best_official_days": best_days, "best_changed_order_days": best_changed,
            "attempts": history,
        }
        filename = output / f"easy_pareto_{run_id}_summary.json"
        write_json(filename, report)
        print(f"{dict(stats)} | best {best_days:.12f} | report {filename}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--stall-seconds", type=float, default=0,
                        help="stop after this long without a gain over 1e-6 days (0 disables)")
    parser.add_argument("--seed-file", type=Path, default=SEED)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/esa")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--native-core", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be positive and finite")
    if not math.isfinite(args.stall_seconds) or args.stall_seconds < 0:
        parser.error("--stall-seconds must be nonnegative and finite")
    run(args)


if __name__ == "__main__":
    main()

"""Kërkim dhe verifikim i rrugëve për instancën Medium të KTTSP."""

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import pykep as pk

from orbital import DAY2SEC, MU_MOON, orbital_state_corrected
from read_data import read_instance


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "medium.kttsp"
OUTPUT = ROOT / "results" / "KTTSP_MEDIUM_BEST_VERIFIED.json"
HISTORY = OUTPUT.parent / "KTTSP_MEDIUM_HISTORY.csv"


@dataclass(frozen=True)
class Instance:
    orbits: list
    epoch: float
    min_flight: float
    duration: float
    normal_dv: float
    max_dv: float
    max_exceptions: int

    @property
    def end(self):
        """Kthen afatin absolut nga epoka dhe kohëzgjatja e misionit."""
        return self.epoch + self.duration


@dataclass(frozen=True)
class Route:
    path: tuple
    legs: tuple
    arrival: float
    exceptions: int


def lambert_options(start, finish, flight):
    """Jep shpejtësitë Lambert dhe koston për të dy drejtimet orbitale."""
    r0, v0 = start
    r1, v1 = finish
    for clockwise in (False, True):
        try:
            problem = pk.lambert_problem(r0, r1, flight * DAY2SEC, MU_MOON, clockwise, 20)
        except (ValueError, RuntimeError, OverflowError):
            continue
        for departure, arrival in zip(problem.v0, problem.v1):
            cost = math.dist(departure, v0) + math.dist(arrival, v1)
            if math.isfinite(cost):
                yield cost, departure, arrival


def verify(instance, path, legs):
    """Rillogarit kohët, orbitat dhe çdo ΔV nga fillimi, pa cache të kërkimit."""
    if (
        not path
        or any(type(i) is not int or not 0 <= i < len(instance.orbits) for i in path)
        or len(set(path)) != len(path)
        or len(legs) != len(path) - 1
    ):
        raise ValueError("Rrugë ose numër transferimesh i pavlefshëm.")

    arrival = instance.epoch
    exceptions = 0
    checked = []
    for index, (source, target, departure, flight, _) in enumerate(legs):
        if (source, target) != tuple(path[index:index + 2]):
            raise ValueError(f"Transferimi {index + 1} nuk përputhet me rrugën.")
        finish = departure + flight
        if (
            not all(map(math.isfinite, (departure, flight, finish)))
            or departure < arrival
            or flight < instance.min_flight
            or finish > instance.end
        ):
            raise ValueError(f"Shkelje kohore në transferimin {index + 1}.")

        # PyKEP avancon nga epoka; kërkimi llogarit anomalinë në ditën e kërkuar.
        start_epoch = orbital_state_corrected(instance.orbits[source], instance.epoch, instance.epoch)
        end_epoch = orbital_state_corrected(instance.orbits[target], instance.epoch, instance.epoch)
        start = pk.propagate_lagrangian(start_epoch, (departure - instance.epoch) * DAY2SEC, MU_MOON)
        end = pk.propagate_lagrangian(end_epoch, (finish - instance.epoch) * DAY2SEC, MU_MOON)
        options = sorted(lambert_options(start, end, flight), key=lambda option: option[0])
        for cost, velocity, expected_velocity in options:
            if cost > instance.max_dv:
                break
            try:
                position, final_velocity = pk.propagate_lagrangian(
                    (start[0], velocity), flight * DAY2SEC, MU_MOON
                )
            except (ValueError, RuntimeError, OverflowError):
                continue
            if math.dist(position, end[0]) <= 1.0 and math.dist(final_velocity, expected_velocity) <= 0.001:
                break
        else:
            raise ValueError(f"Nuk u verifikua transferimi {index + 1}.")
        if cost > instance.max_dv:
            raise ValueError(f"ΔV mbi kufi në transferimin {index + 1}.")
        exceptions += cost > instance.normal_dv
        if exceptions > instance.max_exceptions:
            raise ValueError(f"Shumë përjashtime në transferimin {index + 1}.")
        checked.append((source, target, departure, flight, cost))
        arrival = finish
    return Route(tuple(path), tuple(checked), arrival, exceptions)


def route_value(route):
    """Krahason rekordet sipas numrit të vizitave, kohës dhe përjashtimeve."""
    return len(route.path), -route.arrival, -route.exceptions


class Experiment:
    """Arkivon rezultatet e verifikuara dhe lidh çdo përmirësim me metodën."""

    def __init__(self, instance, seconds, seed):
        """Jep një identifikues të ri dhe ruan parametrat e ekzekutimit."""
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = f"{stamp}_{uuid4().hex[:8]}"
        self.instance = instance
        self.seconds = seconds
        self.seed = seed
        self.sequence = 0
        self.attempts = []
        self.parameters = {}
        self.best_source = {}

    def export(self, route, elapsed, event, update_best=False):
        """Shkruan një JSON të ri dhe shton një rresht në historikun CSV."""
        instance = self.instance
        self.sequence += 1
        identifier = f"KTTSP_MEDIUM_{self.run_id}_{self.sequence:03d}_{event}"
        archive = OUTPUT.parent / f"{identifier}.json"
        method = self.parameters.get("method", event)
        if event in ("baseline", "improvement"):
            self.best_source = {"record": archive.name, "method": method}
        arrivals = [instance.epoch] + [leg[2] + leg[3] for leg in route.legs]
        normal_costs = [leg[4] for leg in route.legs if leg[4] <= instance.normal_dv]
        exception_costs = [leg[4] for leg in route.legs if leg[4] > instance.normal_dv]
        payload = {
            "instance": "data/medium.kttsp",
            "run_id": self.run_id,
            "event": event,
            "path": route.path,
            "transfers": route.legs,
            "visit_days": arrivals,
            "arrival_day": route.arrival,
            "mission_duration_days": route.arrival - instance.epoch,
            "exceptions": route.exceptions,
            "verified_pykep": True,
            "validation": {
                "checked_transfers": len(route.legs),
                "minimum_flight_days": min((leg[3] for leg in route.legs), default=None),
                "max_normal_dv_m_s": max(normal_costs, default=None),
                "max_exception_dv_m_s": max(exception_costs, default=None),
                "max_mission_days": instance.duration,
                "min_allowed_flight_days": instance.min_flight,
                "normal_limit_m_s": instance.normal_dv,
                "exception_limit_m_s": instance.max_dv,
                "exception_budget": instance.max_exceptions,
            },
            "seed": self.seed,
            "parameters": {"budget_seconds": self.seconds, **self.parameters},
            "computation_seconds": round(elapsed, 3),
            "best_source": self.best_source,
        }
        if event == "final":
            payload["attempts"] = self.attempts
        content = json.dumps(payload, indent=2) + "\n"
        with archive.open("x", encoding="utf-8") as file:
            file.write(content)
        with HISTORY.open("a+", newline="", encoding="utf-8") as file:
            file.seek(0, 2)
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow((
                    "run_id", "event", "method", "visited", "arrival_day", "mission_days",
                    "exceptions", "computation_seconds", "json",
                ))
            writer.writerow((
                self.run_id, event, method, len(route.path), route.arrival,
                route.arrival - instance.epoch, route.exceptions, round(elapsed, 3), archive.name,
            ))
        if update_best:
            save_best(content, identifier)
        return archive

    def resume(self, run_id):
        """Rihap historikun e së njëjtës provë dhe kthen kohën e kaluar."""
        with HISTORY.open(newline="", encoding="utf-8") as file:
            entries = [row for row in csv.DictReader(file) if row["run_id"] == run_id]
        if not entries or entries[-1]["event"] == "final":
            raise ValueError("Eksperimenti për vazhdim mungon ose ka përfunduar.")
        checkpoint = next(row for row in reversed(entries) if row["event"] in ("baseline", "improvement"))
        self.run_id = run_id
        self.sequence = len(entries)
        self.best_source = {"record": checkpoint["json"], "method": checkpoint["method"]}
        original_start = datetime.strptime(run_id.split("_")[0], "%Y%m%dT%H%M%SZ")
        return time.time() - original_start.replace(tzinfo=timezone.utc).timestamp() + 1


def save_best(content, identifier):
    """Përditëson kopjen për rinisje pasi JSON-i i arkivit është ruajtur."""
    temporary = OUTPUT.parent / f".{identifier}.tmp"
    temporary.write_text(content, encoding="utf-8")
    # Sinkronizimi i dosjes në Windows mund ta mbajë skedarin hapur për pak.
    for attempt in range(6):
        try:
            temporary.replace(OUTPUT)
            break
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * (attempt + 1))


class Search:
    """Mban të dhënat dhe ndërton rrugë brenda buxhetit të kohës."""

    def __init__(self, instance, seconds, seed, optimize=False, coverage=False, focused=False,
                 resume_experiment=None, initial_attempts=0):
        """Përgatit afatin, gjeneratorin e rinisjeve dhe renditjen e orbitave."""
        self.instance = instance
        self.started = time.monotonic()
        self.deadline = self.started + seconds
        self.rng = random.Random(seed)
        self.seed = seed
        self.optimize = optimize
        self.coverage = coverage
        self.focused = focused
        self.experiment = Experiment(instance, seconds, seed)
        self.resume_experiment = resume_experiment
        if resume_experiment:
            self.started -= self.experiment.resume(resume_experiment)
            self.deadline = self.started + seconds
        self.best = None
        self.attempts = initial_attempts
        self.neighbors = []
        self.distances = []
        # Normalja e planit dallon edhe orbitat me drejtime të kundërta.
        normals = [(math.sin(i) * math.sin(node), -math.sin(i) * math.cos(node), math.cos(i))
                   for _, _, i, node, _, _ in instance.orbits]
        for source, orbit in enumerate(instance.orbits):
            distances = [
                2 * abs(math.log(orbit[0] / target[0])) + math.dist(normals[source], normal)
                for target, normal in zip(instance.orbits, normals)
            ]
            self.distances.append(distances)
            self.neighbors.append(sorted(
                (target for target in range(len(normals)) if target != source),
                key=distances.__getitem__,
            ))

    @lru_cache(maxsize=50000)
    def state(self, tomato, day):
        """Mban në cache pozicionin dhe shpejtësinë për një domate në një ditë."""
        r, v = orbital_state_corrected(self.instance.orbits[tomato], day, self.instance.epoch)
        return r.tolist(), v.tolist()

    @lru_cache(maxsize=150000)
    def cost(self, source, target, departure, flight):
        """Llogarit ΔV më të vogël për nisjen dhe fluturimin e provuar."""
        start = self.state(source, departure)
        end = self.state(target, departure + flight)
        return min((cost for cost, _, _ in lambert_options(start, end, flight)), default=math.inf)

    def successors(self, route, flight_step, broad=False):
        """Provon pritje/fluturime dhe mban mbërritjen më të hershme për çdo lloj."""
        instance = self.instance
        source = route.path[-1]
        visited = set(route.path)
        targets = [target for target in self.neighbors[source] if target not in visited]
        for group, waits, max_flight in (
            (targets[:18 if broad else 10], (0, .25, .75, 1.5, 3) if broad else (0, .25, .75, 1.5),
             8 if broad else 5),
            (targets, (0, .5, 1.5, 3, 6), 12),
        ):
            found = {}
            flights = sorted({instance.min_flight, .0625, .125, .1875} | {
                step * flight_step for step in range(1, int(max_flight / flight_step) + 1)
            })
            for target in group:
                for wait in waits:
                    departure = route.arrival + wait
                    for flight in flights:
                        if time.monotonic() >= self.deadline:
                            return list(found.values())
                        finish = departure + flight
                        if finish > instance.end:
                            break
                        if flight < instance.min_flight:
                            continue
                        previous = found.get((target, 0))
                        if previous is not None and previous.arrival <= finish:
                            break
                        cost = self.cost(source, target, departure, flight)
                        extra = int(cost > instance.normal_dv)
                        if cost > instance.max_dv or route.exceptions + extra > instance.max_exceptions:
                            continue
                        key = target, extra
                        previous = found.get(key)
                        if previous is None or finish < previous.arrival:
                            found[key] = Route(
                                route.path + (target,),
                                route.legs + ((source, target, departure, flight, cost),),
                                finish, route.exceptions + extra,
                            )
            if found:
                return list(found.values())
        return []

    def record(self, candidate):
        """Verifikon dhe ruan kandidatin kur përmirëson rekordin."""
        if self.best is not None and route_value(candidate) <= route_value(self.best):
            return
        try:
            checked = verify(self.instance, candidate.path, candidate.legs)
        except ValueError as error:
            print(f"Kandidat i refuzuar: {error}", flush=True)
            return
        if self.best is not None and route_value(checked) <= route_value(self.best):
            return
        elapsed = time.monotonic() - self.started
        self.experiment.export(checked, elapsed, "improvement", update_best=True)
        self.best = checked
        self.report("Rekord")

    def report(self, label):
        """Shfaq numrin e vizitave, ditën, përjashtimet dhe kohën reale."""
        route = self.best
        print(
            f"{label} | {len(route.path)}/{len(self.instance.orbits)} domate"
            f" | dita {route.arrival:.4f} | përjashtime {route.exceptions}/{self.instance.max_exceptions}"
            f" | {time.monotonic() - self.started:.1f} s | prova {self.attempts}",
            flush=True,
        )

    def build(self, start, width, flight_step, exception_penalty, broad=False):
        """Zgjeron rrugët me Greedy (gjerësi 1) ose Beam Search (gjerësi >1)."""
        beam = [start]
        self.record(start)
        while beam and time.monotonic() < self.deadline:
            expanded = []
            for route in beam:
                expanded.extend(self.successors(route, flight_step, broad))
            expanded.sort(key=lambda route: route.arrival + exception_penalty * route.exceptions)
            beam = []
            endings = set()
            for route in expanded:
                self.record(route)
                key = route.path[-1], route.exceptions
                if len(beam) < width and key not in endings:
                    beam.append(route)
                    endings.add(key)
            if self.best and len(self.best.path) == len(self.instance.orbits):
                break

    def insert_visit(self):
        """Fut një domate të munguar mes dy vizitave, duke ruajtur oraret e tjera."""
        route = self.best
        instance = self.instance
        missing = set(range(len(instance.orbits))) - set(route.path)
        if not missing or not route.legs:
            return False
        for target in sorted(missing, key=lambda i: min(self.distances[i][j] for j in route.path)):
            legs = sorted(enumerate(route.legs), key=lambda pair:
                          self.distances[pair[1][0]][target] + self.distances[target][pair[1][1]])
            for index, (source, dest, departure, flight, old_cost) in legs[:14]:
                lower = (sum(route.legs[index - 1][2:4]) if index else instance.epoch)
                upper = route.legs[index + 1][2] if index + 1 < len(route.legs) else route.arrival
                allowed = instance.max_exceptions - route.exceptions + int(old_cost > instance.normal_dv)
                limit = instance.max_dv if allowed else instance.normal_dv
                if upper - lower < 2 * instance.min_flight:
                    continue
                # Shpërndaj provat brenda hapësirës kohore të dy vizitave ekzistuese.
                slots = min(64, max(8, math.ceil((upper - lower) / .125)))
                grid = [lower + (upper - lower) * j / slots for j in range(1, slots)]
                for start in sorted({lower, departure, lower + .125, lower + .5}):
                    for meet in grid:
                        if time.monotonic() >= self.deadline:
                            return False
                        first_flight = meet - start
                        if first_flight < instance.min_flight:
                            continue
                        cost1 = self.cost(source, target, start, first_flight)
                        if cost1 > limit:
                            continue
                        for finish in (upper, upper - .0625, upper - .25, upper - .5, upper - 1):
                            second_flight = finish - meet
                            if second_flight < instance.min_flight or meet + second_flight > upper:
                                continue
                            second_limit = instance.max_dv if allowed > int(cost1 > instance.normal_dv) else instance.normal_dv
                            cost2 = self.cost(target, dest, meet, second_flight)
                            if cost2 > second_limit:
                                continue
                            count = route.exceptions - int(old_cost > instance.normal_dv)
                            count += int(cost1 > instance.normal_dv) + int(cost2 > instance.normal_dv)
                            new_legs = ((source, target, start, first_flight, cost1),
                                        (target, dest, meet, second_flight, cost2))
                            # Përdor mbërritjen reale me të njëjtën mbledhje si validuesi.
                            if start + first_flight > meet:
                                continue
                            candidate = Route(
                                route.path[:index + 1] + (target,) + route.path[index + 1:],
                                route.legs[:index] + new_legs + route.legs[index + 1:],
                                route.arrival if index + 1 < len(route.legs) else meet + second_flight,
                                count,
                            )
                            self.record(candidate)
                            if self.best is not route:
                                return True
        return False

    def orbital_stages(self, reverse, split):
        """Ndan orbitat sipas lartësisë dhe pjerrësisë për të planifikuar urat."""
        orbits = self.instance.orbits
        radii = [orbit[0] for orbit in orbits]
        # Dy nivele lartësie. Pjerrësia nga normalja është gjithmonë 0–180°.
        boundary = math.sqrt(min(radii) * max(radii))
        self.inclinations = [math.acos(math.cos(orbit[2])) for orbit in orbits]
        stages = [set() for _ in range(5)]
        for tomato, radius in enumerate(radii):
            angle = self.inclinations[tomato]
            index = 2 * round(angle / (math.pi / 2)) if radius < boundary else (
                1 if angle < math.radians(split) else 3
            )
            stages[index].add(tomato)
        return list(reversed(stages)) if reverse else stages

    def segment_options(self, source, target, day, end, step, exceptions, budget, deadline,
                        earliest_only=False):
        """Kërkon mbërritjet më të hershme në rrjetë për secilën kategori ΔV."""
        instance = self.instance
        first = math.ceil(day / step)
        last = math.floor(end / step)
        min_ticks = math.ceil(instance.min_flight / step)
        max_ticks = int(10 / step)
        cost_limit = instance.max_dv if exceptions < budget else instance.normal_dv
        result = {}
        for arrival_tick in range(first + min_ticks, last + 1):
            if time.monotonic() >= deadline:
                break
            earliest = max(first, arrival_tick - max_ticks)
            # Fillojmë me fluturime të shkurtra për të kufizuar provat Lambert.
            for departure_tick in range(arrival_tick - min_ticks, earliest - 1, -1):
                departure = departure_tick * step
                flight = (arrival_tick - departure_tick) * step
                cost = self.cost(source, target, departure, flight)
                if cost <= cost_limit:
                    extra = int(cost > instance.normal_dv)
                    result.setdefault(extra, (source, target, departure, flight, cost))
                    if extra == 0 or earliest_only:
                        return tuple(result.values())
        return tuple(result.values())

    def rebuild_segment(self, route, target, left, right, step, width):
        """Riorganizon vizitat brenda dy pikave fikse, me një domate shtesë."""
        instance = self.instance
        before, after = route.legs[:left], route.legs[right:]
        retained = sum(leg[4] > instance.normal_dv for leg in before + after)
        budget = instance.max_exceptions - retained
        begin = before[-1][2] + before[-1][3] if before else instance.epoch
        end = after[0][2] if after else instance.end
        required = set(route.path[left + 1:right]) | {target}
        anchor = route.path[right]
        beam = [Route((route.path[left],), (), begin, 0)]
        deadline = min(self.deadline, time.monotonic() + 10)
        for depth in range(len(required) + 1):
            expanded = {}
            for state in beam:
                remaining = required - set(state.path)
                candidates = sorted(remaining) if remaining else [anchor]
                for node in candidates:
                    if time.monotonic() >= deadline:
                        return False
                    for leg in self.segment_options(
                        state.path[-1], node, state.arrival,
                        end - len(remaining) * step, step, state.exceptions, budget, deadline,
                    ):
                        exceptions = state.exceptions + int(leg[4] > instance.normal_dv)
                        next_state = Route(state.path + (node,), state.legs + (leg,),
                                           leg[2] + leg[3], exceptions)
                        key = frozenset(next_state.path), node, exceptions
                        previous = expanded.get(key)
                        if previous is None or next_state.arrival < previous.arrival:
                            expanded[key] = next_state
            beam = sorted(expanded.values(), key=lambda state:
                          state.arrival + 4 * state.exceptions)[:width]
            if not beam:
                return False
        for segment in beam:
            candidate = Route(
                route.path[:left] + segment.path + route.path[right + 1:],
                before + segment.legs + after,
                route.arrival if after else segment.arrival,
                retained + segment.exceptions,
            )
            best = self.best
            self.record(candidate)
            if self.best is not best:
                return True
        return False

    def focused_pass(self, route, step, width, windows, offset):
        """Zgjedh segmente pranë orbitave të domateve të pavizituara."""
        missing = sorted(set(range(len(self.instance.orbits))) - set(route.path))
        for target in missing:
            positions = sorted(range(1, len(route.path) - 1),
                               key=lambda pos: self.distances[target][route.path[pos]])
            # Përfshijmë si fqinjët orbitalë, ashtu edhe segmentet rreth tyre.
            positions = positions[:12]
            positions = positions[offset % len(positions):] + positions[:offset % len(positions)]
            tried = set()
            for center in positions:
                for size in windows:
                    for shift in (0, -1, 1):
                        if time.monotonic() >= self.deadline:
                            return
                        left = max(0, center - size // 2 + shift)
                        right = min(len(route.path) - 1, left + size)
                        if (left, right) in tried or right - left < 2:
                            continue
                        tried.add((left, right))
                        self.experiment.parameters.update({
                            "target": target, "left_index": left, "right_index": right,
                            "flight_step_days": step, "beam_width": width,
                            "segment_budget_seconds": 10,
                        })
                        if self.rebuild_segment(route, target, left, right, step, width):
                            return

    def retime_insertion(self, route, target, index, step, delay):
        """Fut një domate dhe zhvendos oraret pas saj derisa të rilidhë rrugën."""
        instance = self.instance
        legs = list(route.legs[:index])
        source, dest, departure, _, original_cost = route.legs[index]
        arrival = (legs[-1][2] + legs[-1][3] if legs else instance.epoch) + delay
        original_exception = int(original_cost > instance.normal_dv)
        # Dy fluturimet zëvendësuese ndajnë buxhetin e transferimit të vjetër.
        affected = [(source, target, original_exception), (target, dest, original_exception)]
        affected.extend(
            (leg[0], leg[1], int(leg[4] > instance.normal_dv)) for leg in route.legs[index + 1:]
        )
        first_exception = 0
        for offset, (from_node, to_node, allowance) in enumerate(affected):
            if time.monotonic() >= self.deadline:
                return
            if offset >= 2:
                old_index = index + offset - 1
                if arrival <= route.legs[old_index][2]:
                    legs.extend(route.legs[old_index:])
                    arrival = route.arrival
                    break
            if offset == 1:
                allowance -= first_exception
            options = self.segment_options(
                from_node, to_node, arrival, min(arrival + 24, instance.end),
                step, 0, allowance, self.deadline, earliest_only=True,
            )
            if not options:
                return
            leg = options[0]
            if offset == 0:
                first_exception = int(leg[4] > instance.normal_dv)
            legs.append(leg)
            arrival = leg[2] + leg[3]
        candidate = Route(
            route.path[:index + 1] + (target,) + route.path[index + 1:],
            tuple(legs), arrival, sum(leg[4] > instance.normal_dv for leg in legs),
        )
        self.record(candidate)

    def retiming_pass(self, route, step, delay):
        """Provo pozicione futjeje sipas ngjashmërisë së orbitave."""
        missing = sorted(set(range(len(self.instance.orbits))) - set(route.path))
        for target in missing:
            positions = sorted(range(len(route.legs)), key=lambda index:
                               self.distances[target][route.path[index]]
                               + self.distances[target][route.path[index + 1]])
            for index in positions[:16]:
                if time.monotonic() >= self.deadline:
                    return
                self.experiment.parameters.update({"target": target, "insertion_index": index})
                self.retime_insertion(route, target, index, step, delay)
                if self.best is not route:
                    return

    def guided_successors(self, route, targets, stage, reverse, flight_step, broad):
        """Zgjeron vetëm grupin aktual dhe rezervon përjashtime për urat e ardhshme."""
        instance = self.instance
        source = route.path[-1]
        outer = stage % 2 == 1
        bridge = source not in targets
        ordered = sorted(targets - set(route.path), key=self.distances[source].__getitem__)
        # Në orbitat e larta ecim përgjatë vargut të pjerrësive. Kjo ruan
        # domatet ndërmjetëse që ndryshe bëhen të vështira për t'u arritur.
        if outer and ordered:
            direction = -1 if reverse else 1
            edge = min(direction * self.inclinations[i] for i in ordered)
            front = [i for i in ordered if direction * self.inclinations[i] <= edge + math.radians(5)]
            ordered = front[:12] + [i for i in ordered if i not in front][:2]
        else:
            ordered = ordered[:12]
        reserve = 4 - stage
        limit = instance.max_dv if route.exceptions < instance.max_exceptions - reserve else instance.normal_dv
        if bridge:
            waits = tuple(i * .125 for i in range(25)) + (4, 5, 6, 8, 10, 12)
            flights = tuple(i * .03125 for i in range(7, 32)) + (1, 1.25, 1.5, 2, 2.5, 3)
        else:
            waits = (0, .25, .75, 1.5, 3, 5, 8, 12) if broad else (0, .25, .75, 1.5, 3)
            flights = (instance.min_flight, .0625, .125, .1875) + tuple(
                flight_step * i for i in range(1, int((14 if outer else 4) / flight_step) + 1)
            )
        found = []
        for target in ordered:
            options = {}
            for wait in waits:
                departure = route.arrival + wait
                for flight in flights:
                    if time.monotonic() >= self.deadline:
                        return found
                    finish = departure + flight
                    if finish > instance.end or flight < instance.min_flight:
                        continue
                    if 0 in options and finish >= options[0].arrival:
                        continue
                    if bridge and 1 in options and finish >= options[1].arrival:
                        continue
                    cost = self.cost(source, target, departure, flight)
                    if cost > limit:
                        continue
                    extra = int(cost > instance.normal_dv)
                    if extra not in options or finish < options[extra].arrival:
                        options[extra] = Route(
                            route.path + (target,),
                            route.legs + ((source, target, departure, flight, cost),),
                            finish, route.exceptions + extra,
                        )
            found.extend(options.values())
        return found

    def build_guided(self, reverse, split, width, flight_step, broad):
        """Provon Beam Search përgjatë pesë grupeve, nga njëri skaj ose tjetri."""
        stages = self.orbital_stages(reverse, split)
        first = sorted(stages[0])
        self.rng.shuffle(first)
        beam = [Route((i,), (), self.instance.epoch, 0) for i in first[:width]]
        for stage, targets in enumerate(stages):
            while time.monotonic() < self.deadline:
                complete = [route for route in beam if targets.issubset(route.path)]
                if complete:
                    beam = complete
                    break
                expanded = []
                for route in beam:
                    expanded.extend(self.guided_successors(route, targets, stage, reverse, flight_step, broad))
                if not expanded:
                    break
                def priority(route):
                    # Lëvizja përpara dhe përjashtimet pa nevojë mund të lënë të
                    # izoluara domate. Një penalitet i mban këto degë më poshtë.
                    angle = self.inclinations[route.path[-1]]
                    direction = -1 if reverse else 1
                    angular_cost = 180 * direction * angle if stage % 2 else 0
                    return route.arrival + 70 * route.exceptions + angular_cost

                expanded.sort(key=priority)
                beam = []
                endings = set()
                for route in expanded:
                    self.record(route)
                    key = route.path[-1], route.exceptions
                    if len(beam) < width and key not in endings:
                        beam.append(route)
                        endings.add(key)
            self.experiment.parameters.setdefault("stage_visits", []).append(
                max((len(targets.intersection(route.path)) for route in beam), default=0)
            )
            if beam:
                deepest = max(beam, key=route_value)
                print(f"Grupi {stage + 1}/{len(stages)} | {len(targets.intersection(deepest.path))}"
                      f"/{len(targets)} në grup | {len(deepest.path)} gjithsej"
                      f" | dita {deepest.arrival:.3f} | {deepest.exceptions} përjashtime"
                      f" | {time.monotonic() - self.started:.1f} s", flush=True)
            if time.monotonic() >= self.deadline:
                break
        if beam:
            # Nëse ndonjë grup mbeti i paplotë, plotëso rrugën pa rendin e grupeve.
            start = max(beam, key=route_value)
            self.build(start, min(width, 3), flight_step, 36, broad)

    def run(self):
        """Nis nga dataset-i dhe përmirëson rrugën duke rindërtuar fundin."""
        if OUTPUT.exists():
            saved = json.loads(OUTPUT.read_text(encoding="utf-8"))
            self.best = verify(self.instance, saved["path"], saved["transfers"])
            if self.resume_experiment:
                self.experiment.parameters = {
                    "method": "focused_retiming", "prior_attempts": self.attempts,
                }
                self.experiment.export(self.best, time.monotonic() - self.started, "phase")
            else:
                self.experiment.export(self.best, time.monotonic() - self.started, "baseline")
            self.report("Rekord i ngarkuar")
        starts = sorted(range(len(self.neighbors)), key=lambda i: sum(
            abs(math.log(self.instance.orbits[i][0] / self.instance.orbits[j][0]))
            for j in self.neighbors[i][:8]
        ))
        last_insertion_path = None
        while time.monotonic() < self.deadline:
            self.attempts += 1
            attempt_start = time.monotonic()
            attempt_best = self.best
            if self.focused:
                if self.best is None:
                    raise ValueError("--focused kërkon një rekord të verifikuar për nisje.")
                pass_index = self.attempts - 1
                if self.resume_experiment:
                    step = (.125, .0625, .25)[pass_index % 3]
                    delay = (0, .5, 1, 2, 4)[pass_index % 5]
                    self.experiment.parameters = {
                        "method": "focused_retiming", "attempt": self.attempts,
                        "flight_step_days": step, "initial_wait_days": delay,
                        "starting_visits": len(self.best.path),
                    }
                    self.retiming_pass(self.best, step, delay)
                    self.finish_attempt(attempt_start, attempt_best)
                    if len(self.best.path) == len(self.instance.orbits):
                        break
                    continue
                step, width, windows = (
                    (.125, 5, (2, 3, 4)),
                    (.0625, 6, (2, 3, 4)),
                    (.125, 9, (5, 6, 8)),
                    (.0625, 10, (4, 5, 6)),
                    (.25, 12, (6, 8, 10)),
                )[pass_index % 5]
                self.experiment.parameters = {
                    "method": "focused_segment", "attempt": self.attempts,
                    "window_sizes": windows, "starting_visits": len(self.best.path),
                }
                self.focused_pass(self.best, step, width, windows, pass_index)
                self.finish_attempt(attempt_start, attempt_best)
                if len(self.best.path) == len(self.instance.orbits):
                    break
                continue
            if (self.optimize or self.coverage) and self.best and self.best.legs and self.best.legs != last_insertion_path:
                self.experiment.parameters = {"method": "insertion_local", "attempt": self.attempts}
                last_insertion_path = self.best.legs
                self.insert_visit()
                self.finish_attempt(attempt_start, attempt_best)
                if len(self.best.path) == len(self.instance.orbits):
                    break
                continue
            if self.coverage and self.attempts % 5 != 0:
                reverse = self.attempts % 2 == 0
                split = (90, 90.5, 91, 89.5)[(self.attempts - 2) % 4]
                width = 3 if self.attempts < 5 else 5
                broad = self.attempts % 3 == 0
                flight_step = .125 if broad else .25
                self.experiment.parameters = {
                    "method": "orbital_group_beam", "attempt": self.attempts,
                    "reverse": reverse, "polar_split_degrees": split, "beam_width": width,
                    "flight_step_days": flight_step, "broad_window": broad,
                }
                self.build_guided(reverse, split, width, flight_step, broad)
                self.finish_attempt(attempt_start, attempt_best)
                if len(self.best.path) == len(self.instance.orbits):
                    break
                continue
            if not self.best or self.attempts % 4 == 0:
                tomato = starts[0] if not self.best else self.rng.choice(starts)
                start = Route((tomato,), (), self.instance.epoch, 0)
                method = "greedy" if self.attempts == 1 else "beam_restart"
            else:
                # Mbaj fillimin e vlefshëm dhe hiq disa vizita nga fundi.
                broad = self.optimize and self.attempts % 3 == 0
                removed = self.rng.randint(18, 65) if broad else self.rng.randint(4, 18)
                keep = max(1, len(self.best.path) - removed)
                legs = self.best.legs[:keep - 1]
                arrival = legs[-1][2] + legs[-1][3] if legs else self.instance.epoch
                exceptions = sum(leg[4] > self.instance.normal_dv for leg in legs)
                start = Route(self.best.path[:keep], legs, arrival, exceptions)
                method = "suffix_local"
            broad = self.optimize and self.attempts % 3 == 0
            width = 1 if self.attempts == 1 else (5 if broad else 3)
            flight_step = .25 if self.attempts % 3 else .125
            penalty = 18 if self.attempts == 1 else self.rng.choice((8, 14, 24, 36))
            self.experiment.parameters = {
                "method": method, "attempt": self.attempts, "starting_visits": len(start.path),
                "starting_tomato": start.path[0], "prefix_last_tomato": start.path[-1],
                "beam_width": width, "flight_step_days": flight_step,
                "exception_penalty_days": penalty, "broad_window": broad,
            }
            self.build(start, width, flight_step, penalty, broad)
            self.finish_attempt(attempt_start, attempt_best)
            if len(self.best.path) == len(self.instance.orbits):
                break
        checked = verify(self.instance, self.best.path, self.best.legs)
        self.best = checked
        self.experiment.parameters = {"method": "final", "attempt": self.attempts}
        archive = self.experiment.export(checked, time.monotonic() - self.started, "final")
        self.report("Përfundim i verifikuar")
        print(f"Arkivi i provës: {archive.name}", flush=True)

    def finish_attempt(self, started, previous):
        """Regjistron edhe provat që përfunduan pa ndryshuar rekordin."""
        self.experiment.attempts.append({
            **self.experiment.parameters,
            "improved": self.best is not previous,
            "record_visits": len(self.best.path),
            "record_mission_days": self.best.arrival - self.instance.epoch,
            "record_exceptions": self.best.exceptions,
            "attempt_seconds": round(time.monotonic() - started, 3),
            "computation_seconds": round(time.monotonic() - self.started, 3),
        })
        self.report(f'Fundi i provës ({self.experiment.parameters["method"]})')


def main():
    """Lexon parametrat dhe zgjedh mes kërkimit dhe verifikimit të JSON-it."""
    global OUTPUT, HISTORY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=float, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--coverage", action="store_true")
    parser.add_argument("--focused", action="store_true")
    parser.add_argument("--scratch", action="store_true",
                        help="Nis nga dataset-i dhe ruan rezultatet në një dosje të re lokale.")
    parser.add_argument("--resume-experiment")
    parser.add_argument("--initial-attempts", type=int, default=0)
    args = parser.parse_args()
    if not math.isfinite(args.minutes) or args.minutes <= 0:
        parser.error("--minutes duhet të jetë numër pozitiv dhe i fundëm.")
    if args.scratch:
        if args.focused or args.verify_only or args.resume_experiment:
            parser.error("--scratch nis një kërkim të ri dhe nuk kombinohet me "
                         "--focused, --verify-only ose --resume-experiment.")
        scratch = ROOT / ".kttsp-scratch" / uuid4().hex
        scratch.mkdir(parents=True)
        OUTPUT = scratch / OUTPUT.name
        HISTORY = scratch / HISTORY.name
        print(f"Rezultatet e kërkimit nga zero: {scratch}", flush=True)
    instance = Instance(*read_instance(DATA))
    print(
        f"Medium | {len(instance.orbits)} domate | {instance.duration:g} ditë"
        f" | minimumi {instance.min_flight:g} ditë | ΔV {instance.normal_dv:g}/{instance.max_dv:g} m/s"
        f" | deri në {instance.max_exceptions} përjashtime",
        flush=True,
    )
    if args.verify_only:
        saved = json.loads(OUTPUT.read_text(encoding="utf-8"))
        checked = verify(instance, saved["path"], saved["transfers"])
        if (saved["arrival_day"], saved["exceptions"]) != (checked.arrival, checked.exceptions):
            raise ValueError("Përmbledhja e JSON-it nuk përputhet me transferimet.")
        print(
            f"U verifikuan {len(checked.legs)} transferime | {len(checked.path)} domate"
            f" | dita {checked.arrival:.4f} | {checked.exceptions} përjashtime"
        )
    else:
        Search(
            instance, args.minutes * 60, args.seed, args.optimize, args.coverage, args.focused,
            args.resume_experiment, args.initial_attempts,
        ).run()


if __name__ == "__main__":
    main()

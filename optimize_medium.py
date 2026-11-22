"""ESA Medium timing beam search. Keep original solution immutable.
Run from the project root in the kttsp-medium conda environment.
"""
import argparse
import json
import math
import random
import runpy
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_JSON = 'results/esa/medium_20261120T080509Z_bc1f8d8f_4_verified.json'
DEFAULT_EVAL = 'esa_evaluation/spoc-4-keplerian-tomato-traveling-salesperson-medium-1782203984713.py'

@dataclass
class State:
    arrival: float
    exceptions: int
    departures: tuple
    flights: tuple


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--solution', default=DEFAULT_JSON)
    ap.add_argument('--evaluator', default=DEFAULT_EVAL)
    ap.add_argument('--beam', type=int, default=7)
    ap.add_argument('--max-evals', type=int, default=30000)
    ap.add_argument('--seconds', type=int, default=900)
    ap.add_argument('--seed', type=int, default=2026)
    args = ap.parse_args()
    random.seed(args.seed)
    src = json.loads(Path(args.solution).read_text())
    udp = runpy.run_path(args.evaluator)['udp']
    path = [int(k) for k in src['path']]
    legs = src['transfers']
    n = len(legs)
    assert len(path) == udp.n_tomatoes == n + 1
    assert all(int(t[0]) == path[i] and int(t[1]) == path[i+1] for i,t in enumerate(legs))
    base_dep = [float(t[2]) for t in legs]
    base_tof = [float(t[3]) for t in legs]
    baseline_fitness = udp.fitness(base_dep + base_tof + path)
    assert baseline_fitness[1:] == [0, 0, 0, 0], baseline_fitness
    best = float(baseline_fitness[0])
    print(f'ESA baseline: {best:.9f} days, {n+1} tomatoes', flush=True)
    print('NOTE: this first search optimizes timing on the same visit order.', flush=True)

    outdir = Path('results/esa/medium_beam_experiments')
    outdir.mkdir(parents=True, exist_ok=True)
    cache = {}
    calls = 0
    started = time.monotonic()
    stopped = False
    max_exceptions = udp.n_dv_exception
    normal = udp.dv_threshold
    exceptional = udp.dv_threshold_exception

    def dv(i, departure, flight):
        nonlocal calls, stopped
        key = (i, round(departure, 8), round(flight, 8))
        if key not in cache:
            if calls >= args.max_evals or time.monotonic() - started >= args.seconds:
                stopped = True
                return math.inf
            calls += 1
            cache[key] = udp.compute_transfer(path[i], path[i+1], departure, flight)
        return cache[key]

    def step_options(i, state):
        """Candidate departure/flight combinations, including original incumbent branch."""
        earliest = max(0., state.arrival)
        original_dep = base_dep[i]
        original_flight = base_tof[i]
        depart_choices = {round(original_dep, 8)} if original_dep + 1e-8 >= earliest else set()
        for delay in [0, .0625, .125, .25, .5, 1., 2.]:
            depart_choices.add(round(earliest + delay, 8))
        for shift in [-1., -.5, -.25, -.125, .125, .25, .5, 1.]:
            if original_dep + shift + 1e-8 >= earliest:
                depart_choices.add(round(original_dep + shift, 8))
        flights = {round(original_flight, 8)}
        for ratio in [.5, .75, .875, 1.125]:
            flights.add(round(max(udp.min_tof, original_flight * ratio), 8))
        for delta in [-.25, -.125, -.0625, .0625, .125]:
            flights.add(round(max(udp.min_tof, original_flight + delta), 8))
        pairs = [(dep, tof) for dep in depart_choices for tof in flights
                 if dep >= earliest - 1e-8 and dep <= udp.max_time
                 and tof >= udp.min_tof and dep + tof <= udp.max_time]
        # Bound costs so a complete 180-leg search can fit the evaluation budget.
        pairs.sort(key=lambda p: (abs(p[0] - original_dep)
                                  + abs(p[1] - original_flight), p))
        close = pairs[:8]
        others = pairs[8:]
        random.shuffle(others)
        yield from (close + others[:8])

    beam = [State(0., 0, (), ())]
    incumbent_exceptions = 0
    for i in range(n):
        next_states = []
        seen = set()
        # Keep the exact validated prefix available at every step.
        if i == 0:
            inc_dep = ()
            inc_tof = ()
        else:
            inc_dep = tuple(base_dep[:i])
            inc_tof = tuple(base_tof[:i])
        # Exception count of incumbent prefix as calculated from saved, ESA-evaluated DV.
        if i > 0:
            previous_dv = legs[i-1][4]
            incumbent_exceptions += int(normal + 1e-6 < previous_dv <= exceptional + 1e-6)
        incumbent = State(base_dep[i-1] + base_tof[i-1] if i else 0.,
                          incumbent_exceptions, inc_dep, inc_tof)
        for state in beam + [incumbent]:
            for dep, tof in step_options(i, state):
                arrival = dep + tof
                if arrival >= best + 1e-9:
                    continue
                v = dv(i, dep, tof)
                if stopped:
                    break
                if not math.isfinite(v) or v > exceptional + 1e-6:
                    continue
                new_exceptions = state.exceptions + int(v > normal + 1e-6)
                if new_exceptions > max_exceptions:
                    continue
                key = (round(arrival, 6), new_exceptions, round(dep, 6))
                if key in seen:
                    continue
                seen.add(key)
                next_states.append(State(arrival, new_exceptions,
                                         state.departures + (dep,),
                                         state.flights + (tof,)))
            if stopped:
                break
        if not next_states:
            print(f'No new candidates at leg {i+1}; retaining incumbent.', flush=True)
            break
        # Retain a diverse arrival-time/exception selection instead of only fastest states.
        next_states.sort(key=lambda s: (s.arrival + .12 * s.exceptions, s.arrival))
        chosen = []
        buckets = set()
        for s in next_states:
            bucket = (s.exceptions, int(s.arrival * 4))
            if bucket not in buckets:
                chosen.append(s)
                buckets.add(bucket)
            if len(chosen) >= args.beam:
                break
        beam = chosen or next_states[:args.beam]
        if i % 10 == 9 or i == n - 1:
            print(f'Leg {i+1}/{n}: earliest {min(s.arrival for s in beam):.4f}, '
                  f'evaluations {calls}, elapsed {time.monotonic()-started:.1f}s', flush=True)
        if stopped:
            print('Search budget reached; no incomplete schedules will be saved.', flush=True)
            break
        if i == n - 1:
            for state in sorted(beam, key=lambda s: s.arrival):
                fit = udp.fitness(list(state.departures) + list(state.flights) + path)
                if fit[1:] == [0, 0, 0, 0] and fit[0] < best - 1e-7:
                    best = fit[0]
                    new_transfers = [
                        [path[j], path[j+1], state.departures[j], state.flights[j],
                         dv(j, state.departures[j], state.flights[j])]
                        for j in range(n)
                    ]
                    obj = {'path': path, 'transfers': new_transfers,
                           'arrival_day': best, 'exceptions': state.exceptions,
                           'validation': {'fitness': fit, 'valid': True},
                           'source': args.solution, 'method': 'timing_beam_search'}
                    output = outdir / f'medium_beam_valid_{best:.9f}.json'
                    output.write_text(json.dumps(obj, indent=2))
                    print('NEW ESA-VALID BEST:', best, 'saved to', output, flush=True)
    print(f'Finished. Best ESA-validated result: {best:.9f} days. '
          f'Delta: {baseline_fitness[0] - best:.9f} days. Lambert calls: {calls}.', flush=True)

if __name__ == '__main__':
    main()

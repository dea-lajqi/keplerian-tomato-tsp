# Medium day-180 deadline experiment

See [CROSSING_BOUND.md](CROSSING_BOUND.md) for the model assumptions, proof and
reference inputs. The four-crossing bound is independent of departure epoch
and says nothing about mission duration.

## Model

The terminal calculation retains the latest ready time for each visited
subset and first tomato in the reversed graph. Waiting makes this dominance
valid for a fixed visit set, endpoint and resource count on the measured
graph. A forward label keeps the earliest ready time. The interior join
requires exactly complementary coverage, matching directed transfer windows,
and a combined count within the original five-exception cap. Every candidate
record is re-evaluated by a fresh instance of the original ESA evaluator.
A 0.0001-day gain is required to write a new record.

The block frontier moves a displacement band across overlapping sets of
tomatoes. It is exact on the chosen terminal event graph. The bounded
upstream beam, finite epochs and fixed prefix preceding the cut exclude
other possibilities. The reported timing gaps and latest departures are
properties of the measured graph and search budgets.

A low-to-low topology needs four exceptional crossings and has one
remaining exception. When one endpoint is in the high shell, all five
exceptions are reserved for crossings. The contracted path-cover experiment
uses this fact to admit only geometrically normal high-to-high edges in
those alternatives. Its flight proxy only orders candidates.

## Reproduce

Run from the repository root using the Medium environment and a new output
directory for each run. The benchmark inputs and results are under
`results/esa/medium_deadline180`. The terminal event file contains 576
completed directed-edge/mesh combinations from a bounded run.

    export MPLCONFIGDIR=/tmp OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
    D=results/esa/medium_frontier
    M=results/esa/medium_deadline180
    R=$D/record_enriched/records/182.254267328103_f5531dbe.json
    nice -n 11 python -B -m medium_frontier.deadline --bridge \
      --seed-record "$R" --atlas "$M/terminal/events.json" \
      --events "$D/exact_late/events.json" "$D/exact_middle/events.json" \
      "$M/joint/learned_events.json" --days 182.2543 180 181.25 \
      --directory results/esa/medium_deadline180_reproduction \
      --trial-seconds 60 --seconds 355 --sample-labels 14 \
      --trials 148 3 240 .035 --trials 131 5 950 .032

For the adaptive variant use `--refine --flight-weight .12`. For the
single-deficit continuous repair experiments use `--repair-schedule` to
save near-joins separately, then `python -B -m medium_frontier.repair` with
those files. The `feasible` directories contain complete ESA-valid schedules,
including controls. Files under `near` contain chronology violations and
are inputs to repair only. Searches using real wall-clock limits can produce
different coverage under different CPU loads.

Use `python -B -m medium_frontier.topology --normalize-passage` to try
five-crossing alternatives from normalized incumbent spines. The independent
geometry-only formulation is `python -B -m medium_frontier.topology_flow`.

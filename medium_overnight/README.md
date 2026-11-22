# ESA SpOC-4 Medium — standalone overnight search (181 tomatoes)

**This folder is a separate experiment. It never modifies Small/Easy, Large, the original ESA evaluator, the original Medium instance, or the baseline JSON. No Git commands are run.**

## Goal and actual evidence

- Independently ESA-validated initial Medium score (from the user's WSL session): **202.8125 days**.
- Aspirational target: **146 days**. No claim that it is reachable or that this solver will achieve it.
- Current version: **experimental bounded hybrid search**. A geometric radius-shell heuristic reserves exceptions; localized route perturbations are followed by feasibility-aware Lambert beam schedules and downstream retiming. Multiple workers preserve a small archive of complete feasible route orders. It is **not** a global exact solver and offers **no optimality guarantee**.
- New records are saved only after a second independent call to the **original ESA Medium** `fitness()` by the supervisor, plus strict manual bound/chronology checks.

## Install (one time)

1. **Extract the provided ZIP.** Copy its entire folder `medium_overnight` into the top level of your existing `keplerian-tomato-tsp` project. **Do not replace the project folders or existing `.py` files.**
2. In VS Code, open the project's **WSL terminal** (the one where `(kttsp-medium)` appears). Run the commands **from the repository root**.
3. Your existing env has Python 3.11 and PyKEP 2.6.4. The new code needs only Python standard-library modules beyond the existing original evaluator's dependencies; **do not install anything in Easy/Codex's environment**.

Folder placement:

```text
keplerian-tomato-tsp/
  medium_overnight/
    campaign.py
    search.py
    validation.py
    test_offline.py
    inspect_baseline.py
    export_sol.py
    start.sh
    stop.sh
    status.sh
    verify.sh
  data/spoc4/kttsp/problems/medium.kttsp          (existing, untouched)
  esa_evaluation/spoc-4-...-medium-...py           (existing, untouched)
  results/esa/medium_..._4_verified.json          (existing, untouched)
  results/esa/small_overnight/                    (Easy, untouched)
```

## Test FIRST, before the overnight launch

**1. Run Python-only unit tests (no PyKEP needed):**

```bash
python medium_overnight/test_offline.py
```

**2. Inspect the saved Medium data (without recalculating orbital physics):**

```bash
python medium_overnight/inspect_baseline.py
```

**3. Independently verify the original ESA Medium baseline with PyKEP:**

```bash
"$HOME/miniforge3/envs/kttsp-medium/bin/python" -B medium_overnight/campaign.py verify-best
```

This should print `[202.8125, 0.0, 0.0, 0.0, 0.0]` for the official ESA fitness. If it fails, **do not start the long run**. In particular, check that `conda list pykep` returns version 2.x.

**4. Smoke test the actual supervisor + worker for 36 seconds in a separate directory:**

```bash
"$HOME/miniforge3/envs/kttsp-medium/bin/python" -u -B medium_overnight/campaign.py run \
  --directory results/esa/medium_overnight/smoke --hours 0.01 --workers 1 --seed 73
```

The smoke test should report a baseline, start worker 0, create checkpoint files, stop gracefully, and print a `FINAL` score. If it finishes at `202.8125`, that is **not** an error; improvements are not guaranteed. The smoke test uses one low-priority worker and cannot modify your Easy files.

## Start the eight-hour campaign

Run:

```bash
bash medium_overnight/start.sh
```

The script starts **2 low-priority workers** as independent detached Linux processes, with one worker per logical core if Linux CPU affinity is supported. The machine previously reported **16 logical CPUs** and **6.9 GiB available RAM**, while Easy had 3 CPU-active workers. This is a conservative initial setting, **not** a hard RAM limit.

**Do not let Windows enter Sleep.** Stay connected to power, leave WSL available, and keep the project directory mounted. The editor/Codex panel can be closed; the local Python process is separate. If Windows/WSL shuts down, use `start.sh` again to resume from checkpoints.

### Follow logs, see result, monitor CPU/memory

```bash
tail -f results/esa/medium_overnight/run_20261122/campaign.log
```

Press `Ctrl+C` to exit `tail -f`; **this stops the log display only**, not the detached optimizer.

```bash
bash medium_overnight/status.sh
free -h
ps -eo pid,stat,%cpu,%mem,args | grep -E 'overnight_small|medium_overnight/campaign.py' | grep -v grep
```

### Stop without discarding work

```bash
bash medium_overnight/stop.sh
```

A supervisor receives SIGTERM and lets workers save checkpoints. The script uses both a stop-request file and a graceful signal.

### Resume (remaining budget, not eight hours from zero)

```bash
bash medium_overnight/start.sh
```

The supervisor reads the existing `campaign.json` and resumes with the **remaining** cumulative eight-hour budget. Worker RNG states, alternative-route pools and counters are restored from checkpoint files. In-memory Lambert caches are intentionally not preserved (a reproducibility/performance trade-off). The supervisor heartbeat is written to `campaign.json` approximately every 20 seconds.

### Revalidate/export your best Medium solution

```bash
bash medium_overnight/verify.sh
"$HOME/miniforge3/envs/kttsp-medium/bin/python" -B medium_overnight/export_sol.py
```

The `.sol` file will be written to:

`results/esa/medium_overnight/run_20261122/best_medium_esa_validated.sol`

It contains **541 numbers**: 180 departure times, 180 flight durations, then 181 tomato indices, exactly in the original evaluator's expected order.

## Saved files

```text
results/esa/medium_overnight/run_20261122/
  campaign.log            supervisor events / new records
  campaign.json           runtime/heartbeat/budget and worker process IDs
  pid                     currently launched supervisor process ID
  best.json               best ESA-revalidated candidate or saved incumbent
  worker_0.log            methods attempted, candidate counts and errors
  worker_1.log
  worker_0.json           independent worker checkpoint / candidate pool / RNG
  worker_1.json
  queue/                  temporary proposed improvements; supervisor verifies them
  records/                permanently saved strict improvements ONLY
  best_medium_esa_validated.sol  optional export
```

## How the search works (and limitations)

- An initial **local Medium baseline** is independently checked with original ESA `fitness` before searching.
- Each worker uses several complementary transformations: within-shell swaps/insertions/reversals, broad diversification, shell-crossing neighborhood changes, and suffix-retiming passes.
- For each modified route block, the worker tests **multiple departure and transfer-duration phases**, calls the original `compute_transfer` (PyKEP Lambert) with an LRU cache, and keeps a diverse beam of feasible timings. It reserves the five exception budget for remaining inner/outer-shell crossings.
- Feasible modified blocks are extended by time-aware suffix rescheduling. **Only complete ESA-valid tours** may enter the worker's candidate pool, and only strictly better independently revalidated tours become permanent records.
- Workers automatically redistribute effort based on how often each operator produces full feasible tours, with continued exploration of underused operators.
- The process runs until the eight-hour *cumulative active* budget expires or it is safely stopped. If it crashes, start it again; checkpoints may lose at most the time since the last save.
- This is a local multi-phase graph search seeded by the incumbent, **not** a full time-expanded-network MILP across all 181 nodes. Large graph models from the literature can be significantly more expensive. This is intended to give a feasible first overnight campaign under a 2-core budget.

If the search reports **0 new completed tours** over an extended period, do not run the same configuration for additional nights without a change in algorithm or sampling. Save the logs and reconsider the reachability library/search structure.

## Tests performed by the assistant before delivery

- Python syntax compilation: passed.
- Unit tests with a deterministic fake Lambert-like evaluator: **10 tests passed**, including exact chronology, permutation, bounds, route-block retiming, checkpoint state, supervisor zero-budget restart metadata and independent supervisor record rejection/acceptance. In addition, 18 trial mutations were exercised on a synthetic 181-node cost model; those are **not** orbital or ESA results.
- Original uploaded Medium dataset and stored result inspected (181 nodes, 5 exceptions and 202.8125-day saved score).
- **Not tested here:** real PyKEP 2.6.4 and 8-hour end-to-end runtime, because PyKEP is unavailable in the assistant's execution environment. The 36-second WSL smoke test above is necessary before starting the overnight run. The baseline is ESA-validated from the user's earlier WSL output, not independently re-run in this container.

## Relevant research (inspiration, not copied code)

- Bannach, Acciarini, Izzo, *The Keplerian Traveling Salesperson Problem* (2026): time-expanded networks, preprocessing, constructive and improvement heuristics. https://arxiv.org/abs/2605.00010
- Weise et al., *Solving the SpOC 4 Challenge 2 with a Multi-Step Permutation Encoding for the Keplerian Traveling Salesperson* (GECCO 2026 Companion): feasibility-oriented permutation encoding. https://scholarprofiles.sunwayuniversity.edu.my/en/publications/solving-the-spoc-4-challenge-2-with-a-multi-step-permutation-enco/

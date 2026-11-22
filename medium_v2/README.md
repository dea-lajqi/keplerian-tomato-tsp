# Medium V2 — phase-diverse ESA-validated retiming

This is an **experimental** Medium-only search, intended to run **in parallel with Codex working on Easy**. It is an independent program, separate from `medium_overnight` (V1). The source for the new run is your previously validated **200.8915914-day result** in `results/esa/medium_overnight/run_20261122/best.json` (not the older 202.8125-day baseline). The latest source is checked against the **original ESA Medium evaluator** before the run. No Git commits, pushes, or edits to ESA/Easy/V1 code are made.

## Why V2 is different

V1's `greedy_suffix` picks the earliest viable arrival at each step. However, a slightly later arrival at an intermediate tomato may unlock a much faster Lambert transfer later. V2's `phase_search.retime_suffix` uses a diverse multi-state beam over future orbital phases, retains slower intermediate arrivals, and tracks the exact ΔV exception budget. V2 searches entire suffixes of varying lengths, using fine/balanced/large phase moves, instead of only greedy suffixes. Intermediate states are heuristics and **not official solutions**. All improved complete candidates go through ESA `fitness()` and get archived as full records.

This is a **first 30-minute controlled experiment**, not a proven stronger optimizer or a guaranteed improvement. It may use many Lambert calculations and find no better mission. Its purpose is to test a fundamentally different timing mechanism before any more overnight search.

## Files and prerequisites

Copy the folder `medium_v2` into your existing repository root, alongside the already installed `medium_overnight`. Python env must be `kttsp-medium` with PyKEP 2.6.4. Requires *no new dependencies* beyond those already installed for V1. Do not copy or overwrite any existing folder.

```text
keplerian-tomato-tsp/
  medium_overnight/   # existing V1 untouched
  medium_v2/          # NEW folder only
  results/esa/medium_overnight/run_20261122/best.json # EXISTING saved verified record
```

## 1. Offline tests

```bash
python medium_v2/test_offline.py
```

## 2. Revalidate 200.8915914 with ESA before launching

```bash
python medium_v2/run.py verify-best
```

Expected: `Original ESA Medium fitness: [200.8915914, 0, 0, 0, 0]` (floating point differences tolerated). Stop if this does not pass.

## 3. Launch a 30-minute V2 experiment on ONE low-priority CPU worker

```bash
bash medium_v2/start.sh
```

This launches a detached process and returns the PID. It does not touch the existing Medium 8-hour campaign and it uses **one worker**, leaving capacity for Codex on Easy. If `start.sh` is called again while its saved PID still exists, it refuses to start a duplicate. The process has a checkpoint and will stop after 30 minutes of accumulated runtime.

## 4. Monitor and validate

```bash
bash medium_v2/status.sh

tail -n 30 results/esa/medium_v2/run_01/campaign.log

python medium_v2/run.py verify-best --directory results/esa/medium_v2/run_01
```

Only new `NEW ESA-VALID MEDIUM V2` log entries count as official improvements.

## 5. Stop / resume

```bash
bash medium_v2/stop.sh
```

After the process stops safely and saves its checkpoint, you can resume with:

```bash
bash medium_v2/start.sh
```

To extend an exhausted 30-minute campaign, run `python medium_v2/run.py run --hours 1` manually (adds up to 1 total hour, including prior active time), or edit the `--hours 0.5` in `start.sh`. New experiments should ideally get a new `--directory` and a new seed.

## Caveats

- The assistant's execution environment does not have PyKEP. The code is syntax- and offline-tested but **the real ESA evaluator must be tested in your WSL** before relying on it.
- V2 currently keeps tomato visit order fixed. It searches **different timing landscapes** and deliberately permits later intermediate arrivals. It complements V1's route mutations; it cannot on its own make the huge 54.89-day jump to the ambitious 146-day target.
- No result is claimed until the original ESA evaluator reports the five-component vector with zero violations and a shorter completion time.

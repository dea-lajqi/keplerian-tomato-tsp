# MEDIUM V3 — New search, existing ESA evaluator

**No ESA-valid improvement has been found in this environment.** We don't have PyKEP 2.x here; on your WSL installation the original ESA evaluator is the final judge. The previously validated local best is **200.8915914 days**, which V3 will discover automatically from `results/esa/medium_overnight/run_20261122/best.json`.

This V3 is an experimental follow-up, not a promise of 146 days. It is different from Medium V2: **it changes the tomato order and the transfer timing together** using inclination-guided permutations; a bounded ESA-transfer beam repairs affected transfer legs; valid seed suffixes are reused where possible; an optional SciPy SLSQP stage tries continuous refinement of the final legs.

## Folder layout

Extract/copy `medium_v3/` beside `medium_overnight/` and `medium_v2/` in your local `keplerian-tomato-tsp` repository. The program reads but NEVER writes the original `medium.kttsp`, evaluator, original `results/esa/medium_20261120T080509Z_bc1f8d8f_4_verified.json`, or existing Medium V1/V2 results. It never touches Easy or Large. New files are stored only in `results/esa/medium_v3/run_01/`.

## WSL instructions (from the project root)

Use an existing WSL terminal where `(kttsp-medium)` appears:

```bash
python medium_v3/test_offline.py
python medium_v3/run.py verify-best
python medium_v3/analysis.py
python medium_v3/offline_experiment.py --tries 4000
```

The `verify-best` command must output original ESA fitness with `200.8915914` (or lower if you've subsequently improved the incumbent).

First do a **3-minute smoke test**, with a different output folder:

```bash
python -u -B medium_v3/run.py run --directory results/esa/medium_v3/smoke --minutes 3 --trials 20
python medium_v3/run.py status --directory results/esa/medium_v3/smoke
python medium_v3/run.py verify-best --directory results/esa/medium_v3/smoke
```

If there are no errors, use the **45-minute benchmark campaign**:

```bash
bash medium_v3/start.sh
```

Monitor any time:

```bash
bash medium_v3/status.sh
tail -n 30 results/esa/medium_v3/run_01/campaign.log
```

Stop gracefully (checkpoint and best retained):

```bash
bash medium_v3/stop.sh
```

Resume a stopped campaign from its checkpoint:

```bash
bash medium_v3/start.sh
```

It accumulates prior elapsed time; if the configured 45 minutes were consumed, start a **new campaign directory**, or launch manually with a higher total cumulative duration:

```bash
python -u -B medium_v3/run.py run --minutes 90
```

Run with a **single** low-priority worker, and leave Easy's processes alone. Check `ps` as needed. The short smoke and the 45-minute run should not run simultaneously.

## Caveats and measurement

- An orbital-plane geometry score is used solely to propose candidate reorders. It does NOT replace ESA ΔV physics.
- New records must satisfy original ESA `fitness()` with 181 unique tomatoes, correct 541-variable chromosome, chronology, all 180 transfers under 600 m/s, and at most 5 exceptions over 100 m/s. The validation is in the imported `medium_overnight.validation` module.
- Continuous polishing uses SciPy only if available. If it's absent, that arm is skipped. **Do not install anything in the Easy/Codex environment.**
- It checkpoints every 60 seconds. `status` reads saved progress. The last checkpoint may lag actual work.
- No output is committed or pushed to GitHub.
- **There is no proven lower bound of 146 days.** The experiment may fail to improve 200.8915914.

See `RESEARCH.md` for the research, diagnostics, previous campaign failure analysis and experimental rationale.

To export the best separately, only after final ESA revalidation:

```bash
python medium_v3/export_best.py
```

The file is created only once at `results/esa/medium_v3/run_01/exported_esa_verified_medium.sol`. It never overwrites it.

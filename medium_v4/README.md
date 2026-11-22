# Medium V4 — joint route-and-timing anchored repair

This version is **MEDIUM ONLY**. It requires the existing project repository and `kttsp-medium` WSL Conda environment with PyKEP 2.6.4. It uses the **unaltered ESA Medium evaluator** and V1 `medium_overnight/validation.py` and `medium_overnight/search.py` for PyKEP calculations and strict ESA checks.

### Install

Unzip the delivered ZIP. Place the `medium_v4` folder **directly** inside `keplerian-tomato-tsp`, beside `medium_overnight`, `medium_v2`, and `medium_v3`. Do not replace other folders. Open your WSL terminal in the repository root and activate `kttsp-medium` as usual.

### Before any long run: syntax, offline tests, original ESA verification

```bash
python -m compileall -q medium_v4
python medium_v4/test_offline.py
python medium_v4/test_integration_offline.py
python medium_v4/driver.py verify-best
```

The final command should say `INDEPENDENT ORIGINAL ESA FITNESS: [200.89157838490243, 0, 0, 0, 0]` or a better **already verified local seed**.

### Important: 3-minute PyKEP smoke test first

```bash
python -u -B medium_v4/driver.py run --directory results/esa/medium_v4/smoke --minutes 3 --trials 60 --beam 12 --call-cap 15000
python medium_v4/driver.py status --directory results/esa/medium_v4/smoke
```

Inspect the `complete`, `new orders`, and `records` counters. This is the **only** way to determine whether the new search is making progress on actual orbital physics. A complete candidate is physically valid only when it has been accepted by the unchanged ESA Medium UDP. The new version **includes** the last inner→outer exception in the local window, which V3's same-shell-only mutations did not do.

### If real smoke succeeds: launch independent 90-minute campaign

```bash
bash medium_v4/start.sh
```

One low-priority worker. It reads V1, V2, V3 and the V4 smoke record as possible seeds, but never edits their files. A new campaign is at `results/esa/medium_v4/run_01/`.

### Monitor, stop, verify

```bash
bash medium_v4/status.sh
tail -n 20 results/esa/medium_v4/run_01/campaign.log
bash medium_v4/stop.sh
python medium_v4/driver.py verify-best
```

`Ctrl+C` when using `tail -f` only stops showing the log. `stop.sh` requests a graceful checkpoint and exits. The script exits automatically after the configured total budget (default 90 minutes).

### Resume from an interrupted campaign

The `--minutes` setting is the **total accumulated budget**, not additional minutes. For example, extend a partially complete or completed 90-minute experiment to 180 total minutes, if diagnostics justify it:

```bash
python -u -B medium_v4/driver.py run --directory results/esa/medium_v4/run_01 --minutes 180 --seed 20261122
```

For a detached resume (after checking the previous PID is gone):

```bash
nohup "$HOME/miniforge3/envs/kttsp-medium/bin/python" -u -B medium_v4/driver.py run --directory results/esa/medium_v4/run_01 --minutes 180 --seed 20261122 >> results/esa/medium_v4/run_01/campaign.log 2>&1 < /dev/null &
```

No dependency installs, no Git commands, no touching Small/Easy/Large files. It is safe to leave Codex's Easy campaign operating separately, but CPU contention is still possible, so monitor machine temperatures and RAM.

### Records and limits

- `best.json` — best **independently ESA-validated** result in this V4 campaign, including chromosome.
- `records/` — immutable saved ESA-validated improvements, with every transfer and ΔV.
- `checkpoint.json` — full search state, random generator state, adaptive method stats, feasible tour library.
- `campaign.log` — persistent monitoring information.
- `pid`, `stop.request` — local run controls.

The current real record before V4 is **200.89157838490243 days**, not 146 days. Do not confuse results printed by synthetic `test_integration_offline.py` with real ESA results. No claim of a new record is made unless it passes the unmodified original ESA `fitness()`.

# Paste this to the existing EASY/SMALL Codex session

We are splitting work: YOU OWN ONLY EASY/SMALL (49 tomatoes); another optimizer owns MEDIUM. Do not access or modify Medium or Large files or the V4 run directory. Do not commit/push.

Current independently ESA-validated Easy incumbent: **111.75601410426336 days**. Goal: approach or beat 101 days; NO guaranteed improvements. Both previous campaigns completed without improvement: 8-hour old beam (222 batches, 32 feasible orders) and the 90-minute V2 destroy-and-repair (45 episodes, thousands of feasible tours but best unchanged).

**Please research seriously before burning another long budget.** Study and cite Bannach/Acciarini/Izzo's 2026 KTSP time-expanded/dynamic-discretization paper; Weise et al. multi-step permutation encoding; the new public pykep-lambert transfer-table acceleration and its numerical parity caveats; and inspect existing successful public SpOC4 approaches when code is available. Do NOT assume any other competitor's solution vector is public.

**NEW concrete tasks:**
1. Audit whether V2's 45 episodes produced hundreds of complete routes with essentially the SAME last transfer departure/arrival; quantify distinct *final times*, not just distinct orders. Inspect checkpoints and entire candidate library. Derive the **earliest lower achievable time for the last transfer** under the fixed last several tomatoes and investigate whether that bottleneck persists across many routes.
2. Write an instrumented feasibility/quality funnel: candidate order → physically valid complete timing → feasible finish below incumbent → ESA-validated record. Aggregate acceptance/rejection by segment and true final-time improvement.
3. Develop **joint sequence+timing anchored neighborhood search** with backward-compatible suffix and accurate exception budgets, rather than choosing permutation then hoping to repair a 49-edge suffix. Focus on last 1–8 transfers as well as the **actual exceptionally expensive inter-shell transitions**.
4. Try *backward timing DP* on the last 5–12 legs, and a separate time-expanded reachability table around the orbital phase windows of the limiting transfers. Preserve more than the earliest phase in each beam; reason about reachability rather than raw Lambert counts.
5. Use the **original ESA Small evaluator** and PyKEP 2.6.x as the only record authority; do not silently replace PyKEP with Numba/Rust for final validation. If acceleration is explored, first write numerical parity tests including multi-revolution Lambert.
6. Stage benchmarks: a small set of 5–10 minute pilots with comparable CPU budgets and measured VALID complete routes, new finish times, and exact evaluator fitness. Abandon strategies that merely generate novel orders with no effect on the finishing time. Only then consider a 2-hour, or later longer, **detached** search with checkpoints and bounded low-priority 1–2 workers (Medium may be running on 1 CPU). Choose runtime on evidence, not arbitrary 8-hour spinning.
7. Continue from the current 111.75601410426336 incumbent, keep prior data and all best files. Run in a NEW result directory such as `results/esa/small_v3/`. Do not overwrite best Small. Set nice +12; preserve Easy environment and do not modify `kttsp-medium`.
8. Deliver code, exact commands, diagnoses, checkpoint/resume and real ESA validation. Report truthfully if nothing improves.

Start audit + code + short benchmarks now. Once validated, launch the productive detached Easy campaign and show the actual PID and log.

Sources:
- https://www.sciopen.com/article/10.1007/s42064-026-0325-5
- https://scholarprofiles.sunwayuniversity.edu.my/en/publications/solving-the-spoc-4-challenge-2-with-a-multi-step-permutation-enco/
- https://github.com/dietmarwo/pykep-lambert
- https://github.com/esa/SpOC4

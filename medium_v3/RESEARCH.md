# Investigation of ESA SpOC-4 Medium (181 targets)

Date: 22 November 2026. Inputs: uploaded original ESA Medium problem and evaluator, original verified 202.8125-day solution, logs provided by the user for V1 and V2. The more recent 200.8915914-day record was verified by the user using the **original ESA Medium fitness**; it is on the user's WSL device, not in the older ZIP uploaded for this analysis.

## 1. What is actually hard?

The ESA evaluator computes two-impulse Lambert transfers for each pair of consecutive targets. The objective is the last arrival time, not transfer count, orbital-plane DV, total flight-only time, or a TSP distance matrix. Each orbit moves over time, and transfer cost depends on both departure time and time of flight. The full chromosome has 541 entries; only the original ESA evaluator can certify a solution.

The original 202.8125-day seed has exactly 181 distinct targets, 180 legs, 191.3125 days in flight, 11.5 days waiting, and five ΔV exceptions. From `medium.kttsp`: 60 objects are at ~3,750 km orbital semimajor axis, 121 at ~15,000 km. The route has five between-shell transfers which consume all five exceptions. Its shell sequence is:

- Inner: 20 nodes.
- Outer: 52 nodes.
- Inner: 20 nodes.
- Outer: 68 nodes.
- Inner: 20 nodes.
- Outer: 1 node.

All 14 flights lasting strictly >3 days occur *inside the outer shell*. Outer-shell same-shell flights add to **172.875 days**, or **90.36% of all flight time**; inner-shell same-shell flights add to 16.375 days; five crossing legs total 2.0625 days. Long legs are mainly incline-ladder steps, some about 3°–6° apart. They represent the most important route-and-phase optimization opportunity, while waiting-time-only optimization is insufficient.

The outer shell contains clusters near 0°, 90°, and 180° and a sequence stepping through the inclinations. Slow long flights can appear even between nearby inclinations because Lambert feasibility depends on phase and time, not only orbital-plane separation. A pure geometric proxy cannot be treated as ΔV or time.

## 2. Diagnosis of earlier search

- V1 medium: 330,929 trials in 8 hours, 125,571 completed ESA evaluations, 966,705,710 Lambert calls. Best 200.8915914. It successfully reduced the mission by 1.9209086 days from 202.8125, but then stagnated. Its `greedy_suffix` optimizes each subsequent edge for earliest arrival, without enough lookahead. Feasibility can fail when an earlier choice uses up later ΔV exceptions or misses favorable orbital phases.
- V2 medium: 596 trials, 549 fully completed candidates, 22,105,677 Lambert calls, 30 minutes, no improvement over 200.8915914. Its fixed-order timing search keeps phase-diverse beam states but **cannot alter the underlying 181-target sequence**.
- Early attempts (A/B/C) had 13,342, 63,247 and 3,047,746 Lambert calls, without reaching a new ESA-valid complete tour. They failed to repair long suffixes after naive order mutations.

An 8-hour rerun of V1 or repeated fixed-order V2 does not address the source of stagnation.

## 3. Public research (2026)

1. **Weise, Abdipoor, Li, Wu (GECCO Companion, 2026)**, *Solving the SpOC 4 Challenge 2 with a Multi-Step Permutation Encoding*: https://thomasweise.github.io/assets/papers/WALW2026STS4C2WAMSPEFTKTS.pdf . Explicitly covers the 181-target Medium. It first builds a transfer reachability matrix (0=unreachable, 1=exception, 2=normal), then constructs plausible permutations before expensive trajectory evaluation. Their Medium reachability counts are about 45.9 normal and 17.6 exception-class accessible destinations per object under their sampling scheme. Their orbit accessibility is not an exact full-schedule guarantee. Their decoding merges path fragments, limits exception edges, constructs transfers, and switches to higher-DV reconstruction if necessary.
2. **Bannach, Acciarini, Izzo (2026)**, *The Keplerian Traveling Salesperson Problem*: https://arxiv.org/abs/2605.00010 . Formulates a time-expanded network and interval dynamic discretization discovery plus ILP, with preprocessing and heuristics. Time-dependent arcs matter, rather than Euclidean-TSP distance.
3. **dietmarwo/pykep-lambert**: https://github.com/dietmarwo/pykep-lambert . Public Izzo/Numba port benchmarked against PyKEP; its author reports 12.88× faster batch table generation on a benchmark. That performance is environment-specific. For safety, parity with original PyKEP must be established on the user's problem before using a faster approximate backend; V3 uses original ESA transfers only.
4. ESA SpOC4 official competition and validation: https://github.com/esa/SpOC4 and official leaderboard announcement https://github.com/esa/SpOC4/discussions/13 . The public leaderboard discussion does not publish an independently reusable 146-day Medium chromosome. A public result on the 1,051-target Large instance is not evidence that the 181-target Medium score has been solved to 146.

## 4. Offline candidate experiments in this ChatGPT runtime

`offline_experiment.py` (included in V3) tested 4,000 randomized proposals per operator against the original **202.8125-day order**, retaining different orders that preserved the same five shell crossings. The geometric proxy is the sum of the plane-change DV estimates (inclination and node angle) for same-shell edges, NOT an ESA-valid score and NOT a proven bound on trip time.

| Operator | Unique crossing-preserving orders | Orders with lower proxy | Best proxy delta (m/s aggregate) |
|---|---:|---:|---:|
| Adjacent swap | 174 | 60 | -59.9 |
| Short reversal | 928 | 238 | -75.3 |
| Short insertion | 1,639 | 391 | -119.4 |
| Near-plane same-shell swap | 1,173 | 355 | -89.0 |
| Long-leg local shuffle | 2,108 | 29 | -119.4 |

This is a **candidate generation screen only**; without original PyKEP we cannot claim any of these reduce arrival days or even satisfy real transfer constraints.

## 5. New strategy being tested in `medium_v3/`

**Stage A — Candidate topology search.** Preserve the 5 expensive shell changes, focus attention on expensive outer-shell legs, propose insertion/reversal/near-plane exchange/endgame changes; rank via a geometry proxy. This samples different order families rather than spending hours adjusting a fixed permutation.

**Stage B — ESA-exact local time-expanded beam.** For each new permutation, rebuild a bounded 3–12-edge segment with multiple departure/flight times using the unchanged official `compute_transfer()`, preserving exceptions and selecting time-phase-diverse states. Unlike a random global mutation, this starts from a complete validated route with a fixed prefix and suffix and changes only a small region.

**Stage C — Suffix improvement.** If a repaired segment joins an old verified suffix, it provides another complete ESA-feasible candidate (not necessarily an improvement). Additionally, recompute subsequent departure phases to exploit any locally banked time. The full 181-body chromosome is independently checked with `fitness()` before archival.

**Stage D — Continuous polishing.** Optionally use SciPy SLSQP on a few final legs. The function is nonsmooth (minimum over Lambert branches), so SLSQP is only a heuristic; any candidate must satisfy every official constraint and be rechecked.

**Stage E — adaptive scheduling.** Reward operators that create fully validated diverse tours and new records, and checkpoint all progress. Do not launch an 8-hour run until the 3-minute smoke and 45-minute benchmark show reasonable completed-tour throughput and at least useful candidates.

### What remains unproven

- No ESA-valid tour better than 200.8915914 has been obtained in this ChatGPT environment.
- 146 days may not be reachable with these heuristics. No bound has been established.
- Source candidate efficacy and fast-Lambert parity require the user's WSL PyKEP 2.6.4.
- If 45 minutes yields many complete but non-improving tours, move to a genuinely global transfer table / multistep permutation graph solver rather than rerunning V3 for hours unchanged.

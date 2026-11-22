# Medium V4 — research and diagnosis (22 Nov 2026)

## Evidence from the project, not claims of a new record

- Medium: 181 tomatoes / 180 transfers, chromosome length 541; original Medium problem is `data/spoc4/kttsp/problems/medium.kttsp`, evaluator is the project's unchanged original ESA `.py`.
- The user's **latest actual independently revalidated score** from local logs: `200.89157838490243` days (`medium_v3/run_01`). The baseline in the shared project ZIP is `202.8125` days; the V4 code loads newer local records when run in WSL.
- The original full tour uses all 5 exception legs on inner/outer shell crossings. Shell blocks in the original tour are indices 0–19 inner, 20–71 outer, 72–91 inner, 92–159 outer, 160–179 inner, 180 outer. Hence the **final transfer is a critical, exceptional inner→outer leg**, not a generic transfer within a shell.
- Of 180 original transfers, 14 have flight time >3 d; the original flight-time sum is 191.3125 d, waiting 11.5 d. About 172.875 d of flight time occurs in the outer shell. Those are original-tour metrics, not yet necessarily the exact updated V3 tour.
- Old V1 achieved 202.8125→200.8915914 with feasible tours; V2 produced 549 complete schedules but no new record; V3 made 5,891 trials, 38,122,677 Lambert calls, **one completed candidate**, no novel complete orders, and improved the V1 record by ~1.124 seconds with continuous timing.
- V3's official fitness penalties ending in 2, 3, 4 correspond to too many exceptions, not a small rounding error. The official exception penalty is `exceptions - 5`, allowed <=0. V4 counts exception availability in the modified window directly, reserving the exceptions required by the unchanged prefix/suffix.
- V3 selected an order first, then attempted timing/repair. This is too brittle near 200.9 d because feasible times depend on the ordering. V4 combines **subset ordering and timing in each beam state**, while retaining both anchors and the verified suffix.

## External findings

1. **Bannach, Acciarini & Izzo (2026)**: Keplerian TSP has time-dependent asymmetric transfers; use time-expanded networks, discretized optimization, ILP and dynamic discretization discovery. Source: https://www.sciopen.com/article/10.1007/s42064-026-0325-5
2. **Weise, Abdipoor, Li & Wu (GECCO 2026)**: multi-step permutation encoding can screen reachability before paying the physical transfer costs. Source: https://scholarprofiles.sunwayuniversity.edu.my/en/publications/solving-the-spoc-4-challenge-2-with-a-multi-step-permutation-enco/
3. **dietmarwo / pykep-lambert (2026)**: a Numba Lambert batch implementation reported a 12.88× speedup for table generation on its benchmark platform. This V4 package deliberately does **not** replace ESA PyKEP because precise parity for Medium's multi-revolution transfers needs independent testing before the new backend may be trusted. Source: https://github.com/dietmarwo/pykep-lambert
4. **ESA official competition and evaluation**: https://github.com/esa/SpOC4. The official evaluator (`udp.fitness()`) is our final authority, not any geometric model or synthetic tests.

## V4 search design

- **Anchored joint repair:** choose 2–5 interior tomatoes, permute their order AND calculate feasible departure epochs and durations at each step; each beam state is (visited mask, new partial ordering, arrival time, exception count, scheduled legs).
- **Preserve complete tours:** both endpoint tomatoes, the entire prefix, and the entire suffix are fixed during each splice. For nonfinal windows, arrival must precede the unchanged next departure. Therefore a successful splice is a *complete schedule*, ready for ESA verification.
- **Exact budget accounting:** number of exceptions available in window is 5 minus exceptions already consumed by unchanged prefix and suffix. All ΔV feasibility calculations use the original ESA `compute_transfer` through V1's bounded memoization.
- **Final crossing branch:** unlike a same-shell-only swap, V4 permits a terminal anchor in the other shell, so the last exceptional crossing can be timed jointly with the preceding inner-shell tomato sequence.
- **Terminal-only polish:** a second search changes the departure and flight time of the last crossing, locally refines feasible durations and leaves the other 179 legs unchanged.
- **Proven V1 fallback:** the `v1_retime` search remains as a reference arm, together with fresh joint phase and ordering arms.
- **Multi-solution archive:** imports validated V1 workers' feasible tours, retains up to 32 distinct orders, and can combine useful anchored splices across successive trials.
- **Adaptive online allocation:** each arm is rewarded for producing complete routes, genuinely new orders, and independently verified improvements, not merely Lambert-call count.
- **Safety and accounting:** 1 worker at `nice +12` with best-effort affinity; 90-minute default maximum, deterministic seeds, checkpoint every 60 seconds; graceful stop and resume; no writes outside a dedicated Medium V4 directory.

## What tests have and have not established

The authoring runtime does **not** provide PyKEP. Tests here used a **FAKE transfer oracle**, exercising time/ordering consistency, new complete visit orders, a last-leg cross-shell exception, full-route completion, saving, and checkpoint/resume. The baseline medium file was inspected structurally. **No new ESA Medium solution has been produced or validated by this V4 package in this runtime.**

The new method *should* have a more suitable feasibility construction than V3, but its real-world rate of complete, ESA-valid alternatives is an empirical question. Run the 3-minute real ESA smoke test and examine `complete`, `novel`, `records`, and failures. If it fails to produce a novel complete route, do not assume a 90-minute run will solve the problem; send the logs for further redesign.

The target 146 d remains an ambitious aspiration: from 200.891578 it would require **54.891578 days** saved. There is no lower bound or validated evidence here that 146 d is reachable.

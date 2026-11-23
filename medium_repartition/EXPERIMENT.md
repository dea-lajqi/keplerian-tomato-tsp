# Medium passage repartitioning and hard-cap repair

The experiment begins at Medium record 182.25426732810323. It uses the
unchanged original ESA evaluator and instance. The angular-momentum proof,
assumptions and input fingerprint are in
[../medium_frontier/CROSSING_BOUND.md](../medium_frontier/CROSSING_BOUND.md).

## Reproducing the repaired record

Use the existing Medium environment from the repository root. Use fresh
output directories for a rerun.

    export MPLCONFIGDIR=/tmp OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
    R=results/esa/medium_frontier/record_enriched/records/182.254267328103_f5531dbe.json
    P=results/esa/medium_deadline180
    N=$P/bridge_retiming/near/182.15_131_573fa0e5f2ad3cab.json
    python -B -m medium_repartition.near --seed-record "$R" --near "$N" \
      --directory results/esa/medium_repartition_replay/near --seconds 300
    python -B -m medium_repartition.prefix --seed-record "$R" --original "$N" \
      --near results/esa/medium_repartition_replay/near/replayed_near.json \
      --directory results/esa/medium_repartition_replay/prefix --seconds 315
    python -B -m medium_repartition.terminal --seed-record "$R" \
      --profiles results/esa/medium_repartition_replay/prefix/gate_pool.json \
      --atlas "$P/archive_phase/combined_events.json" \
      --directory results/esa/medium_repartition_replay/terminal --seconds 168 \
      --enrich-seconds 85

The first stage minimizes the largest ΔV violation, with hard chronology
and optional final arrival deadlines. The recovered prefix runs with four
exceptions and clears the last high-shell tomato at day 178.787442834.
The final low-shell itinerary is rebuilt from measured transfer windows.
Successful complete candidates pass the original evaluator and a separate
hard maximum of 600 m/s. Files under `near` are repair inputs.

## Membership and timing

The adaptive destroy-and-repair worker changes both high-shell visit orders,
their membership and lengths. It tracks actual departure/arrival windows
from a shared event bank. It switches early and late propagation at several
positions; exact timed subblocks repair the broken adjacencies in both
passages. Blocking windows and measured join gaps inform subsequent moves.

The complementary-set worker (`medium_repartition.seam_dp`) combines
forward and reverse labels over a common movable set at the polar crossing.
A label records visited tomatoes, endpoint, measured ready epoch, and
exception use. Each decision to cross into the middle low shell leaves an
exact complementary high-shell subset for the subsequent passage. The
worker reserves both middle-shell crossings and the terminal crossing
before admitting optional exceptions elsewhere. Its upstream and downstream
cores are held fixed within each trial. Changed core orders and normalized
all-normal cores can be tested in separate arms. Each resulting tour covers
every original index exactly once.

The day-180 terminal profiles come from the earlier measured graph. Later
profiles at 183.5, 184.5 and 185.2 days allow route families with useful
flight times to enter the archive. Progress and latest validated records
are in `results/esa/medium_repartition/nov23_campaign/campaign.json`.
A rerun may see different partial progress from wall-clock cutoffs.

    nice -n 11 python -B -m medium_repartition.campaign \
      --directory results/esa/medium_repartition_replay/campaign --seconds 3900

The event graph and orbital-node window scans give sampled opportunities.
The crossing-count proof provides a geometric resource bound independent of
these samples and gives no total-duration estimate.

## Reproducing the under-175 join

The exact head worker selects the earliest ready time for each visited subset
and endpoint on its sampled normal-transfer graph. It starts from measured
polar schedules plus an exceptional exit. It joins only to the timed
remainder of the second high-shell passage, with a terminal crossing and a
complete backward low-shell schedule. The fine 0.0325-day reverse mesh
connected to the day-174.99 windows. The 0.052-day control missed.
Replaying the exact join and polishing the full result separately keeps the
effect of each step measurable.

    D=results/esa/medium_repartition
    R=$D/coupled_175/records/175.394500000000_74e4588befe98.json
    python -B -m medium_repartition.head --seed-record "$R" \
      --events "$D/coupled_175/events.json" "$D/high_head/events.json" \
      "$D/mapped_clusters/events.json" "$D/mapped_early/events.json" \
      --terminal-extra "$D/under175" "$D/early_177" \
      --middle-profiles "$D/middle_subsets/middle_profiles.json" \
      --directory results/esa/medium_repartition_replay/target_175 \
      --days 174.99 174.82 175.07 175.22 174.63 174.3 \
      --width 20 --core-step .0325 --compile-seconds 0 --seconds 265

Use `python -B -m medium_repartition.polish` with the independently valid
174.981796875-day result to refine its timing. Use
`python -B -m medium_repartition.summary` for a strict replay of saved
records and a deduplicated archive count. The complete final record history
is in `results/esa/medium_repartition/results_summary.json`.

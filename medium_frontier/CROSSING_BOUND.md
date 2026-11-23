# Medium exceptional shell-crossing bound

The official model treats every leg as two endpoint impulses around a central
Keplerian coast. Let r1 and r2 be the encounter position vectors, h1 and h2 the
tomatoes' specific orbital angular momenta, and ht the constant specific
angular momentum of a Lambert transfer arc. At departure
ht − h1 = r1 × Δv1. At arrival h2 − ht = r2 × Δv2. The triangle inequality gives

    ||h2 − h1|| ≤ ||r1|| ||Δv1|| + ||r2|| ||Δv2||
                ≤ max(a1(1+e1), a2(1+e2)) (||Δv1|| + ||Δv2||).

This holds for every encounter epoch, direction and revolution count. It uses
the unchanged evaluator's sum of endpoint impulse norms and the elliptic
tomato orbits' apocentre radii. It does not require estimates of transfer time.

The calculation evaluates every pair of the supplied 181 orbits. It computes
h = sqrt(mu a(1-e²)) (sin(i) sin(Ω), −sin(i) cos(Ω), cos(i)).
The low shell (a < 8,000,000 m) has three 20-node groups, split at inclination
gaps greater than 0.5 rad. Pairwise minima between groups are 1552.990,
2260.527 and 1554.435 m/s, each above the 600 m/s exception cap.
Every low/high pair requires at least 282.860 m/s, above the 100 m/s normal cap.
The separations from these caps are far greater than numerical roundoff.

Project any complete itinerary to visits to the three low groups and the high
shell. Consecutive distinct low groups in that projection must be separated by
a high-shell visit, and touching all three low groups requires at least two
such transitions. Each costs two exceptional crossings. A high-shell start
or finish adds one crossing per high endpoint, so both endpoints cannot be
high under the five-exception budget. A low-to-low tour needs an even number
of crossings and therefore exactly four. These are necessary conditions.
They do not imply that a particular order is feasible or bound mission duration.

Reproduce from the repository root with the Medium Python environment:

    OPENBLAS_NUM_THREADS=1 python -B medium_frontier/crossing_bound.py
    sha256sum data/spoc4/kttsp/problems/medium.kttsp

Input SHA-256
`8601c11daaff300c91a1345c2aae6a9140ec034a9812439576ce17406f91ddec`.

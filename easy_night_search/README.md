# Easy overnight search
Extract folder into repository root. Run from root in existing PyKEP 2.6.4 environment:

`python -B easy_night_search/run.py --hours 8 --workers 1`

Smoke test first: `python -B easy_night_search/run.py --hours 0.02 --workers 1`

Automatically locates complete 49-node path/transfers JSON schedules, verifies seed using existing official evaluator wrapper. Use --incumbent PATH if automatic discovery fails. Uses global mutations, phased timing lattices, multiple readiness labels per exception count and Pareto transfer anchors. Heuristic, not exhaustive. Existing repository modules required. Outputs only to results/esa/easy_night_search. Restart discovers saved records. Ctrl+C stops. No Git actions. Syntax checked only; PyKEP integration not run in authoring environment. Keep one worker if Medium also runs.

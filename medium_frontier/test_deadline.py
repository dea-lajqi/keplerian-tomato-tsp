import itertools
import time
import unittest

from medium_coupled.exact_terminal import ExactTerminal
from medium_coupled.terminal_atlas import pareto
from medium_frontier.crossing_bound import crossing_bounds
from medium_frontier.deadline import Deadline
from pareto_easy import arrival


class DeadlineTests(unittest.TestCase):
    def test_latest_complete_terminal_schedule_matches_brute_force(self):
        rows = {
            (0, 1): pareto([(0, 1, 1., .25, 90.), (0, 1, 2., .5, 91.)]),
            (1, 2): pareto([(1, 2, 2., .5, 89.), (1, 2, 3., .5, 90.)]),
            (0, 2): ((0, 2, 1., 1., 93.),),
            (2, 1): ((2, 1, 2., .5, 99.),),
            (2, 0): ((2, 0, .75, .25, 90.),),
        }
        group = (0, 1, 2)
        due = 3.
        brute = {}
        for order in itertools.permutations(group):
            for legs in itertools.product(*(rows.get(p, ()) for p in itertools.pairwise(order))):
                if arrival(legs[0]) <= legs[1][2] and arrival(legs[-1]) <= due:
                    brute[order[0]] = max(brute.get(order[0], 0.), legs[0][2])
        exact = ExactTerminal.__new__(ExactTerminal)
        exact.stop = time.monotonic() + 5.
        _, labels = exact.execute(group, dict.fromkeys(group, -due), rows, 0., True)
        self.assertEqual({route[-1]: -clock for clock, route in labels}, brute)
        for clock, route in labels:
            legs = exact.reconstruct_reverse(route, rows, due)
            self.assertEqual(legs[0][2], -clock)
            self.assertLessEqual(arrival(legs[0]), legs[1][2])
            self.assertLessEqual(arrival(legs[-1]), due)

    def test_budget_classes_remain_in_bounded_frontier(self):
        search = Deadline.__new__(Deadline)
        values = {(3, 0, 0): (1.,), (5, 0, 0): (2.,), (6, 0, 0): (3.,),
                  (11, 0, 0): (4.,), (14, 0, 1): (10.,)}
        kept = search.retain(values, 3, lambda row: row[1][0])
        self.assertEqual(len(kept), 3)
        self.assertEqual({key[2] for key in kept}, {0, 1})

    def test_crossing_bound_on_official_instance(self):
        result = crossing_bounds()
        self.assertEqual(result["low_group_sizes"], [20, 20, 20])
        self.assertEqual(result["necessary_exceptional_crossings"], 4)
        self.assertGreater(result["low_to_high_minimum_m_per_s"], 282.)
        self.assertGreater(min(result["between_low_groups_minima_m_per_s"]), 1552.)


if __name__ == "__main__":
    unittest.main()

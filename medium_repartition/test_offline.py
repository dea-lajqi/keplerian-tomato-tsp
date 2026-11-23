import time
import unittest

from medium_repartition.caps import strict_schedule
from medium_repartition.seam_dp import SeamDP


class FakeUdp:
    min_tof = .01
    dv_threshold_exception = 600.

    def compute_transfer(self, a, b, departure, flight):
        return 599. + 10.*flight


class RepartitionTests(unittest.TestCase):
    def test_strict_cap_tightens_only_roundoff_sized_overage(self):
        near={"path":[0,1,2],"dep":[0.,.2],"tof":[.10000004,.04]}
        fixed=strict_schedule(FakeUdp(),near)
        self.assertIsNotNone(fixed)
        self.assertLessEqual(FakeUdp().compute_transfer(0,1,fixed["dep"][0],fixed["tof"][0]),600.)
        self.assertLessEqual(fixed["dep"][0]+fixed["tof"][0],fixed["dep"][1])
        near["tof"][0]=.10001
        self.assertIsNone(strict_schedule(FakeUdp(),near))

    def test_split_accepts_different_high_passage_sizes(self):
        search=SeamDP.__new__(SeamDP)
        search.high={1,2,4}
        self.assertEqual(search.split([0,1,2,3,4,5]),((0,),(1,2),(3,),(4,),(5,)))
        self.assertEqual(search.split([0,1,3,2,4,5]),((0,),(1,),(3,),(2,4),(5,)))

    def test_subset_layers_distinguish_exception_counts(self):
        search=SeamDP.__new__(SeamDP)
        search.normal=100.
        search.allowed={(a,b):True for a in range(3) for b in range(3)}
        search.normal_edges=search.allowed.copy()

        def choices(a,b,clock,extra,reverse,*_):
            yield a,b,clock,.4,80.
            if extra:
                yield a,b,clock,.2,200.

        search.events=choices
        rows=search.layers([1,2],{(0,0):(0.,0.,())},1,.04,5.,
                                 time.monotonic()+1.,False)
        self.assertEqual(rows[3,2,0][0],.8)
        self.assertAlmostEqual(rows[3,2,1][0],.6)
        self.assertTrue(all(key[2]<=1 for key in rows))


if __name__=="__main__":
    unittest.main()

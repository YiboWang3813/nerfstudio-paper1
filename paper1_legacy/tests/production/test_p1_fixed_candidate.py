import unittest
from scripts.production.ablations.ours.p1_fixed_candidate_report import paired,rank,METRICS


class CandidateStatisticsTest(unittest.TestCase):
    def test_all_metric_directions(self):
        for m in METRICS:
            p=paired([.01,.02,.03],m)
            self.assertEqual(p['wins'],3 if m in ('psnr','ssim') else 0)
            self.assertEqual(p['losses'],0 if m in ('psnr','ssim') else 3)
            self.assertAlmostEqual(p['std_paired_delta'],.01)
            self.assertAlmostEqual(p['median_delta'],.02)

    def test_ties(self):
        self.assertEqual(paired([0.,1e-7,-1e-7],'l1')['ties'],3)

    def test_rank_directions_and_exact_tie(self):
        self.assertEqual(rank(2,[1,2,3],'psnr'),2)
        self.assertEqual(rank(2,[1,2,3],'lpips'),2)
        self.assertEqual(rank(0,[1,2,3],'dists'),1)

    def test_reject_nonfinite(self):
        with self.assertRaises(AssertionError):paired([0,float('nan')],'gmsd')

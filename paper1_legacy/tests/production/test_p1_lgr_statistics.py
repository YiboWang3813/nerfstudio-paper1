import unittest
from scripts.production.ablations.ours.p1_lgr_report import paired_statistics


class PairedStatisticsTest(unittest.TestCase):
    def test_psnr_direction_and_sample_std(self):
        r=paired_statistics([-1.,0.,1.],'psnr')
        self.assertEqual((r['wins'],r['ties'],r['losses']),(1,1,1))
        self.assertEqual(r['std_paired_delta'],1.)

    def test_lpips_reverses_only_win_direction(self):
        r=paired_statistics([-.1,-.2,.1],'lpips')
        self.assertEqual((r['wins'],r['ties'],r['losses']),(2,0,1))
        self.assertAlmostEqual(r['mean_delta'],-.2/3)

    def test_numeric_tie(self):
        r=paired_statistics([1e-7,-1e-7],'ssim')
        self.assertEqual(r['ties'],2)

import unittest
from scripts.production.ablations.ours.p1_lgr_interaction_report import interaction_delta
from scripts.production.ablations.ours.p1_lgr_report import paired_statistics


class InteractionTest(unittest.TestCase):
    def test_no_interaction(self):
        self.assertEqual(interaction_delta(10,12,11,13),0)

    def test_interaction_not_absolute_superiority(self):
        self.assertEqual(interaction_delta(8,11,10,12),1)
        self.assertLess(11,12)

    def test_lpips_direction(self):
        x=interaction_delta(.3,.2,.2,.15)
        self.assertAlmostEqual(x,-.05)
        self.assertEqual(paired_statistics([x,x],'lpips')['wins'],2)

import unittest
from scripts.production.ablations.ours.p1_checkpoint_report import efficiency_metadata


class EfficiencyMissingTest(unittest.TestCase):
    def test_missing_not_zero_or_silently_partial(self):
        r=efficiency_metadata([{'train_wall_time_sec':'10'},{'train_wall_time_sec':''}])
        self.assertEqual(r['train_wall_time_sec'],'')
        self.assertEqual(r['time_available_cases'],1)
        self.assertEqual(r['time_expected_cases'],2)
        self.assertEqual(r['available_case_time_mean'],10)

    def test_complete(self):
        self.assertEqual(efficiency_metadata([{'train_wall_time_sec':10},{'train_wall_time_sec':20}])['train_wall_time_sec'],15)

    def test_nonfinite(self):
        self.assertEqual(efficiency_metadata([{'train_wall_time_sec':'nan'}])['time_available_cases'],0)

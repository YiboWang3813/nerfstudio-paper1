import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.production.ablations.ours.p1_checkpoint_cleanup import METRICS, validated_targets


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='p1-retention-')
        self.root=Path(self.temp.name)
        self.run=self.root/'runs'/'case'
        for name in ('logs','eval','vis/item_1','checkpoints'):(self.run/name).mkdir(parents=True)
        for name in ('resolved_config.yaml','run_manifest.yaml'):(self.run/name).write_text('present')
        (self.run/'batch1_stats.json').write_text(json.dumps(dict(success=True,finite_final_parameters=True,final_num_gaussians=10)))
        (self.run/'logs/scalars.tsv').write_text('step\tkey\tvalue\n'+''.join('30000\ttest/{}\t1\n'.format(m) for m in METRICS)+'30000\ttrain/num_gaussians\t10\n')
        (self.run/'eval/per_frame_metrics_iteration_030000.tsv').write_text('index\t'+'\t'.join(METRICS)+'\n'+'10\t'+'\t'.join('1' for _ in METRICS)+'\n')
        for name in ('pred.png','gt.png','diff.png','compose.png'):(self.run/'vis/item_1'/name).write_bytes(b'fixture')
        for name in ('solver','container'):(self.run/'checkpoints'/('iteration_30000_'+name+'.pth')).write_bytes(b'fixture')
        self.row=dict(output_dir=str(self.run),group='main',target_status='RERUN')

    def tearDown(self):self.temp.cleanup()

    def targets(self, control=None, fps=False):
        return [p.name for p in validated_targets(self.row,control if control is not None else {'exit_code':0},[10],[self.root/'runs'],fps)]

    def test_main_retains_model_until_fps(self):
        self.assertEqual(self.targets(),['iteration_30000_solver.pth'])

    def test_main_after_fps(self):self.assertEqual(len(self.targets(fps=True)),2)

    def test_structural_exports_suffice(self):
        self.row['group']='structural';self.assertEqual(len(self.targets()),2)

    def test_failed_or_active_retained(self):
        self.assertEqual(self.targets({'exit_code':1}),[])
        self.assertEqual(self.targets({}),[])

    def test_anchor_untouched(self):
        self.row['target_status']='FINAL-REUSE';self.assertEqual(self.targets(fps=True),[])

    def test_missing_image_retained(self):
        (self.run/'vis/item_1/gt.png').unlink();self.assertEqual(self.targets(),[])

    def test_nonfinite_metric_retained(self):
        p=self.run/'logs/scalars.tsv';p.write_text(p.read_text().replace('test/psnr\t1','test/psnr\tnan'))
        self.assertEqual(self.targets(),[])

    def test_outside_scope_rejected(self):
        with self.assertRaises(ValueError):validated_targets(self.row,{'exit_code':0},[10],[self.root/'other'])


if __name__=='__main__':unittest.main()

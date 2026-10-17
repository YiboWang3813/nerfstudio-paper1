"""Audit textual ablation artifacts without loading models or starting jobs."""
import csv
import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / 'docs/production/final_ablation_tables_20260916'


def rows(path):
    with (P / path).open() as f:
        return list(csv.DictReader(f, delimiter='\t'))


def test_raw_sizes_and_unrounded_aggregates():
    for rel, n in [('joint/raw.tsv', 16), ('afgm/raw_representative.tsv', 8), ('afgm/raw_multisweep.tsv', 32), ('plgr/raw.tsv', 55), ('cgas/raw.tsv', 10)]:
        assert len(rows(rel)) == n
    checked = rows('frozen_aggregate_verification.tsv')
    assert len(checked) == 156
    assert max(Decimal(r['absolute_difference']) for r in checked) < Decimal('5e-13')


def test_factorial_effects_and_population_deltas():
    effects = rows('joint/mean_component_effects.tsv')
    for ds, expected in [('Liver', '3.863299175'), ('Arm', '10.339061125')]:
        row = next(r for r in effects if r['dataset'] == ds and r['module'] == 'PLGR')
        assert Decimal(row['mean_delta_PSNR']) == Decimal(expected)
    transitions = rows('cgas/transitions.tsv')
    for ds, expected in [('Liver', 43672), ('Arm', 22418)]:
        r = next(r for r in transitions if r['dataset'] == ds and r['transition'] == 'RGD + LP -> Full CGAS')
        assert Decimal(r['delta_G']) == expected
    assert all(Decimal(r['delta_PSNR']) > 0 for r in transitions if r['transition'] == 'RGD + OP -> RGD + LP')


def test_sample_std_and_paired_cases():
    from statistics import mean, stdev
    raw = rows('afgm/raw_multisweep.tsv')
    for r in rows('afgm/multisweep_summary_raw.tsv'):
        sub = [x for x in raw if x['dataset'] == r['dataset'] and x['variant'] == r['variant']]
        assert len(sub) == int(r['n'])
        for metric in ['PSNR', 'SSIM']:
            for suffix, fun in [('mean', mean), ('std', stdev)]:
                assert abs(fun(float(x[metric]) for x in sub) - float(r[metric + '_' + suffix])) < 1e-12


def test_quality_only_ranking_and_new_terminology():
    ranks = rows('rank_display_audit.tsv')
    assert len(ranks) == 122
    assert all('G_k' not in r['metric'] for r in ranks)
    for section in ['joint', 'afgm', 'plgr', 'cgas']:
        for filename in ['table.md', 'table_latex.tex']:
            assert not re.search(r'\b(PPP|LGR|CGA)\b', (P / section / filename).read_text())
    for r in rows('rounding_ties.tsv'):
        assert r['exact_tie'] == 'False'


def test_all_frozen_inputs_unchanged():
    manifest = json.loads((P / 'source_manifest.json').read_text())
    source = Path(manifest['frozen_package'])
    for rel, expected in manifest['input_sha256'].items():
        assert hashlib.sha256((source / rel).read_bytes()).hexdigest() == expected
    assert manifest['unique_run_configs'] == 96
    assert manifest['no_training'] and manifest['no_jobs'] and manifest['no_manuscript_edits']

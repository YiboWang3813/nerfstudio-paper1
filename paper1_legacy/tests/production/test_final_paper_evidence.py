"""Read-only final evidence integrity tests; never run training."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'docs/production/final_paper_evidence_20260910'


def read(name):
    with (P/name).open() as f:return list(csv.DictReader(f,delimiter='\t'))


def test_package_hashes_and_source_configs():
    m=json.loads((P/'audits/final_data_source_manifest.json').read_text())
    assert m['training_submitted'] is False
    assert m['P2_source_sha256']=='f0ef8b8a16f36c57cd30e43c50f86b0262c18f8f65b8d7d33babb5bd67534a27'
    for path,h in m['package_file_sha256'].items():assert hashlib.sha256((P/path).read_bytes()).hexdigest()==h,path
    for path,h in m['analytical_inputs'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
    assert len(m['run_sources'])==238
    for r in m['run_sources']:
        assert (P/r['packaged_config']).read_bytes()==Path(r['source_config']).read_bytes()


def test_tables_and_nonmonotonic_quadrature():
    assert len(read('main/final_main_table_data.tsv'))==32
    assert len(read('main/final_main_rank_data.tsv'))==192
    assert len(read('joint/final_joint_table.tsv'))==8
    assert len(read('ppp/final_ppp_table.tsv'))==4
    assert len(read('ppp/final_ppp_multisweep.tsv'))==6
    assert len(read('ppp/rho_data.tsv'))==14
    assert len(read('lgr/final_lgr_table.tsv'))==5
    assert len(read('cga/final_cga_table.tsv'))==5
    rows=read('lgr/Nq_data.tsv')
    for ds in ['synthetic_liver','tus_rec_2024_qc']:
        v={int(float(r['point'])):float(r['final_psnr']) for r in rows if r['dataset']==ds}
        assert v[2]<v[1]<v[3]
        assert v[3]-v[2]>.4


def test_pruning_saturation_and_active_path_control():
    rows=[r for r in read('cga/selection_behavior_summary.tsv') if r['panel']=='qp_by_qA']
    assert len(rows)==18
    assert all(int(r['zero_C_threshold_events'])==int(r['events'])==12 for r in rows)
    for r in rows:
        qa=float(r['point'].split('_')[1])
        assert abs(float(r['path_protection_fraction_min'])-(1-qa))<1e-4
        assert abs(float(r['path_protection_fraction_max'])-(1-qa))<1e-4


def test_figures_and_no_missing_categories():
    for name in ['ppp/H1','lgr/H2','cga/H3']:
        assert (P/(name+'.pdf')).read_bytes().startswith(b'%PDF')
        assert (P/(name+'.png')).read_bytes().startswith(b'\x89PNG')
    text=(P/'audits/FINAL_EXPERIMENT_COMPLETENESS.md').read_text()
    assert text.count('| COMPLETE |')==12
    assert 'additional training experiments? NO.' in text

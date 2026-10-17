"""Checks frozen plotting artifacts, never evaluates a model."""
import csv
import hashlib
import json
import re
import subprocess
from decimal import Decimal as D
from pathlib import Path

P = Path(__file__).resolve().parents[2] / 'docs/production/final_hyperparameter_figures_20260916'


def rows(rel):
    with (P / rel).open() as f:
        return list(csv.DictReader(f, delimiter='\t'))


def test_rho_semantics():
    data = rows('H1_AFGM/plotted_data.tsv')
    liver = sorted([r for r in data if r['dataset']=='synthetic_liver'],key=lambda r:float(r['rho']))
    assert len(liver)==7
    for metric in ['psnr','ssim','lpips']:
        assert all(D(a['final_'+metric])>D(b['final_'+metric]) for a,b in zip(liver,liver[1:]))
    arm=[r for r in data if r['dataset']=='tus_rec_2024_qc']
    best=max(arm,key=lambda r:D(r['final_psnr']))
    assert D(best['rho'])==D('1.25')
    assert D(best['delta_psnr'])==D('.0689481')


def test_nonmonotonic_quadrature():
    data=rows('H2_PLGR/plotted_data.tsv')
    assert len(data)==40
    for ds in ['synthetic_liver','tus_rec_2024_qc']:
        q={int(float(r['point'])):D(r['final_psnr']) for r in data if r['panel']=='Nq' and r['dataset']==ds}
        assert q[2]<q[1] and q[2]<q[3]


def test_cgas_stability_and_saturation():
    data=rows('H3_CGAS/plotted_data.tsv')
    assert len(data)==42
    for ds in ['synthetic_liver','tus_rec_2024_qc']:
        for panel in ['qd_by_B','qp_by_qA','tau_alpha']:
            v=[D(r['final_psnr']) for r in data if r['dataset']==ds and r['panel']==panel]
            assert max(v)-min(v)<D('.016')
    events=[r for r in rows('H3_CGAS/selection_behavior_audit.tsv') if r['panel']=='qp_by_qA']
    assert sum(int(r['events']) for r in events)==216
    assert all(r['events']==r['zero_C_threshold_events'] for r in events)


def test_population_deduplicated_and_unpruned():
    data=rows('H4_CGAS_population/plotted_data.tsv')
    assert len(data)==46
    assert len({r['output_dir'] for r in data})==46
    assert sorted(D(r['final_G']) for r in data if r['structural_variant']=='RGD')==[D(1022586),D(1050672)]


def test_pdf_canvas_vector_and_frozen_inputs():
    renders=json.loads((P/'render_audit.json').read_text())
    for r in renders:
        assert not r['outside_canvas_text']
        info=subprocess.check_output(['pdfinfo',r['path']+'.pdf'],text=True)
        assert re.search(r'Page size:\s+252 x ',info)
        images=subprocess.check_output(['pdfimages','-list',r['path']+'.pdf'],text=True)
        assert not re.search(r'^\s*1\s+\d+\s+image',images,re.M)
    audit=json.loads((P/'source_audit.json').read_text())
    for rel,h in audit['input_sha256'].items():
        assert hashlib.sha256((Path(audit['source_root'])/rel).read_bytes()).hexdigest()==h

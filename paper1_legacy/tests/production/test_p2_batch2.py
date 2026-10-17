"""Submission/postprocessing tests; synthetic reports live only in pytest tmp dirs."""
import copy
import hashlib
import json
from pathlib import Path
from scripts.production.ablations.ours import p2_batch2 as b
from scripts.production.ablations.ours import p2_batch2_report as rep


def test_frozen_matrix_and_partition():
    from omegaconf import OmegaConf
    b.base.guard()
    rows=json.loads((b.DOC/b.MATRIX).read_text())
    assert len(rows)==68 and sum(r['execution_authorized'] for r in rows)==66
    assert len({r['semantic_config_sha256'] for r in rows})==68
    ids=sum(json.loads((b.DOC/'worker_partition.json').read_text())['slots'],[])
    assert len(ids)==len(set(ids))==66
    assert set(ids)=={r['batch1_id'] for r in rows if r['execution_authorized']}
    for r in rows:
        cfg=OmegaConf.to_container(OmegaConf.load(b.DOC/'exact_configs'/(r['batch1_id']+'.yaml')),resolve=False)
        assert hashlib.sha256(json.dumps(b.base.effective(cfg),sort_keys=True).encode()).hexdigest()==r['semantic_config_sha256']
        sr=cfg['model']['slice_render']
        assert sr['local_conditioning_mode']=='fixed'
        assert sr['us_focus_sigma_ele_min']<=sr['us_focus_sigma_ele_0']<=sr['us_focus_sigma_ele_max']
        assert cfg['system']['seed']==42


def test_report_draft_pipeline_with_fixture_metrics(tmp_path,monkeypatch):
    rows=json.loads((b.DOC/b.MATRIX).read_text())
    defaults={r['dataset']:r for r in rows if not r['execution_authorized']}
    source=rep.previous.read(b.base.OUT/'reports/completed_per_run.tsv')
    anchor={ds:next(r for r in source if r['output_dir']==default['output_dir']) for ds,default in defaults.items()}
    for r in rows:r['output_dir']=defaults[r['dataset']]['output_dir']
    doc=tmp_path/'doc';doc.mkdir()
    (doc/b.MATRIX).write_text(json.dumps(rows))
    (doc/'panel_points.json').write_text((b.DOC/'panel_points.json').read_text())
    out=tmp_path/'TEST_FIXTURE_ONLY'
    def fake_collect(r,expected):
        v=copy.deepcopy(anchor[r['dataset']]);v['row_id']=r['batch1_id']
        for k in list(v):
            if k.startswith('final_'):v[k]=float(v[k])
        return v
    monkeypatch.setattr(rep,'DOC',doc);monkeypatch.setattr(rep,'OUT',out)
    monkeypatch.setattr(rep.previous,'collect',fake_collect)
    rep.report()
    assert json.loads((out/'reports/progress.json').read_text())['pending']==[]
    assert len(rep.previous.read(out/'reports/all_panel_points.tsv'))==82
    assert len(rep.previous.read(out/'reports/parameter_ranges.tsv'))==16
    for name in ['H1_PPP_rho','H2_LGR_delta_psnr','H2_LGR_delta_lpips','H3_CGA_grids_delta_psnr','H3_CGA_tau_quality_population']:
        assert (out/'reports'/(name+'.png')).stat().st_size>1000
        assert (out/'reports'/(name+'.pdf')).stat().st_size>1000

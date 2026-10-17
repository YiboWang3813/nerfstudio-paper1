"""P2 changes only the production Hydra default; legacy backend behavior stays intact."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import pytest
import torch
from scripts.production.ablations.ours.p1_fixed_candidate import effective
from scripts.production.ablations.ours.p0_renderer_probe import sample
from ours_renderer.autograd import _us_focus_params_vector
from ours_renderer.us_focus_reference import render_us_focus_reference

ROOT=Path(__file__).resolve().parents[3]
P05=ROOT/'outputs/preproduction/regression/p05_corrected_covariance_20260908/freeze_v2'


def test_algorithm_and_binary_unchanged():
    allowed={'configs_hydra/model/ours_us_focus.yaml','configs_hydra/experiment/production/ablations/ours/paper_semantic_freeze.yaml'}
    for filename in ('hashes.json','binary_hashes.json'):
        for name,digest in json.loads((P05/filename).read_text()).items():
            if name.startswith(('docs/','scripts/','tests/')) or name in allowed:continue
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name


@pytest.mark.parametrize('dataset',['synthetic_liver','spine_phantom','tus_rec_2024_qc','tdsc_abus_2023'])
def test_production_default_exact_candidate_config(dataset):
    from configs.setup import load_hydra_config
    from omegaconf import OmegaConf
    rows=json.loads((ROOT/'docs/production/p1_fixed_main_candidate/fixed_full_main_candidate_matrix.json').read_text())
    r=next(r for r in rows if r['dataset']==dataset)
    overrides=[v for v in r['overrides'] if v.lstrip('+').split('=')[0]!='model.slice_render.local_conditioning_mode']
    cfg=OmegaConf.to_container(load_hydra_config('production/ablations/ours/paper_semantic_freeze',overrides),resolve=True)
    old=OmegaConf.to_container(OmegaConf.load(Path(r['output_dir'])/'resolved_config.yaml'),resolve=False)
    assert cfg['model']['slice_render']['local_conditioning_mode']=='fixed'
    assert effective(cfg)==effective(old)
    sr=cfg['model']['slice_render']
    for key,value in {'us_focus_sigma_ele_0':1.,'us_focus_det_gamma':.25,'ray_attenuation_beta':.8,'ray_attenuation_strength':.55,'ray_attenuation_min_transmission':.35,'ray_attenuation_quadrature_points':3,'ray_attenuation_quadrature_window_mm':2.}.items():assert sr[key]==value
    assert sr['ray_attenuation_formula']=='integral' and sr['ray_attenuation_source']=='attenuation'


def test_fixed_reference_and_packed_parameters_identical():
    cfg,m,s,q,a,i,tau,pose=sample('cpu','fixed')
    equivalent=replace(cfg,local_conditioning_mode='adaptive',us_focus_sigma_ele_1=0.)
    assert torch.equal(_us_focus_params_vector(cfg,m.device),_us_focus_params_vector(equivalent,m.device))
    assert torch.equal(render_us_focus_reference(cfg,m,s,q,a,i,pose),render_us_focus_reference(equivalent,m,s,q,a,i,pose))

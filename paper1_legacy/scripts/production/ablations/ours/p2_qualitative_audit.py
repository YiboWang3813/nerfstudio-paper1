"""Read-only image/source checks; writes audit metadata only, never edits images."""
import hashlib
import json
from pathlib import Path
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
from scripts.production.ablations.ours import p2_refresh as p2


def run():
    p2.guard(); root=p2.ROOT
    main=p2.MAIN/'reports/qualitative/paper_all_datasets_2d_rendering_manifest.json'
    contact=p2.OUT/'reports/contact_sheets/manifest.json'
    joint=json.loads((root/'docs/production/ablation_final_outputs/manifests/paper_ablation_joint_visual_manifest.json').read_text())
    internal=json.loads((root/'docs/production/ablation_final_outputs/manifests/paper_ablation_internal_visual_manifest.json').read_text())
    rows=json.loads((p2.DOC/p2.MATRIX).read_text()); consumers=json.loads((p2.DOC/'consumers.json').read_text())
    byid={r['batch1_id']:r for r in rows}
    panel_meta=[];gt_cache={}
    def check(source,crop,key,gt):
        im=np.asarray(Image.open(source).convert('L')); truth=np.asarray(Image.open(gt).convert('L'))
        assert im.shape==truth.shape
        x,y,w,h=crop;assert x>=0 and y>=0 and w>0 and h>0 and x+w<=im.shape[1] and y+h<=im.shape[0]
        if key in gt_cache:assert np.array_equal(gt_cache[key],truth),(key,'GT pixels differ')
        else:gt_cache[key]=truth
        panel_meta.append(dict(key=list(key),source=str(source),crop_xywh=crop,image_shape=list(im.shape),sha256=p2.sha(source),gt_pixel_sha256=hashlib.sha256(truth.tobytes()).hexdigest()))
    def config_check(row):
        cfg=OmegaConf.to_container(OmegaConf.load(Path(row['output_dir'])/'resolved_config.yaml'),resolve=False)
        digest=hashlib.sha256(json.dumps(p2.effective(cfg),sort_keys=True).encode()).hexdigest()
        assert digest==row['semantic_config_sha256']
        stats=json.loads((Path(row['output_dir'])/'batch1_stats.json').read_text());assert stats['success'] and stats['finite_final_parameters']
        return cfg
    main_rows=json.loads(main.read_text())['panels']
    from scripts.production.comparisons.us_recon_four_dataset_p100k_v1 import make_tmi_all_datasets_2d_rendering_v2 as drawing
    # The immutable figure spec remains the case/frame/crop authority.
    for r in main_rows:
        src=Path(r['source_image']);gt=src.parent/'gt.png'
        check(src,r['crop_xywh'],('main',r['dataset'],r['sweep'],r['item']),gt)
        if r['method']=='ours':
            row=next(x for x in rows if x['dataset']==r['dataset'] and x['sweep']==r['sweep'] and 'main' in x['consumers'])
            assert src==Path(row['output_dir'])/'vis'/('item_'+str(r['item']))/'pred.png'
            cfg=config_check(row);assert cfg['model']['slice_render']['local_conditioning_mode']=='fixed'
    # Compare selection to pre-P2 frozen main figure manifest, never pick new crops.
    old=root/'outputs/production/comparisons/paper_final_corrected_covariance_20260908/reports'
    candidates=list(old.rglob('paper_all_datasets_2d_rendering_manifest.json'))
    assert candidates, 'Missing previous frozen main selection'
    frozen=json.loads(candidates[0].read_text())['panels']
    fields=['dataset','sweep','item','crop_xywh','method']
    assert [[r[k] for k in fields] for r in main_rows]==[[r[k] for k in fields] for r in frozen]
    for r in json.loads(contact.read_text()):
        ds=r['dataset'];group=r['group'];src=Path(r['source'])
        spec=next(x for x in joint if x['dataset']==ds) if group=='joint' else next(x for x in internal if x['group']=={'ppp':'Easy-Pose','cga':'DC','lgr':'Renderer'}[group] and x['dataset_label']==('Synthetic' if ds=='synthetic_liver' else 'TUS-QC'))
        assert r['item']==spec['item'] and r['crop_xywh']==spec['crop_xywh']
        assert p2.sha(src)==r['sha256'] and p2.sha(src.parent/'gt.png')==r['gt_sha256']
        check(src,r['crop_xywh'],(group,ds,r['sweep'],r['item']),src.parent/'gt.png')
        if r['label']!='GT':
            c=next(x for x in consumers if x['dataset']==ds and x['sweep']==r['sweep'] and x['consumer']==group+'/'+r['label'])
            row=byid[c['row_id']]
            assert src==Path(row['output_dir'])/'vis'/('item_'+str(r['item']))/'pred.png'
            config_check(row)
    dest=root/'docs/production/p2_batch2_final'
    output=dict(main_panels=len(main_rows),contact_panels=len(panel_meta)-len(main_rows),same_gt_groups=len(gt_cache),all_checks_passed=True,
                source_sha256=(p2.FREEZE/'snapshot.sha256').read_text().strip(),previous_main_manifest=str(candidates[0]),panels=panel_meta,
                limitation='Checks establish archived frame/crop/source consistency, not bitwise reconstruction of deleted checkpoints or clinical validity.')
    (dest/'qualitative_integrity_checks.json').write_text(json.dumps(output,indent=2)+'\n')
    print('QUALITATIVE INTEGRITY PASS',len(main_rows),'main panels;',len(panel_meta)-len(main_rows),'contact panels')


if __name__=='__main__':run()

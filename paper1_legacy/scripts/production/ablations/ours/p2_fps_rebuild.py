"""Explicitly authorized eight-model FPS reconstruction; original metrics untouched."""
import copy
import csv
import json
import sys
from pathlib import Path
from scripts.production.ablations.ours import p1_fixed_candidate as worker_impl
from scripts.production.ablations.ours.p1_fixed_candidate import effective,base

ROOT=base.ROOT
DOC=ROOT/'docs/production/p2_fps_rebuild'
OUT=ROOT/'outputs/production/comparisons/P2_fixed_full_final_20260909/fps_rebuild'
MATRIX='fps_rebuild_matrix.json'


def plan():
    from omegaconf import OmegaConf
    import hashlib
    base.guard();DOC.mkdir(exist_ok=True);(DOC/'exact_configs').mkdir(exist_ok=True)
    assert not (DOC/MATRIX).exists()
    old=list(csv.DictReader((ROOT/'outputs/production/comparisons/us_recon_four_dataset_p100k_v1/tables/render_fps_complete_cached_ours_per_run.tsv').open(),delimiter='\t'))
    selected={(r['dataset'],r['sweep']) for r in old if r['method']=='ours'};assert len(selected)==8
    candidates=json.loads((worker_impl.DOC/worker_impl.MATRIX).read_text());rows=[]
    for source in candidates:
        if (source['dataset'],source['sweep']) not in selected:continue
        row=copy.deepcopy(source);row.update(batch1_id='fr_{:03d}'.format(len(rows)),group='main',study='FPS_model_reconstruction_only',execution_authorized=True,target_status='NEW',quality_source_dir=source['output_dir'])
        row['output_dir']=str(OUT/'runs'/row['dataset']/row['sweep']/row['batch1_id']);assert not Path(row['output_dir']).exists()
        cfg=OmegaConf.to_container(OmegaConf.load(Path(source['output_dir'])/'resolved_config.yaml'),resolve=False)
        assert hashlib.sha256(json.dumps(effective(cfg),sort_keys=True).encode()).hexdigest()==row['semantic_config_sha256']
        stats=json.loads((Path(source['output_dir'])/'batch1_stats.json').read_text());assert stats['success']
        row['estimated_seconds']=stats['wrapper_wall_seconds'];row['consumers']=['FPS_ONLY_NOT_QUALITY_REPLACEMENT']
        OmegaConf.save(OmegaConf.create(cfg),DOC/'exact_configs'/(row['batch1_id']+'.yaml'));rows.append(row)
    assert len(rows)==8
    bins=[[] for _ in range(4)];loads=[0.]*4
    for r in sorted(rows,key=lambda r:r['estimated_seconds'],reverse=True):
        slot=min(range(4),key=lambda s:loads[s]);bins[slot].append(r['batch1_id']);loads[slot]+=r['estimated_seconds']
    (DOC/MATRIX).write_text(json.dumps(rows,indent=2)+'\n');(DOC/'worker_partition.json').write_text(json.dumps(dict(slots=bins,estimated_seconds=loads),indent=2)+'\n')
    orig=ROOT/'scripts/production/ablations/ours/p1_batch1_run.py';sealed=json.loads((base.OLD/'submission_hashes.json').read_text());assert base.sha(orig)==sealed[str(orig.relative_to(ROOT))]
    code=orig.read_text().replace("PROTOCOL = ROOT/'docs/production/p1_paper_protocol_freeze'","PROTOCOL = ROOT/'docs/production/p2_fps_rebuild'").replace("PROTOCOL/'batch1_matrix.json'","PROTOCOL/'"+MATRIX+"'").replace("'Batch1_only'","'FPS_rebuild_only'")
    (DOC/'executed_runner.py').write_text(code)
    files=list((DOC/'exact_configs').glob('*.yaml'))+[DOC/MATRIX,DOC/'worker_partition.json',DOC/'executed_runner.py',Path(__file__).resolve(),Path(worker_impl.__file__).resolve(),Path(base.__file__).resolve()]
    files += [ROOT/'scripts/production/ablations/ours'/n for n in ('p2_fps_rebuild_gpu.sh','p2_fps_rebuild_post.sh','p2_fps_rebuild_report.py')]
    (DOC/'submission_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):base.sha(p) for p in files},indent=2)+'\n')
    print('Eight authorized FPS model reconstructions. No checkpoint cleanup. Worker seconds:',loads)


if __name__=='__main__':
    if sys.argv[1]=='plan':plan()
    elif sys.argv[1]=='run':base.DOC=DOC;base.run(sys.argv[2])
    elif sys.argv[1]=='worker':
        worker_impl.DOC=DOC;worker_impl.OUT=OUT;worker_impl.MATRIX=MATRIX
        worker_impl.__file__=__file__
        worker_impl.worker(int(sys.argv[2]))

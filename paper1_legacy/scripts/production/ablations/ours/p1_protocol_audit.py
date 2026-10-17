"""Read-only source/data audit; write protocol evidence, never submit training."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "docs/production/p1_paper_protocol_freeze"
P05 = ROOT / "outputs/preproduction/regression/p05_corrected_covariance_20260908"
DATA = ROOT.parent / "data"
TABLE = ROOT / "outputs/production/comparisons/us_recon_four_dataset_p100k_v1/tables"
SOURCES = ["gs_stage2_fixed_subset_per_run.tsv", "gs_full_sweep_per_run.tsv",
           "gs_ultragauss_tus_qc_chi2_5500_replacement.tsv", "implicit_full_sweep_per_run.tsv"]


def read(path):
    return list(csv.DictReader(path.open(), delimiter="\t"))


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {}
    for name in SOURCES:
        for r in read(TABLE / name):
            assert r['status'] == 'success' and int(r['final_step']) == 30000
            r['source_table'] = name
            rows[r['dataset'], r['sweep'], r['method']] = r
    methods = sorted({k[2] for k in rows})
    cases = sorted({k[:2] for k in rows})
    assert len(cases) == 78 and len(rows) == 624
    plans = json.loads((P05 / 'minimal_run_matrix.json').read_text())
    assert set(cases) == {(r['dataset'], r['sweep']) for r in plans if r['group'] == 'main'}
    multi = {(r['dataset'], r['sweep']) for r in read(ROOT / 'outputs/production/ablations/ours/summary.tsv') if r['table'] == 't06'}
    qual = {'synthetic_liver': ('l3', 27), 'spine_phantom': ('left3_2', 17),
            'tus_rec_2024_qc': ('010_RH_Per_S_PtD_qc_0165_0364_s02', 18),
            'tdsc_abus_2023': ('validation_116', 39)}
    frame_evidence, protocol, configs, mismatches = [], [], {}, []
    for ds, sw in cases:
        array = np.load(str(DATA / ds / sw / 'images.npy'), mmap_mode='r')
        n = len(array)
        case_rows = [rows[ds, sw, m] for m in methods]
        for r in case_rows:
            cfgfile = Path(r['output_dir']) / 'resolved_config.yaml'
            configs[ds, sw, r['method']] = yaml.safe_load(cfgfile.read_text())
        c = configs[ds, sw, 'ours']
        load = c['scene']['loading']
        trim = load['holdout_trim_each']
        nt = max(1, min(n - 2 * trim - 1, int(round(n * .2))))
        indices = np.rint(np.linspace(trim, n - 1 - trim, nt)).astype(int).tolist()
        for r in case_rows:
            cfg = configs[ds, sw, r['method']]
            if cfg['scene']['loading'] != load or cfg['metric'] != c['metric']:
                mismatches.append([ds, sw, r['method'], 'scene/metric config'])
            pf = Path(r['output_dir']) / 'eval/per_frame_metrics_iteration_030000.tsv'
            actual = [int(x['index']) for x in read(pf)] if pf.exists() else None
            if actual != indices:
                mismatches.append([ds, sw, r['method'], 'frame IDs', actual, indices])
            frame_evidence.append(dict(dataset=ds, sweep=sw, method=r['method'],
                                       source=r['output_dir'], source_table=r['source_table'],
                                       config_sha256=hashlib.sha256((Path(r['output_dir']) / 'resolved_config.yaml').read_bytes()).hexdigest(),
                                       test_indices=actual, expected_test_indices=indices,
                                       final_psnr=float(r['final_psnr']), final_step=r['final_step']))
        protocol.append(dict(dataset=ds, case_id=sw, include_in_main=1,
                             include_in_ablation=int((ds, sw) in multi),
                             include_in_qualitative=int(sw == qual[ds][0]), reason_if_excluded='',
                             baseline_available='all_7_final_30k', ours_available='legacy_30k;corrected_anchor' if (ds,sw) in [('synthetic_liver','l1'),('tus_rec_2024_qc','010_LH_Per_L_DtP_qc_0005_0370')] else 'legacy_30k;corrected_pending',
                             source_case_id=sw, num_frames=n, test_frames=nt, trim_each=trim,
                             qualitative_item=qual[ds][1] if sw == qual[ds][0] else ''))
    qc = json.loads((DATA / 'tus_rec_2024_qc/_qc_report.json').read_text())
    segments = read(DATA / 'tus_rec_2024_qc/_qc_segments.tsv')
    segmap = {r['output_sweep']: r for r in segments}
    for r in protocol:
        if r['dataset'] == 'tus_rec_2024_qc':
            r['source_case_id'] = segmap[r['case_id']]['source_sweep']
    for r in qc['sweep_summaries']:
        protocol.append(dict(dataset='tus_rec_2024_qc', case_id=r['source_sweep'], include_in_main=0,
                             include_in_ablation=0, include_in_qualitative=0,
                             reason_if_excluded='raw source; replaced by {} frozen geometric-QC segments'.format(r['segments']) if r['segments'] else 'no segment >=80 frames after frozen motion-QC, gap bridging and boundary trim',
                             baseline_available='not_evaluated_raw', ours_available='not_paper_protocol',
                             source_case_id=r['source_sweep'], num_frames=r['num_frames']))
    fields = list(protocol[0])
    with (OUT / 'paper_final_dataset_protocol.tsv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter='\t');writer.writeheader();writer.writerows(protocol)
    save('baseline_case_frame_audit.json', frame_evidence)
    save('protocol_mismatches.json', mismatches)
    save('legacy_main_selected_rows.json', list(rows.values()))
    save('arm_qc_provenance.json', qc)
    save('arm_qc_segments.json', segments)
    counts = Counter(ds for ds, sw in cases)
    local = {ds: sorted(p.name for p in (DATA/ds).iterdir() if (p/'images.npy').exists()) for ds in counts}
    assert all(set(local[ds]) == {sw for d,sw in cases if d==ds} for ds in counts)
    save('protocol_summary.json', dict(selected_counts=counts, local_processed_ids=local,
         raw_arm_sources=48, arm_sources_retained=sum(r['segments']>0 for r in qc['sweep_summaries']),
         arm_source_excluded=sum(r['segments']==0 for r in qc['sweep_summaries']),
         selected_multi_sweep=sorted(multi), qualitative=qual,
         exact_scene_and_metric_and_frame_agreement=not mismatches,
         baseline_methods=methods, caveat='Original external release totals are not local evaluated counts; raw backup /tmp/omniultra_raw_data_20260614_p100k is unavailable.'))
    print('counts', dict(counts), 'mismatches', len(mismatches), flush=True)
    assert not mismatches, 'Protocol blocker: inspect protocol_mismatches.json'


if __name__ == '__main__':
    main()

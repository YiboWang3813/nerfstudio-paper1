"""Match corrected FPS to exactly the historical two-case subset for every method."""
import csv
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'outputs/production/comparisons/paper_final_corrected_covariance_20260908/reports'


def read(path):
    return list(csv.DictReader(path.open(),delimiter='\t'))


def main():
    old=read(ROOT/'outputs/production/comparisons/us_recon_four_dataset_p100k_v1/tables/render_fps_complete_cached_ours_per_run.tsv')
    new=read(OUT/'corrected_main_fps.tsv')
    assert len(new)==78 and all(r['status']=='success' for r in new)
    rows=[]
    for ds in sorted({r['dataset'] for r in old}):
        chosen={r['sweep'] for r in old if r['dataset']==ds and r['method']=='ours'}
        assert len(chosen)==2
        for method in sorted({r['method'] for r in old}):
            source=new if method=='ours' else old
            available=[r for r in source if r['dataset']==ds and r['method']==method and r['sweep'] in chosen]
            assert len(available)==2 and {r['sweep'] for r in available}==chosen,(ds,method)
            assert all(r['status']=='success' for r in available)
            rows.append(dict(dataset=ds,method=method,num_runs=2,selected_ids=';'.join(sorted(chosen)),
                render_fps_mean=statistics.mean(float(r['render_fps']) for r in available),
                minimum_selected_case_fps=min(float(r['render_fps']) for r in available),
                provenance='corrected_4090' if method=='ours' else 'FINAL-REUSE_historical_4090'))
    with (OUT/'fair_fps_comparison.tsv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
    text=['# Fair FPS checkpoint\n','Same two cases per dataset for all methods; corrected FPS measured on a verified RTX4090, pure forward.\n',
          '|Dataset|Corrected mean FPS|Minimum selected-case FPS|\n|---|---:|---:|']
    for r in rows:
        if r['method']=='ours':text.append('|{}|{:.3f}|{:.3f}|'.format(r['dataset'],r['render_fps_mean'],r['minimum_selected_case_fps']))
    text.append('\nTraining remains4080SUPER except reused4090 anchors; do not infer hardware-controlled training speedup. See fair_fps_comparison.tsv for every baseline.\n')
    (OUT/'FAIR_FPS_CHECKPOINT.md').write_text('\n'.join(text))


if __name__=='__main__':
    main()

"""Index external data and existing paper evidence without rewriting sources."""
import ast
import hashlib
import json
import os
from pathlib import Path
import struct

W = Path(__file__).resolve().parents[2]
OLD = '/home/boywang25/ultrasound-scene-reconstruction/omniultra'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def npy_header(path):
    with path.open('rb') as f:
        assert f.read(6)==b'\x93NUMPY'
        version=tuple(f.read(2))
        n=struct.unpack('<H' if version[0]==1 else '<I',f.read(2 if version[0]==1 else 4))[0]
        return ast.literal_eval(f.read(n).decode('latin1'))


def relocated(path):
    prefix=OLD+'/outputs/'
    if path.startswith(prefix): return str(W/'results/legacy/outputs'/path[len(prefix):])
    return path


def main():
    target=W/'protocol/assets/data.jsonl'
    target.parent.mkdir(parents=True,exist_ok=True)
    count=0; total=0
    with target.open('w') as f:
        for base,dirs,files in os.walk(str(W/'data')):
            dirs.sort()
            for name in sorted(files):
                path=Path(base)/name
                st=path.stat(); rel=str(path.relative_to(W/'data'))
                row={'asset_id':'data:'+rel,'path':rel,'size_bytes':st.st_size,'mtime_ns':st.st_mtime_ns,
                     'sha256':sha(path),'device':st.st_dev,'inode':st.st_ino}
                if path.suffix=='.npy': row['npy_header']=npy_header(path)
                f.write(json.dumps(row)+'\n')
                count+=1; total+=st.st_size
                if count%100==0: print('DATA_HASHED',count,flush=True)
    docs=W/'docs/04_experiments/production'
    package=docs/'final_paper_evidence_20260910'
    manifest=json.loads((package/'audits/final_data_source_manifest.json').read_text())
    rows=[]
    for old in manifest['run_sources']:
        row=dict(old)
        row['original_output_dir']=old['output_dir']
        row['output_dir']=relocated(old['output_dir'])
        row['source_config']=relocated(old['source_config'])
        ckpt=Path(row['output_dir'])/'checkpoints'
        weights=sorted(str(p.resolve()) for p in ckpt.glob('*.pth')) if ckpt.is_dir() else []
        row['retained_checkpoint_files']=weights
        row['reproduction_capability']='checkpoint_present_unvalidated' if weights else 'metrics_and_existing_images_only'
        rows.append(row)
    result={'source_manifest':str(package/'audits/final_data_source_manifest.json'),
            'source_manifest_sha256':sha(package/'audits/final_data_source_manifest.json'),
            'source_rows':rows,'source_identity_policy':'Execution source SHA remains historical; migration/fork SHAs are separate identities.'}
    dst=W/'protocol/releases/legacy_paper_final.json';dst.parent.mkdir(parents=True,exist_ok=True)
    dst.write_text(json.dumps(result,indent=2)+'\n')
    figure=docs/'comparison_final_outputs/manifests/paper_all_datasets_2d_rendering_manifest.json'
    panels=[]
    for panel in json.loads(figure.read_text())['panels']:
        if 'source_image' not in panel: continue
        row=dict(panel);path=Path(relocated(panel['source_image']))
        row['original_source_image']=panel['source_image'];row['source_image']=str(path)
        row['sha256']=sha(path);panels.append(row)
    (W/'protocol/releases/legacy_qualitative_sources.json').write_text(json.dumps({'original_manifest':str(figure),'panels':panels},indent=2)+'\n')
    summary={'data_file_count':count,'data_bytes':total,'final_source_rows':len(rows),
             'rows_with_retained_checkpoints':sum(bool(r['retained_checkpoint_files']) for r in rows),
             'qualitative_source_images':len(panels)}
    (W/'results/migration/2026-10-17/asset_index_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__': main()

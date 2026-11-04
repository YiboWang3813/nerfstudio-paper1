"""Content-address retained weights and derived experimental arrays outside Git."""
import hashlib
import json
import os
from pathlib import Path

W=Path(__file__).resolve().parents[2]
DERIVED_EXTENSIONS={'.npy','.npz','.pkl','.pickle','.ply','.txt','.sha256','.sha1',
                    '.so','.tar','.gz','.zip','.mat','.h5','.hdf5','.nrrd','.mha','.mhd'}


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def index(root,target,select,kind):
    count=0;size=0
    with target.open('w') as out:
        for base,dirs,files in os.walk(str(root),followlinks=False):
            dirs.sort()
            for name in sorted(files):
                path=Path(base)/name
                if not path.is_file() or not select(path):continue
                st=path.stat();sha=digest(path)
                out.write(json.dumps({'asset_id':kind+':'+str(path.relative_to(root)),
                                      'content_id':'sha256:'+sha,'path':str(path.relative_to(root)),
                                      'size_bytes':st.st_size,'mtime_ns':st.st_mtime_ns})+'\n')
                count+=1;size+=st.st_size
                if count%100==0:print(kind,count,size,flush=True)
    return {'files':count,'bytes':size}


if __name__=='__main__':
    assets=W/'protocol/assets';assets.mkdir(exist_ok=True)
    report={}
    report['checkpoints']=index(W/'checkpoints',assets/'checkpoints.jsonl',lambda p:True,'checkpoint')
    report['derived_artifacts']=index(W/'results/legacy',assets/'derived_artifacts.jsonl',lambda p:p.suffix.lower() in DERIVED_EXTENSIONS,'derived')
    (W/'results/migration/2026-10-17/external_artifact_index_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

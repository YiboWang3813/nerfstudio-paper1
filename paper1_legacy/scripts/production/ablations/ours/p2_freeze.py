"""Complete reconstructable P2 source snapshot plus unchanged runtime binary."""
import difflib
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from scripts.production.ablations.ours.p0_snapshot import capture
from scripts.production.ablations.ours.p2_refresh import ROOT,DOC,FREEZE,sha


def main():
    from omegaconf import OmegaConf
    import xml.etree.ElementTree as ET
    reg=FREEZE.parent;xml=ET.parse(reg/'tests/pytest.xml').getroot()
    for suite in xml.iter('testsuite'):assert int(suite.get('failures',0))==0 and int(suite.get('errors',0))==0
    for ds in ('liver','arm'):assert json.loads((reg/'smoke'/ds/'smoke_stats.json').read_text())['success']
    capture(FREEZE)
    p05=ROOT/'outputs/preproduction/regression/p05_corrected_covariance_20260908/freeze_v2'
    old=json.loads((p05/'hashes.json').read_text());new=json.loads((FREEZE/'hashes.json').read_text());patch=[];production_changes=[]
    with tarfile.open(p05/'source.tar.gz') as archive:
        for name in new:
            if old.get(name)==new[name] or name.startswith(('scripts/','tests/','docs/')):continue
            production_changes.append(name)
            a=archive.extractfile(name).read().decode().splitlines(True);b=(ROOT/name).read_text().splitlines(True)
            patch.extend(difflib.unified_diff(a,b,'a/'+name,'b/'+name))
    assert set(production_changes)=={'configs_hydra/model/ours_us_focus.yaml','configs_hydra/experiment/production/ablations/ours/paper_semantic_freeze.yaml'},production_changes
    (FREEZE/'p2_production_only.patch').write_text(''.join(patch))
    hashes=json.loads((p05/'binary_hashes.json').read_text())
    for p,h in hashes.items():assert sha(ROOT/p)==h;shutil.copy2(ROOT/p,FREEZE/Path(p).name)
    (FREEZE/'binary_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    (FREEZE/'pip_freeze.txt').write_bytes(subprocess.check_output([sys.executable,'-m','pip','freeze']))
    (FREEZE/'conda_explicit.txt').write_bytes(subprocess.check_output(['/home/boywang25/miniconda3/bin/conda','list','--explicit','-p',str(Path(sys.executable).parents[1])]))
    (FREEZE/'python_version.txt').write_text(sys.version+'\n'+sys.executable+'\n')
    (FREEZE/'compiler.txt').write_bytes(subprocess.check_output(['gcc','--version']))
    for p in (ROOT/'logs/ours_ablation').glob('p2test_*'):shutil.copy2(p,FREEZE/p.name)
    shutil.copytree(reg/'tests',FREEZE/'tests');shutil.copytree(reg/'smoke',FREEZE/'smoke')
    compile_hashes={str(p.relative_to(ROOT)):sha(p) for p in (reg/'compile/lib').rglob('*.so')};assert compile_hashes
    (FREEZE/'compile_test_binary_hashes.json').write_text(json.dumps(compile_hashes,indent=2)+'\n')
    rows=json.loads((DOC/'final_refresh_matrix.json').read_text())
    for ds in ('synthetic_liver','spine_phantom','tus_rec_2024_qc','tdsc_abus_2023'):
        r=next(r for r in rows if r['dataset']==ds and 'main' in r['consumers'])
        shutil.copy2(DOC/'exact_configs'/(r['batch1_id']+'.yaml'),FREEZE/('main_'+ds+'.yaml'))
    r=next(r for r in rows if r['dataset']=='synthetic_liver' and 'ppp/PPP' in r['consumers'])
    shutil.copy2(DOC/'exact_configs'/(r['batch1_id']+'.yaml'),FREEZE/'production_PPP_representative.yaml')
    print('P2_SOURCE_SHA256', (FREEZE/'snapshot.sha256').read_text().strip())


if __name__=='__main__':main()

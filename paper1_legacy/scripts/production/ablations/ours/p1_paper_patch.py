"""Combine P0.5 semantic patch and audited P1 datasets without editing exchange tex."""
import difflib
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
DST=ROOT/'docs/production/p0_semantic_freeze'
SRC=ROOT/'docs/production/experiment_results_io'


def main():
    work=Path(tempfile.mkdtemp(prefix='p1-paper-',dir='/tmp'))
    for name in ('03_method.tex','04_experiments_and_results.tex'):
        shutil.copy2(SRC/name,work/name)
    subprocess.run(['git','apply',str(DST/'paper_alignment.patch')],cwd=str(work),check=True)
    path=work/'04_experiments_and_results.tex'
    text=path.read_text()
    start=text.index('We conduct experiments on two synthetic/phantom datasets')
    end=text.index('\\subsubsection{Baseline Methods}',start)
    replacement=r'''We evaluate six locally available Liver sweeps, nine Spine phantom sweeps,
33 quality-controlled Arm sweep segments, and 30 Breast volumes. Liver and Spine
originate from UltraNeRF~\cite{wysockiUltraNeRFNeuralRadiance2024}; our within-sweep
holdout protocol differs from that study's acquisition-level train/test split.
Arm derives from TUS-REC-2024~\cite{liTUSREC2024Challenge2025}: 48 local source sweeps
yield 33 continuous segments from 28 source sweeps after fixed pose-motion QC,
while 20 source sweeps yield no qualifying segment. QC is performed before
reconstruction, identically for all methods. Breast uses the 30 public validation
volumes (IDs 100--129) of TDSC-ABUS-2023~\cite{luoTDSCABUSChallenge2025}.
All methods are evaluated on exactly the same case and held-out frame IDs.
Processed frame sizes are $448\times224$, $896\times112$, $288\times384$, and
$280\times356$ for Liver, Spine, Arm, and Breast, respectively; frame counts
vary by case except for the 200-frame Liver sweeps. We use interval holdout with
approximately 20\% of each case's frames, restricting test eligibility by eight
boundary frames for Spine and ten otherwise. Boundary frames remain available
for training. Dataset metrics are arithmetic means of sweep-level final-30k
metrics, each itself a mean over held-out frames, without frame-count weighting.
The measured-pose main comparison is complemented by a separately identified
16-case MP-versus-PPP validation subset (six Liver, five Arm and five Breast).
For Arm, pose-based preprocessing remains a limitation: PPP does not use measured
probe geometry to construct its pseudo-pose trajectory, but this experiment does
not establish a wholly pose-free preprocessing pipeline.

'''
    path.write_text(text[:start]+replacement+text[end:])
    patch=[]
    for name in ('03_method.tex','04_experiments_and_results.tex'):
        patch.extend(difflib.unified_diff((SRC/name).read_text().splitlines(True),
                     (work/name).read_text().splitlines(True),'a/'+name,'b/'+name))
    target=DST/'paper_alignment_p1.patch'
    target.write_text(''.join(patch))
    subprocess.run(['git','apply','--check',str(target)],cwd=str(SRC),check=True)
    print('Prepared/check passed:',target)


if __name__=='__main__':
    main()

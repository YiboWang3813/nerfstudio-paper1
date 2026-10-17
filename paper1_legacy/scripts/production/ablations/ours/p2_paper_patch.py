"""Combined apply-ready semantic patch; never edit uncertain LaTeX include sources."""
import difflib
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
SRC=ROOT/'docs/production/experiment_results_io'
DST=ROOT/'docs/production/p0_semantic_freeze'


def main():
    names=('03_method.tex','04_experiments_and_results.tex')
    with tempfile.TemporaryDirectory(prefix='p2_paper_patch_') as tmp:
        for name in names:shutil.copy2(SRC/name,Path(tmp)/name)
        subprocess.run(['git','apply',str(DST/'paper_alignment_p1.patch')],cwd=tmp,check=True)
        method=(Path(tmp)/names[0]).read_text();results=(Path(tmp)/names[1]).read_text()
    start=method.index('To preserve spatial correspondence between pixels and scatterers')
    end=method.index('We first establish the probe-space representation',start)
    method=method[:start]+'''The local pathway forms echoes through two successive stages:
finite-slice elevational conditioning, followed by footprint-compensated
probe-plane echo aggregation. Together these operations account for finite
acoustic-slice overlap and image-plane Gaussian response.

'''+method[end:]
    start=method.index('On this probe-space representation, we introduce a depth-adaptive')
    end=method.index('probe-space Gaussian\n',start)
    method=method[:start]+r'''\paragraph{Stage 1: Finite-Slice Elevational Conditioning.}
We use an elevational kernel centered on the imaging plane, with one fixed
effective Gaussian slice-width parameter shared across datasets and frames:
\begin{equation}
\sigma^{\mathrm{ele}}_{ig}=\sigma_{\mathrm{ele}}=1.0\,\mathrm{mm}.
\label{eq:fixed-ele-width}
\end{equation}
This parameter controls overlap with the finite acoustic slice; no depth-growth
term or per-sequence width tuning is used in the final method. Multiplying this
kernel by the
'''+method[end:]
    target='With the conditioned mean $\\boldsymbol\\mu^{e}_{ig}$ and covariance'
    assert target in method
    method=method.replace(target,'\\paragraph{Stage 2: Footprint-Compensated Probe-Plane Echo Aggregation.}\n'+target,1)
    start=results.index('Table~\\ref{tab:ablation:renderer} compares')
    end=results.index('\\input{tables/main/tab_ablation_gaussian_adaptation}',start)
    results=results[:start]+r'''Table~\ref{tab:ablation:renderer} compares Plane, Fixed, Adaptive,
Fixed+Simple and Fixed+Full using the same six Liver and five frozen Arm sweeps.
Plane-to-Fixed tests finite-slice conditioning. Fixed versus Adaptive compares
alternative local formulations, not progressive additions: the depth-adaptive
alternative uses $\operatorname{clip}(1+0.01z,0.8,5)$ mm and exhibits
dataset-dependent reconstruction/perceptual trade-offs. It is not the final
production formulation. Fixed-to-Fixed+Simple-to-Fixed+Full examines preceding-path
modulation and explicit causal ray integration. These rows are not assumed to
improve monotonically on every metric.
Simple Global uses the equivalent Gaussian scale
$s_g=\sqrt{(s_{gx}^2+s_{gy}^2+s_{gz}^2)/3}$ and
$\rho^{\mathrm{simple}}_{igp}=s_g\exp[-(d_{\mathrm{lat}}^2+d_{\mathrm{ele}}^2)/(2s_g^2)]$,
with $T_{igp}=\tau_g+(1-\tau_g)\exp(-\lambda_{\mathrm{ray}}\rho^{\mathrm{simple}}_{igp})$
for preceding-depth candidates. It is not pixel-depth-only decay.
Global-off rows use local-aware pruning; global-on rows use path-aware pruning.
No separate acoustic-attribute-decoupling ablation is claimed.
% Import final P2 table values from the manifest-backed reports, not old text.

'''+results[end:]
    results='% P2 SEMANTIC PATCH: import all final Fixed numerical tables/figures from P2 reports before publication; remaining historical numerical prose is not final evidence.\n'+results
    delta=[]
    for name,new in zip(names,(method,results)):
        old=(SRC/name).read_text();delta.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),'a/'+name,'b/'+name))
    (DST/'paper_alignment_p2.patch').write_text(''.join(delta))
    print('P2 combined semantic patch generated; canonical LaTeX untouched')


if __name__=='__main__':main()

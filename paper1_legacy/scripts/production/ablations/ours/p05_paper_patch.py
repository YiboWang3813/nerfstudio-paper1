"""Prepare a checked semantic patch; never edit the uncertain exchange copies."""
import difflib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "docs/production/experiment_results_io"
DST = ROOT / "docs/production/p0_semantic_freeze"


def main():
    method = (SRC / "03_method.tex").read_text()
    changes = [
        ("coherent scan geometry without measured probe poses", "coherent scan geometry without using measured probe geometry to construct the pseudo-pose trajectory"),
        ("$\\mathcal T=\\{\\mathbf T_i^{p\\rightarrow w}\\}_{i=1}^{N}$ is unavailable.", "$\\mathcal T=\\{\\mathbf T_i^{p\\rightarrow w}\\}_{i=1}^{N}$ is not used to construct the pseudo-pose trajectory."),
        ("$d_0>0$ is the nominal acquisition spacing", "$d_0>0$ is the dataset-specific nominal/calibrated acquisition spacing stored in DatasetMeta"),
        ("frame index:\n", "retained frame index. The loader may read the pose file for frame bookkeeping and consistency processing; measured probe geometry is not used to construct the pseudo-pose trajectory:\n"),
        ("\\eta_{ig}\\nu_{ig}\\phi_{igp}\n>\n\\tau_{\\mathrm{loc}}", "\\alpha_g\\eta_{ig}\\nu_{ig}\\phi_{igp}\n\\geq\n\\tau_{\\mathrm{loc}}"),
        (r"\operatorname*{arg\,min}_{0\leq t\leq t_p}", r"\operatorname*{arg\,min}_{t\in\mathbb R}"),
        ("and $t_s,t_e$ the clipped endpoints. Integrating over this interval yields", "and $t_s,t_e$ the clipped endpoints. The interaction center is first found on the infinite line; the overlap is zero if $t_e\\leq t_s$. Otherwise, integrating over this interval yields"),
        ("we accumulate the local contribution", "on the adaptation-event frame, we compute the local contribution"),
        (r"\frac{R_g}{C_g+\varepsilon}", r"\frac{R_g}{\max(C_g,10^{-8})}"),
        (r"Q_{q_{\mathrm d}}(\{\omega\})", r"Q_{q_{\mathrm d}}(\{\omega_g\mid\omega_g>0\})"),
        ("Gaussians to subdivide broad support, whereas smaller candidates are cloned to", "Gaussians: the parent is retained and two children are appended, with each child's raw scale parameter reduced by $\\log 2$. This does not exactly halve the bounded-sigmoid physical scale. Smaller candidates are cloned to"),
        ("global transmission, we jointly evaluate their accumulated local contribution", "global transmission, we jointly evaluate their adaptation-event-frame local contribution"),
        (r"C_g<Q_{q_{\mathrm p}}", r"C_g\leq Q_{q_{\mathrm p}}"),
        ("global-contribution pruning levels, respectively.", "global-contribution pruning levels, respectively. When the global pathway is unavailable, path-aware pruning degrades to local-aware pruning by omitting the global-protection condition."),
    ]
    updated = method
    for old, new in changes:
        assert old in updated, old
        updated = updated.replace(old, new)
    results = (SRC / "04_experiments_and_results.tex").read_text()
    revised = results.replace("obtain $d_0$ from acquisition metadata", "use dataset-specific nominal/calibrated acquisition spacing $d_0$ stored in DatasetMeta")
    revised = revised.replace("contains 8 robotic sweeps", "contains 9 protocol-selected sweeps")
    revised = revised.replace("all 6 Liver and 8 Spine", "all 6 Liver and 9 Spine").replace("48 Arm sweeps", "33 Arm sweeps")
    start = revised.index("We examine how reconstruction changes as measured pose information")
    end = revised.index("\\input{tables/main/tab_ablation_local_global_renderer}", start)
    revised = revised[:start] + r"""We compare measured 6-DoF poses (MP), pseudo poses built from cumulative
Euclidean adjacent measured translation magnitudes (PP-S), cumulative signed
measured translation projections on the predefined axis (PP-A), and acquisition-
guided uniform pseudo poses (PPP). PP-S and PP-A discard measured rotations;
neither fits the scan axis. PPP uses only retained frame indices, nominal/calibrated
spacing, a predefined signed axis and a reference frame to construct its trajectory.
Pose-file loading for bookkeeping is distinct from using measured probe geometry.
% P0.5: insert corrected PPP and multi-sweep results after authorized reruns.

""" + revised[end:]
    revised = "% P0.5 SEMANTIC PATCH ONLY: all remaining legacy Ours numerical claims/tables/figures must be replaced before final publication.\n" + revised
    start = revised.index("Table~\\ref{tab:ablation:renderer} traces")
    end = revised.index("\\input{tables/main/tab_ablation_gaussian_adaptation}", start)
    revised = revised[:start] + r"""Table~\ref{tab:ablation:renderer} compares Plane Local, Fixed Slice
(constant 1-mm elevational sensitivity), Adaptive Slice, Simple Global, and
Full LGR. Plane-to-Fixed isolates finite elevational conditioning;
Fixed-to-Adaptive isolates depth dependence. Simple Global uses
$s_g=\sqrt{(s_{gx}^2+s_{gy}^2+s_{gz}^2)/3}$ and
$\rho^{\mathrm{simple}}_{igp}=s_g\exp[-(d_{\mathrm{lat}}^2+d_{\mathrm{ele}}^2)/(2s_g^2)]$,
with $T_{igp}=\tau_g+(1-\tau_g)\exp(-\lambda_{\mathrm{ray}}\rho^{\mathrm{simple}}_{igp})$
for preceding-depth candidates. Full LGR instead uses causal ray integration.
Global-off rows use local-aware pruning; global-on rows use path-aware pruning.
No separate attribute-decoupling ablation is claimed.
% P0.5: insert corrected five-row results only after their training is authorized.

""" + revised[end:]
    patch = []
    for name, old, new in (("03_method.tex", method, updated), ("04_experiments_and_results.tex", results, revised)):
        patch.extend(difflib.unified_diff(old.splitlines(True), new.splitlines(True), "a/" + name, "b/" + name))
    (DST / "paper_alignment.patch").write_text("".join(patch))
    print("Prepared patch against exchanged source text; apply only after canonical-path/content verification.")


if __name__ == "__main__":
    main()

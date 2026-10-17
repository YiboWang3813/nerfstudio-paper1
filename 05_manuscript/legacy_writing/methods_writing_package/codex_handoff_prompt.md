# Handoff Prompt For Local Codex

You are writing the Methods section for an offline one-sweep ultrasound
Gaussian reconstruction paper. Read this package first:

```text
docs/paper/methods_writing_package/README.md
docs/paper/methods_writing_package/ours_methods_source.md
docs/paper/methods_writing_package/code_formula_map.md
docs/paper/methods_writing_package/methods_draft.md
```

Then inspect implementation snapshots under:

```text
docs/paper/methods_writing_package/source_snapshot/
```

## Source Of Truth

Use `ours_methods_source.md` as the source of truth. It resolves conflicts in
older notes. In particular, the final production renderer uses:

```text
local us_focus branch
ray-CSR integral attenuation branch
quadrature_points = 3
beta = 0.80
strength = 0.55
min_transmission = 0.35
candidate_k_sigma = 2.0
```

Do not describe `slice_depth` as the final production renderer. It was a
development/ablation variant.

## Required Methods Organization

Write the Methods in this order:

1. Problem formulation and Gaussian scene representation.
2. Local-global ultrasound renderer.
3. Responsibility-aware density control.
4. Easy-Pose canonical sweep prior.
5. Training objective and implementation details.

## Claims To Preserve

Use this framing:

```text
Local-global renderer is the dominant reconstruction-quality contribution.
Responsibility DC is a topology/capacity allocation mechanism and is
performance-neutral to weakly positive.
Easy-Pose reduces pose dependency under regular or near-regular sweep
assumptions.
```

## Claims To Avoid

Do not claim:

```text
universal pose-free reconstruction
global-only renderer
responsibility DC as the main PSNR booster
max-hit as part of final responsibility score
cross-method 3D SuperSplat figures as fair qualitative evidence
```

## Style

Keep Methods technical and code-aligned. Use equations where they clarify the
pipeline. Avoid overexplaining historical failed ideas in the main Methods
section. Put implementation-specific details such as quaternion layout and CUDA
candidate construction in concise paragraphs or appendix-style text.

## Deliverable

Produce a manuscript-ready Methods draft in Markdown or LaTeX-like Markdown.
Include equations for:

```text
Gaussian covariance
local renderer L_ip
global modulation M_ip
responsibility score
canonical sweep pose
training objective
```


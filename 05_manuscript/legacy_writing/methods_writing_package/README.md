# Ours Methods Writing Package

This directory is intentionally ignored by git. It is a local handoff package
for writing the paper Methods section with another Codex session or a local
writing agent.

## Read Order

1. `ours_methods_source.md`
2. `code_formula_map.md`
3. `methods_draft.md`
4. `codex_handoff_prompt.md`

The bundled source code and evidence snapshots live under:

```text
source_snapshot/
paper_assets/
```

Use `source_snapshot/` as a local reference copy. The original repo files remain
the authoritative implementation, but this snapshot is packaged so a local
Codex can inspect the relevant code without re-discovering the whole project.

## Final Methods Position

The paper-facing method should be written as:

```text
Ours = local-global ultrasound renderer
     + responsibility-aware density control
     + easy-pose canonical sweep prior
```

For the main multi-method comparison row, Ours uses file poses because the
baselines require the same pose input class. Easy-Pose is a method module and
pose-dependency experiment, not the main comparison row.

## Important Version Note

Some older method notes still describe the final ray branch as `slice_depth`.
The latest production Ours row and expanded ablation package use:

```text
ray_attenuation_formula = integral
ray_attenuation_quadrature_points = 3
ray_attenuation_candidate_k_sigma = 2.0
ray_attenuation_beta = 0.80
ray_attenuation_strength = 0.55
ray_attenuation_min_transmission = 0.35
```

Use the latest production configuration above for Methods writing.


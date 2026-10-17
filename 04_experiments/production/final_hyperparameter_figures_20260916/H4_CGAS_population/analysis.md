# H4 — CGAS quality–population relationship

Authoritative sources: `cga/hyperparameter_data.tsv` and `cga/structural_sources.tsv` under the sealed final evidence package; independently matched against `cga/quality_population.tsv`. Plot each physical output run once, merging repeated default memberships by output path. Original memberships/IDs are in plotted_data.tsv. No point jitter or regression line.

KEEP. The unpruned RGD cases (1,050,672 Liver / 1,022,586 Arm) extend far beyond the sensitivity scans and add information absent from H3. Sensitivity points use hollow circles; structural strategies use diamonds; the production Full CGAS default uses a star. Separate vertically aligned Liver/Arm axes retain actual PSNR units and show narrow within-dataset changes without a ~3.4-dB inter-dataset gap flattening both clouds. The x-axis is shared; y-scales are explicitly dataset-specific.

Population changes substantially while PSNR stays within 0.0277768 dB for Liver and 0.1160482 dB for Arm across the combined evidence. This is an observed quality–capacity trade-off, not a monotonic causal law, efficiency ratio, or argument that fewer primitives are always better. GD vs unpruned RGD changes pruning as well as densification. No fitted lines. Overlapping points remain overlapping: multiplicity is not invented by offsets.

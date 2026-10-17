# Eight authorized Fixed FPS model reconstructions

Approved explicitly after the user was informed that the original Fixed models
had been deleted. These8 runs reconstruct models only for the frozen FPS cases;
they do not replace any of the78 original quality observations or select better
replicas. Same P0.5 source, exact Fixed+Full/MP/FullCGA/1M/30k/seed42 config.

Training jobs80170(gpu2,2x4080SUPER) and80171(gpu3,2x4090); afterok FPS job80172
uses4090. Four disjoint worker bins. No production-default/source edits occur
until these jobs complete. Baseline case choices remain exactly2 per dataset.

Benchmark:5warmup,20timed heldout frames,3passes,pureforward. Average the two
case FPS values within a dataset, then equal dataset means for Liver+Spine and
Arm+Breast panels. Record every raw value and replica-vs-original quality/#G
delta; do not substitute reconstructed metrics into the paper quality table.

The8 final models are retained through P2 and final-efficiency review. No cleanup
script is scheduled for these runs. All configs/logs/manifests remain available.

Submission archive SHA256:
`9edf26b606df540f2eeb64c763d0e51e3ab322871d0136fd9f4857786f3f17ff`.
Production source SHA256:
`b2525342593c8dd84a1f1ddb8c1ac124c9b1fb179ddcffea3fb902eb2a073ad2`.

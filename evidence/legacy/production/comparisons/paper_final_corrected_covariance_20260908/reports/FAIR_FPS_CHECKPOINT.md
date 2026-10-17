# Fair FPS checkpoint

Same two cases per dataset for all methods; corrected FPS measured on a verified RTX4090, pure forward.

|Dataset|Corrected mean FPS|Minimum selected-case FPS|
|---|---:|---:|
|spine_phantom|229.263|221.055|
|synthetic_liver|240.161|236.578|
|tdsc_abus_2023|316.247|313.735|
|tus_rec_2024_qc|293.142|271.621|

Training remains4080SUPER except reused4090 anchors; do not infer hardware-controlled training speedup. See fair_fps_comparison.tsv for every baseline.

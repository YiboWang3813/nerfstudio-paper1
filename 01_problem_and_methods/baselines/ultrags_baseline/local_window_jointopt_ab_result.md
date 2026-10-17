# Local Window Joint Optimization A/B Result (2026-05-08)

## Purpose

Validate whether seam/joint optimization at local-window boundaries should be enabled by default.

- A: `joint_opt.enable=false`
- B: `joint_opt.enable=true`
- Shared setup:
  - local-window + batch training
  - `batch_size=4`
  - `iterations=10000`
  - `num_init=1000000`
  - density control disabled
  - evaluation every `2500` iterations

Run entry:

- Historical experiment-script and legacy-config entries were removed during script/config reorganization. Inspect git history for the exact legacy launcher, or reproduce with current Hydra aliases plus equivalent overrides.

## Final Runtime

- A (no joint): `189.94s`
- B (with joint): `235.42s`
- B/A speed: `0.8068x` (about 24% slower with joint optimization)

## Test Metrics

`Iter 2500`

- A: `psnr=26.6687`, `ssim=0.8343`, `lpips=0.2060`, `dists=0.2378`
- B: `psnr=26.6427`, `ssim=0.8364`, `lpips=0.2045`, `dists=0.2387`

`Iter 5000`

- A: `psnr=33.3545`, `ssim=0.8712`, `lpips=0.1284`, `dists=0.1811`
- B: `psnr=33.3541`, `ssim=0.8703`, `lpips=0.1300`, `dists=0.1824`

`Iter 7500`

- A: `psnr=34.3017`, `ssim=0.8764`, `lpips=0.1126`, `dists=0.1554`
- B: `psnr=34.3087`, `ssim=0.8766`, `lpips=0.1109`, `dists=0.1545`

`Iter 10000` (final)

- A: `l1=0.0133`, `psnr=34.5709`, `ssim=0.8801`, `lpips=0.1035`, `dists=0.1411`, `gmsd=0.0426`
- B: `l1=0.0134`, `psnr=34.5407`, `ssim=0.8794`, `lpips=0.1054`, `dists=0.1418`, `gmsd=0.0427`

## Conclusion

- Test accuracy is effectively tied; no consistent gain from enabling joint optimization.
- Joint optimization introduces substantial runtime overhead (~24%).
- Recommended default for current offline local-window GS training:
  - `joint_opt.enable=false`
  - keep seam optimization as an optional switch for future boundary-specific studies.

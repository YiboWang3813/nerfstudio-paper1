# IEEE single-column sensitivity figure contract

The user's 2026-09-18 specification controls all visual choices. Backend:
Python/Matplotlib. Archetype: quantitative trend (H1/H2) and quantitative grid
(H3). Existing sealed representative-case data only, one seed per setting.

- H1 question: how does nominal AFGM spacing scale affect reconstruction PSNR
  on Liver and Arm? One panel; two dataset traces of default-relative PSNR.
  LPIPS remains in the data inventory and prose, not on an incompatible shared
  y-axis. Seven sampled rho values; reference rho=1.
- H2 question: how do five numerical PLGR settings change PSNR around the
  fixed final operating point? Five aligned rows ordered sigma_ele, beta,
  lambda_ray, Nq, half-window. Each row contains Liver and Arm traces only.
- H3 question: do CGAS selection settings vary population while preserving
  quality? Left grid qd × B, right grid qp × qA, lower tau_alpha strip.
  Hue = default-relative PSNR on one symmetric shared scale; text = final #G(k).
  Each source table has *two* datasets, so the grids display the complete
  Liver three-row block above the Arm three-row block; tau has one strip row
  per dataset. This avoids averaging or silently dropping a dataset.

Journal contract: exactly 3.50-inch wide white canvases; H1 1.40-inch high,
H2 3.80-inch high, H3 2.10-inch high. Font DejaVu Sans and DejaVu Sans mathtext;
closed 0.6-point frames; exact user palette; PDF vector and PNG 600 dpi.
Every rendered glyph at least 5 pt, with text editable. Run physical alignment
gate on H2/H3 before export, then PDF glyph and collision audits. No invented
uncertainty: every sensitivity value is one completed 30k run.

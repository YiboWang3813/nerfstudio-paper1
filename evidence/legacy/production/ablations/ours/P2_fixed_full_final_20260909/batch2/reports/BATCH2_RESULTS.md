# Final Batch2 numeric sensitivity (no production retuning)

66 newly trained configs + 2 exact reused defaults; 82 panel slots. MP, Fixed+Full, accumulated-A_g CGA, same P2 source and metrics. B uses .001/.005/.05; sigma uses .8/1/1.5/2 mm to respect the existing .8 mm clamp. No significance testing.

| dataset | panel | n | default_point | psnr_min | psnr_max | psnr_range | psnr_default | ssim_min | ssim_max | ssim_range | ssim_default | lpips_min | lpips_max | lpips_range | lpips_default | G_min | G_max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| synthetic_liver | sigma_ele | 4 | 1.0 | 29.2498 | 29.8371 | 0.587318 | 29.4522 | 0.651978 | 0.67603 | 0.0240518 | 0.660275 | 0.152107 | 0.187124 | 0.0350173 | 0.157044 | 591920 | 724160 |
| tus_rec_2024_qc | sigma_ele | 4 | 1.0 | 32.0909 | 32.8614 | 0.770497 | 32.8167 | 0.8547 | 0.866248 | 0.0115475 | 0.865764 | 0.109099 | 0.130615 | 0.0215161 | 0.11204 | 614276 | 742257 |
| synthetic_liver | beta | 4 | 0.8 | 29.4468 | 29.5381 | 0.0912602 | 29.4522 | 0.65746 | 0.665149 | 0.00768936 | 0.660275 | 0.148773 | 0.16573 | 0.0169571 | 0.157044 | 683014 | 704516 |
| tus_rec_2024_qc | beta | 4 | 0.8 | 32.6617 | 32.9188 | 0.257008 | 32.8167 | 0.865037 | 0.866548 | 0.0015107 | 0.865764 | 0.104291 | 0.118836 | 0.0145447 | 0.11204 | 704465 | 731788 |
| synthetic_liver | lambda_ray | 5 | 0.55 | 29.4423 | 29.4595 | 0.0171847 | 29.4522 | 0.660189 | 0.660629 | 0.0004405 | 0.660275 | 0.157044 | 0.158933 | 0.00188873 | 0.157044 | 687324 | 688797 |
| tus_rec_2024_qc | lambda_ray | 5 | 0.55 | 32.7697 | 32.8238 | 0.0541477 | 32.8167 | 0.865587 | 0.865827 | 0.000240351 | 0.865764 | 0.11102 | 0.11204 | 0.00102001 | 0.11204 | 711157 | 713263 |
| synthetic_liver | Nq | 3 | 3 | 29.0425 | 29.4522 | 0.409655 | 29.4522 | 0.653721 | 0.660401 | 0.00667985 | 0.660275 | 0.157044 | 0.165619 | 0.00857464 | 0.157044 | 641666 | 688611 |
| tus_rec_2024_qc | Nq | 3 | 3 | 32.0386 | 32.8167 | 0.778078 | 32.8167 | 0.860396 | 0.865764 | 0.00536848 | 0.865764 | 0.11176 | 0.115322 | 0.00356132 | 0.11204 | 655788 | 713263 |
| synthetic_liver | half_window | 4 | 2.0 | 29.4315 | 29.4522 | 0.0206915 | 29.4522 | 0.660091 | 0.660437 | 0.000345877 | 0.660275 | 0.157044 | 0.158291 | 0.00124723 | 0.157044 | 686604 | 690898 |
| tus_rec_2024_qc | half_window | 4 | 2.0 | 32.7942 | 32.8187 | 0.0244875 | 32.8167 | 0.865527 | 0.865787 | 0.00025956 | 0.865764 | 0.11048 | 0.11204 | 0.00155951 | 0.11204 | 712137 | 713677 |
| synthetic_liver | qd_by_B | 9 | 0.98_0.05 | 29.4416 | 29.4561 | 0.0144313 | 29.4522 | 0.660275 | 0.660822 | 0.000546785 | 0.660275 | 0.156864 | 0.158069 | 0.00120527 | 0.157044 | 675559 | 711076 |
| tus_rec_2024_qc | qd_by_B | 9 | 0.98_0.05 | 32.8107 | 32.8238 | 0.013171 | 32.8167 | 0.865679 | 0.865805 | 0.000126305 | 0.865764 | 0.110903 | 0.11204 | 0.00113702 | 0.11204 | 707735 | 724242 |
| synthetic_liver | qp_by_qA | 9 | 0.1_0.8 | 29.4429 | 29.4573 | 0.0144032 | 29.4522 | 0.660199 | 0.660614 | 0.000414969 | 0.660275 | 0.156913 | 0.15849 | 0.00157637 | 0.157044 | 661306 | 722969 |
| tus_rec_2024_qc | qp_by_qA | 9 | 0.1_0.8 | 32.8142 | 32.8233 | 0.0090822 | 32.8167 | 0.865668 | 0.865819 | 0.000151588 | 0.865764 | 0.110967 | 0.11204 | 0.00107263 | 0.11204 | 701034 | 728740 |
| synthetic_liver | tau_alpha | 3 | 0.0001 | 29.4462 | 29.4622 | 0.0159422 | 29.4522 | 0.660275 | 0.660851 | 0.000576574 | 0.660275 | 0.157011 | 0.159315 | 0.00230338 | 0.157044 | 565867 | 732376 |
| tus_rec_2024_qc | tau_alpha | 3 | 0.0001 | 32.8128 | 32.8246 | 0.0117541 | 32.8167 | 0.865607 | 0.865832 | 0.000224872 | 0.865764 | 0.111071 | 0.112183 | 0.00111259 | 0.11204 | 639733 | 738377 |


Inspect raw LPIPS companion with delta-PSNR; do not infer a universal optimum or broad plateau from a small grid. Check adaptation_event_statistics.tsv for inactive budgets and zero local-contribution thresholds. Wall time is shared-worker engineering metadata, NOT a controlled FPS comparison. No new benchmark per point.

All panels retain raw PSNR/SSIM/LPIPS and default-relative deltas. H3 additionally retains population and raw event statistics. Figures are drafts, not approved paper layouts. PPP H1 is reused unchanged.

No further training is planned or automatically submitted. Final scientific interpretation and qualitative audit must be reviewed before marking the paper evidence complete.

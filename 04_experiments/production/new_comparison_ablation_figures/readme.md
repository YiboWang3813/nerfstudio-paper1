最终正文建议只保留 3 张 Results 图：

  1. 四数据集、全部方法的一体化定性对比图
  2. 三个创新点逐步加入的紧凑定性消融图
  3. 三个创新点的一体化参数敏感性图

  现有图片的合并关系是：

  - paper_simulated_2d_rendering + paper_real_2d_rendering → 新图 1
  - paper_ablation_joint_visual → 精简为新图 2
  - 三张独立 sensitivity 图 → 合并为新图 3
  - paper_ablation_internal_visual 不再单独进入正文，其有无性证据由三个消融表承担

  下面这段可以直接发给服务器 Codex。

  ———

  ## 给服务器 Codex 的完整任务说明

  请为 TMI 版本重新制作 Results 中的三张最终图片。不要修改实验结果，不要重新训练，只基于现有 production outputs、manifest 和可视化结果重新排版。

  现有 frozen 脚本和图片必须保留。由于原配置已标记 frozen，请新建 TMI v2 脚本、配置文件和输出，不要直接覆盖旧版本。

  ### 全局规范

  - 最终图片均用于 IEEE TMI 双栏排版，宽度按 \textwidth，约 6.9 inch。
  - 同时输出矢量 PDF 和 300 DPI PNG。
  - PDF 中文字应保持矢量字体，设置：

    pdf.fonttype = 42
    ps.fonttype = 42

  - 字体使用 Times New Roman、Times 或 DejaVu Serif。
  - 最终插入 6.9 inch 宽度后，任何文字不得小于约 7 pt。
  - 超声图像保持灰度显示。
  - 同一行的所有方法必须使用完全相同的显示范围、图像尺寸、crop 和 inset 放大倍率。
  - 禁止对不同方法分别进行 min–max normalization。
  - 黄色框建议使用 #FFD54F，最终线宽约 0.7–0.9 pt。
  - 所有方法名称严格写成：

    UltraNeRF
    NeRF-US
    UltraGauss
    UltraGRay
    UltraGS
    3DGS
    GaussianPile
    ULGSplat
    GT

  - 数据集名称严格写成：

    Synthetic Liver
    Spine Phantom
    TUS-REC-2024
    TDSC-ABUS-2023

  - 不得再出现以下旧名称：

    UltraGray
    Slice-3DGS
    Ours
    TUS-QC
    Easy-Pose
    DC
    EP

  - 每张图都要生成 JSON manifest，记录每个 panel 的：
      - dataset
      - sweep
      - item
      - method/variant
      - run directory
      - source image
      - crop coordinates

  - 不要修改论文 LaTeX，只生成图片、脚本、配置和 manifest。

  ———

  ## 图 1：四数据集整体定性对比图

  ### 输出名称

  paper_all_datasets_2d_rendering.pdf
  paper_all_datasets_2d_rendering.png
  paper_all_datasets_2d_rendering_manifest.json

  建议新脚本：

  scripts/production/comparisons/us_recon_four_dataset_p100k_v1/
  make_tmi_all_datasets_2d_rendering_v2.py

  建议新配置：

  docs/production/comparison_final_outputs/
  2d_rendering_figure_config_tmi_v2.md

  ### 版式

  采用严格的 4 rows × 9 columns：

                   UltraNeRF  NeRF-US  UltraGauss  UltraGRay  UltraGS  3DGS  GaussianPile  ULGSplat  GT
  Synthetic Liver
  Spine Phantom
  TUS-REC-2024
  TDSC-ABUS-2023

  列顺序不能改变。

  内部 method id 对应关系：

  UltraNeRF    -> ultranerf
  NeRF-US      -> nerfus_no_prior
  UltraGauss   -> ultragauss
  UltraGRay    -> ultragray
  UltraGS      -> ultrags_baseline
  3DGS         -> slice_adapted_3dgs
  GaussianPile -> gaussianpile
  ULGSplat     -> ours
  GT           -> gt

  ### 固定样例

  每个数据集只保留一个代表性样例：

  Synthetic Liver
  dataset: synthetic_liver
  sweep: l3
  item: 27
  crop_xywh: [19, 0, 53, 107]

  Spine Phantom
  dataset: spine_phantom
  sweep: left3_2
  item: 17
  crop_xywh: [59, 0, 36, 215]

  TUS-REC-2024
  dataset: tus_rec_2024_qc
  sweep: 010_RH_Per_S_PtD_qc_0165_0364_s02
  item: 18
  crop_xywh: [129, 24, 92, 69]

  TDSC-ABUS-2023
  dataset: tdsc_abus_2023
  sweep: validation_116
  item: 39
  crop_xywh: [150, 23, 85, 67]

  这些样例和 crop 已经冻结，不要再次根据方法性能挑选新帧。

  ### Panel 样式

  每个 panel：

  1. 显示完整预测图。
  2. 在固定区域画黄色 source rectangle。
  3. 放置同倍率的放大 inset。
  4. 用两根黄色连线连接 crop 与 inset。
  5. inset 的位置通过四角候选位置选择，但同一行所有方法必须使用相同位置。
  6. GT 也使用完全相同的 crop 和 inset。
  7. 不再在图内叠加 PSNR/SSIM。九列情况下数值会过小，而且定量结果已经由 Table I 完整报告。
  8. 不在图内写 sweep id；左侧只显示完整数据集名称。

  ### 尺寸建议

  figsize = (18.0, 8.0)
  dpi = 300
  wspace = 0.02–0.03
  hspace = 0.05–0.07

  源图中的字体可设置得较大，以保证缩放到 6.9 inch 后仍有约 7–8 pt：

  method headers: 18–19 pt at source size
  dataset labels: 17–18 pt at source size

  尽量减少顶部、底部和左右空白，最终宽高比建议约为 2.2–2.4，插入论文后的高度控制在约 2.9–3.2 inch。

  ### 建议 caption

  Qualitative comparison on four ultrasound datasets. Rows correspond to
  Synthetic Liver, Spine Phantom, TUS-REC-2024, and TDSC-ABUS-2023, while
  columns show all baseline methods, ULGSplat, and the ground truth. Yellow
  boxes indicate fixed regions shown at a common magnification. All methods in
  each row are evaluated on the same held-out slice using identical display
  ranges and crop coordinates.

  ———

  ## 图 2：三个创新点的紧凑定性消融图

  基于现有：

  paper_ablation_joint_visual
  paper_ablation_joint_visual_manifest.json

  但不能继续使用目前的四行 Prediction/Error 设计，它太高。最终只保留预测结果，改成 2 rows × 5 columns。

  ### 输出名称

  paper_ablation_joint_visual_compact.pdf
  paper_ablation_joint_visual_compact.png
  paper_ablation_joint_visual_compact_manifest.json

  ### 版式

                   Base   +LGR   +LGR+CGA   +LGR+CGA+PP   GT
  Synthetic Liver
  TUS-REC-2024

  缩写含义：

  LGR = physics-aware local-global Gaussian renderer
  CGA = contribution-aware Gaussian adaptation
  PP  = acquisition-guided pseudo-pose parameterization

  不要继续使用 LG/DC/EP。

  ### Variant 与 run 的映射

  Synthetic Liver：

  dataset: synthetic_liver
  sweep: l1
  item: 23
  crop_xywh: [0, 0, 53, 107]

  Base:
  runs/t02/synthetic_liver/l1/local_gradient

  +LGR:
  runs/t02/synthetic_liver/l1/localglobal_gradient

  +LGR+CGA:
  runs/t02/synthetic_liver/l1/localglobal_responsibility

  +LGR+CGA+PP:
  runs/t06/synthetic_liver/l1/canonical_sweep

  TUS-REC-2024：

  dataset: tus_rec_2024_qc
  sweep: 010_LH_Per_L_DtP_qc_0005_0370
  item: 9
  crop_xywh: [129, 24, 92, 69]

  Base:
  runs/t02/tus_rec_2024_qc/010_LH_Per_L_DtP_qc_0005_0370/local_gradient

  +LGR:
  runs/t02/tus_rec_2024_qc/010_LH_Per_L_DtP_qc_0005_0370/localglobal_gradient

  +LGR+CGA:
  runs/t02/tus_rec_2024_qc/010_LH_Per_L_DtP_qc_0005_0370/localglobal_responsibility

  +LGR+CGA+PP:
  runs/t06/tus_rec_2024_qc/010_LH_Per_L_DtP_qc_0005_0370/canonical_sweep

  使用当前 manifest 里的完整绝对路径解析实际 run directory。

  ### Panel 样式

  - 保留黄色 crop、放大 inset 和连接线。
  - 删除单独的 Error 行。
  - 每个非 GT panel 左下角可以保留两行紧凑数值：

    PSNR xx.xx
    SSIM x.xxx

  - 数值框不得遮挡主要结构或 inset。
  - GT 不显示指标。
  - 两行使用不同 crop，但同一行所有 variant 必须完全对齐。
  - 不允许对各 variant 独立调整亮度或对比度。

  ### 尺寸建议

  figsize = (12.5, 4.6)
  dpi = 300
  wspace = 0.025
  hspace = 0.07

  最终插入双栏后的高度建议控制在 2.3–2.6 inch。

  ### 建议 caption

  Qualitative ablation of the three proposed components on Synthetic Liver and
  TUS-REC-2024. Starting from the base model, LGR, CGA, and PP are progressively
  introduced. LGR denotes the physics-aware local-global Gaussian renderer, CGA
  the contribution-aware Gaussian adaptation, and PP the acquisition-guided
  pseudo-pose parameterization. Yellow boxes show identical regions at a common
  magnification within each row.

  ———

  ## 图 3：三个创新点的一体化参数敏感性图

  把下面三张旧图的数据重新组织：

  paper_easy_pose_rho_sensitivity
  paper_renderer_parameter_sensitivity
  paper_dc_parameter_sensitivity

  不要简单把三张旧 PNG 拼接起来，需要直接读取 manifest 后重新绘制，统一字体、颜色、坐标轴和 panel 风格。

  ### 输出名称

  paper_ablation_parameter_sensitivity.pdf
  paper_ablation_parameter_sensitivity.png
  paper_ablation_parameter_sensitivity_manifest.json

  建议新脚本：

  scripts/production/ablations/ours/
  make_tmi_ablation_sensitivity_summary_v2.py

  ### 整体布局

  采用横向三部分：

  (a) Pseudo-pose spacing
  (b) Local-global renderer
  (c) Contribution-aware adaptation

  建议 1 × 3 主布局，其中 (c) 内部再嵌入两个小 heatmap。

  figsize = (13.8, 4.4)
  dpi = 300

  最终插入双栏后的高度约 2.2 inch。

  ### Panel (a)：Pseudo-pose spacing scale

  数据源：

  paper_easy_pose_rho_sensitivity_manifest.json

  处理方式：

  - 横轴：\rho
  - 纵轴：\Delta PSNR，相对于同一数据集 rho=1.0 的结果
  - 两条曲线：
      - Synthetic Liver：蓝色圆点实线
      - TUS-REC-2024：橙色方块虚线

  - 在 rho=1.0 处画灰色竖向虚线并标记 Default
  - 用空心大圆标出各数据集的最优点
  - 图例名称必须是：

    Synthetic Liver
    TUS-REC-2024

  - 不再使用 Synthetic、TUS-QC 或 Easy-Pose

  使用相对 PSNR 而不是绝对 PSNR，可以把两个数据集放在同一个纵轴上，直接显示参数稳定范围。

  ### Panel (b)：Local-global renderer sensitivity

  数据源：

  paper_renderer_parameter_sensitivity_manifest.json

  采用紧凑的横向 range/dot plot。

  纵轴依次排列：

  λ_ray
  β
  T_min
  N_q
  κ_glo

  字段映射：

  strength             -> λ_ray
  beta                 -> β
  min_transmission     -> T_min
  quadrature_points    -> N_q
  candidate_k_sigma    -> κ_glo

  横轴：

  ΔPSNR relative to the production default

  绘制规则：

  - 每个参数分别计算所有测试设置相对于 default 的 ΔPSNR。
  - Synthetic Liver 使用蓝色圆点。
  - TUS-REC-2024 使用橙色方块。
  - 同一参数的两个数据集在 y 方向轻微错开。
  - 每组测试值用细线连接其最小值与最大值，并画出全部测试点。
  - x=0 画灰色竖向虚线。
  - 删除旧图右侧大段 Default and best 文字说明。
  - 默认值和 tested range 放入 figure caption 或 manifest，不在图内占据空间。
  - 不再出现 quad. points、min trans.、candidate k 等非 Methods 术语。

  ### Panel (c)：Contribution-aware adaptation sensitivity

  数据源：

  paper_dc_parameter_sensitivity_manifest.json

  只使用 Group == densify 的 4×4 网格，因为它直接对应最新版 Methods 中的：

  q_d = densification-score quantile
  B   = per-event densification budget

  在 Panel (c) 内放置两个并排的小 heatmap：

  Synthetic Liver
  TUS-REC-2024

  坐标轴：

  x-axis: B
  values: 0.02, 0.05, 0.08, 0.12

  y-axis: q_d
  values: 0.94, 0.96, 0.98, 0.99

  颜色表示：

  ΔPSNR relative to the production default

  要求：

  - 两个 heatmap 使用相同色域。
  - 使用蓝—灰—橙的发散色图，中心固定为 0。
  - 建议共享范围：

    -0.03 dB to +0.03 dB

  - 只使用一个共享 colorbar。
  - 默认设置 q_d=0.98, B=0.05 用黑色星号或黑框标记。
  - 每个数据集的最优 cell 用黄色边框标记。
  - cell 中仅显示带符号的三位小数，例如：

    +.008
    -.015

  - 不再使用 DC、max fraction、score quantile 等旧模块名称。

  pruning 的有无性和组成作用由对应消融表报告，因此不需要把旧图的所有 pruning 网格继续塞进这张紧凑图。

  ### 建议 caption

  Parameter sensitivity of the three proposed components on Synthetic Liver and
  TUS-REC-2024. (a) PSNR variation with the pseudo-pose spacing scale rho.
  (b) PSNR ranges induced by the principal parameters of the local-global
  renderer. (c) Joint sensitivity to the densification-score quantile q_d and
  per-event budget B in contribution-aware Gaussian adaptation. All PSNR
  differences are measured relative to the corresponding production defaults;
  default settings are marked in black.

  ———

  ## 最终验收要求

  服务器 Codex 完成后必须逐项检查：

  1. 三张 PDF 均可被 pdfinfo 正常读取。
  2. PNG 与 PDF 内容一致。
  3. 没有缺失 method panel。
  4. 图 1 必须是严格的 4×9。
  5. 图 1 的八个方法名称和顺序完全正确。
  6. 每一行所有方法和 GT 使用相同 item、crop、显示范围和 inset 位置。
  7. 图 2 不得保留 Error 行。
  8. 图 2 不得出现 LG/DC/EP/Easy-Pose。
  9. 图 3 不得出现 TUS-QC 或旧模块名称。
  10. 最终缩放到 6.9 inch 后，字体和线条仍可辨认。
  11. 所有数值必须由 manifest 读取，不允许手工重新录入。
  12. 输出 manifest 必须能追溯每一个 panel 的数据来源。
  13. 保留旧 frozen 图片和脚本，新结果全部使用新文件名。
  14. 返回以下文件：

     paper_all_datasets_2d_rendering.pdf
     paper_all_datasets_2d_rendering.png
     paper_all_datasets_2d_rendering_manifest.json

     paper_ablation_joint_visual_compact.pdf
     paper_ablation_joint_visual_compact.png
     paper_ablation_joint_visual_compact_manifest.json

     paper_ablation_parameter_sensitivity.pdf
     paper_ablation_parameter_sensitivity.png
     paper_ablation_parameter_sensitivity_manifest.json

  这套设计受学术可视化工作流影响的关键点是：三张图各自承担不同证据功能——完整方法对比、模块增量效果、参数稳定性——避免在有限页面里重复呈现同一种证
  据。
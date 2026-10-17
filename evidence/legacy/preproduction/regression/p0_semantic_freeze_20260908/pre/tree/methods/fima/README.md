# FiMA 离线管线说明

本目录实现 **传感器无关超声序列** 下的 **FiMA 风格 PEM（图像序列 → 相对位姿）**，以及与 ultrags **解耦 fork** 的 **PoseRefiner** / PoseWash 式 **内循环优化**。适用于 omniultra 的统一 **Engine（state + callbacks）** 与配置合并流程。

**本期不包含**：DCL-Net / 3D ResNeXt baseline；UltraGS 在线渲染接入（协议与矩阵约定与 ultrags 对齐，便于日后拼接）。

---

## 1. 代码结构

| 文件 | 作用 |
|------|------|
| `configs.py` | `FiMAConfig` 及子配置（PEM、损失、序列划分、PoseRefiner、PoseWash、`pose_gt_delta` 权重等） |
| `state.py` | `FiMAOfflineState`：scene、sampler、PEM / Refiner 及其优化器 |
| `callbacks.py` | `FiMAOffline*Callback`：Setup / Train / Test / Save / Finish |
| `pose_geometry.py` | 相对第 0 帧的 \(T_i = P_i P_0^{-1}\)、欧拉↔`4×4`、世界系合成 \( \hat{P}_i^{world}=\hat{T}_i P_0^{GT}\)、轨迹误差 |
| `sequence_dataset.py` | `FiMASequenceSampler`：因果滑窗、train/val 帧划分、`FiMAWindowBatch` |
| `remamba_blocks.py` | 时序混合：优先 `mamba_ssm.Mamba`，失败或未安装时退化为 `TransformerEncoderLayer` |
| `pose_estimator.py` | `FiMAPoseEstimator`：CNN stem + 序列混合 + 回归头 |
| `losses.py` | PEM 监督：L1 / 可选 MSE / Pearson（旋转分量用最短弧包装） |
| `pose_refiner.py` | 自 `methods/ultrags/pose_refiner.py` fork，语义应保持同步 |
| `pose_wash_losses.py` | `PoseGtDeltaLossBatch`，与 ultrags `pose_gt_delta` 一致（批量版） |
| `pose_wash.py` | Refiner 内循环 Adam（无 GS 渲染） |

文档与参考：`Sensorless Freehand...md`、`Fine-Grained Context...md`、`TUS-REC-2024.html` 等保留在同级目录，不参与 import。

---

## 2. 姿态与监督约定

- **数据集真值**：每帧 `SliceMeta.pose` 为 **`probe → world`** 的 `4×4`（与场景加载一致）。
- **锚帧**：全局索引 **0**，\(P_0^{GT}\) 固定为参考。
- **`scene.loading.pose_type: relative`（与 `data/synthetic_liver/gen_relative.py` 一致）**：磁盘存 \(R_i=\mathrm{inv}(P_0^{abs})\,P_i^{abs}\)，\(R_0=I\)。此时 `SliceMeta.pose` 即「相对第 0 帧」的齐次阵，且 \(P_0=I\)，PEM 中 `relative_pose_from_world(P_i,P_0)` 与直接读 \(P_i\) 一致。
- **PEM 回归目标**（\(i \ge 1\)）：  
  \[
  T_i^{GT} = P_i^{GT}\,(P_0^{GT})^{-1}
  \]
  用 `pose_geometry` 转为 6 维：**平移（mm）+ ZYX 欧拉（rad）**，旋转矩阵约定为 \(R = R_z R_y R_x\)。
- **预测回到世界系**（评测或 Refiner 输入）：  
  \[
  \hat{P}_i^{world} = \hat{T}_i\, P_0^{GT}
  \]
- **PoseRefiner**：仍在 **`4×4` 矩阵 + se(3) twist** 上工作；**不要把 PEM 的 6D 欧拉与 twist 混算**，一律先化为矩阵再 compose。

---

## 3. 因果滑窗与数据划分

- 对全局末帧索引 **`i`**，图像窗口为 **`[i-N+1,\, i]`**，共 **`N = pem.window_size`** 帧（默认 10）；**当前帧为窗口最后一帧**，与「逐帧流过」的 GS 管线对齐。
- **训练**：在 **非 val** 且 **`i ≥ max(1, N-1)`** 的帧上随机采样末帧 `i`，保证完整窗口（实现上要求 **`len(sweep) ≥ window_size`**，否则初始化 sampler 即报错）。
- **验证**：按 **`sequence.val_ratio`**（默认 0.1）与 **`sequence.split_seed`** 将帧划分为 val；PEM/Refiner 的 **`on_test` / 训练期 eval** 默认在 **val 子集**上统计（若无可用 val 帧则回退行为见代码）。
- **左侧不足窗口**：由 **`sequence.pad_mode`**（`repeat_edge` / `zero`）处理；完整窗口训练路径下一般不触发。

### 3.1 与 UltraGS 一致：**下采样 + NVS 留出**

- **分辨率**：FiMA 从 `SingleSweepContainer` 读图，**`H×W` 随 `scene` 加载的 `images_name`（如 `images_x2.npy`）与 `DatasetMeta` 自动一致**；无需在 FiMA 内写死分辨率。实验 yaml 使用与 GS 相同的 `liver_x2` + `images_x2.npy` 即可。
- **留出法**：当 `scene.loading.test_mode: holdout_interval` 且 `Scene` 已生成 **`train_indices` / `test_indices`** 时，若 **`sequence.respect_scene_holdout: true`（默认）**，FiMA 会：
  - 用 **同一组** train/test 全局帧索引划分；
  - **训练采样**仅保留「因果窗内每一真实帧索引都属于训练集」的末帧 `i`，避免窗跨过留出帧造成监督泄漏。
- **验证 / `on_test`**：仍在 **`test_indices`（留出帧）** 上评 pose 误差；此时窗内可含训练帧作为上下文（与推理一致），仅末帧落在留出集。

### 3.2 与「Local Window」叙事对齐（姿态侧）

- FiMA 的 **`pem.window_size` 本身就是短时序局部上下文**（因果窗），论文叙事上可与 UltraGS 的 **`model.offline.local_window.window_size_frames`** 刻意设为 **相同或成比例**，写成「姿态网络与 GS 共用同一局部时间尺度」。
- 若需 **显式对照**：可跑两组实验——**大窗（长上下文）** vs **与 GS local window 同宽的小窗**，比较 **val 上 trans/rot 误差与漂移**，作为「local 化姿态估计」的消融，无需改网络结构。

### 3.3 双图相邻帧 PEM（`pem.input_mode: adjacent_pair`）

- 输入 **`(B, 2, 1, H, W)`**（全局相邻帧 `i` 与 `i+1`），网络输出 **6D**，对应 \(T_{i\to i+1}\) 满足 **`P_{i+1} = P_i\,T`**（与 `pose_geometry.adjacent_frame_transform_gt` 一致）。
- **`pem.train_protocol`**：`standard` / `phase_a_full` / `phase_a_local_window`；阶段 A 评测在留出 **test** 帧上 **从第 0 帧链式相乘** 得到 \(\hat R_t\) 再与 `SliceMeta.pose` 比 SE(3) 误差，日志键为 **`eval/pem_chain/val/...`**。
- 历史一键脚本已清理；新入口使用 `experiment=preproduction/baseline_reproduction/fima/paper_edges`。**Refiner 阶段**与 `adjacent_pair` 尚未接线（Setup 会 `NotImplementedError`）。

---

## 4. PEM 输出形状

由 **`pem.pem_output_layout`** 控制：

| 取值 | `pred_pose_6` shape | 含义 |
|------|---------------------|------|
| `last_frame_rel0`（默认） | `(B, 6)` | 多帧：对窗口**最后一帧**（全局 `i`）回归相对 \(P_0\) 的 \(T_i\)；**相邻对模式**下即为单条边的 6D |
| `window_frames_rel0` | `(B, N-1, 6)` | 对窗口内局部索引 **`1 … N-1`** 各回归相对 \(P_0\) 的 \(T\)（局部 0 槽不做多头输出） |

损失在 `losses.pem_supervision_loss` 中与 GT 对齐；旋转分量使用 `angle_wrap_diff` 避免 \(2\pi\) 跳变。

---

## 5. 损失与 Refiner 目标

**PEM（`pem_loss`）**

- **L1**：默认开启（`use_l1`, `lambda_l1`）。
- **MSE**：默认关闭（`use_mse`），可做对照实验。
- **Pearson**：默认开启（`use_pearson`, `lambda_pearson`）；`pearson_mode` 为 `per_channel`（对 6 维分别相关再平均）或 `batch_flat`。

**Refiner 阶段**

- 优化目标为 **`PoseGtDeltaLossBatch`**：令 \(\Delta_{pred} = P_{coarse}^{-1} P_{ref}\) 接近 \(\Delta_{gt} = P_{coarse}^{-1} P_{gt}\)（与 ultrags `pose_gt_delta` 一致）。
- **训练**：对每个 batch 执行 **`pose_wash.num_pose_wash_steps`** 次 Refiner 内循环 Adam（见 `pose_wash.optimize_pose_refiner_batch`）。
- **评测（`on_test`）**：对当前 Refiner **只做单次前向**，不在 eval 中改写权重，避免指标与 checkpoint 不一致。

---

## 6. 与 Engine / ultrags 的关系

- **入口**：仓库根目录 `run.py` 已 `import methods.fima.state`、`methods.fima.callbacks`，用于注册 `STATE_REGISTRY` / `CALLBACK_REGISTRY`。
- **模型注册**：`configs/base/models.py` 中 `MODEL_CONFIG_REGISTRY["fima"] = FiMAConfig`。
- **配置加载**：实验 yaml 需指定 `model.name: fima`，并与 `FiMAConfig` 结构对应（参见 `_base_/models/fima.yaml`）。
- **ultrags**：本包 **不引用** ultrags 在线回调；`pose_refiner.py` 为 **独立 fork**，若修正数值语义，建议按文件头说明考虑是否与 ultrags 双端同步。

---

## 7. 配置要点（`FiMAConfig`）

| 字段 | 说明 |
|------|------|
| `fima_component` | `pem`：只训/评 PEM；`refiner`：冻结 PEM，训/评 PoseRefiner |
| `pem.*` | 窗口大小、布局、stem/mixer、是否优先 Mamba、学习率、batch 等 |
| `pem_loss.*` | L1/MSE/Pearson 开关与权重 |
| `sequence.*` | val 比例、划分种子、pad 模式 |
| `pose_refiner.*` | 与 ultrags PoseRefiner 对齐的超参（含 `max_delta_rot_deg` 等） |
| `pose_wash.*` | 内循环步数、门控相关开关（训练路径使用步数；eval 单次前向） |
| `pose_gt_delta.*` | Refiner 损失的旋转/平移权重 |
| `pem_checkpoint_path` | **`fima_component=refiner` 时推荐必填**：冻结 PEM 的权重文件路径 |
| `refiner_checkpoint_path` | 可选：继续训练 Refiner 时加载 |

命令行使用 **`task=train` / `task=test`**（见 `engine.engine.Engine`）；**阶段选择**靠 **`model.fima_component`**，而非额外 task 字符串。

---

## 8. 使用方法

### 8.1 依赖

- **PyTorch**（与工程其余部分一致）。
- **可选**：`mamba-ssm`（与当前 CUDA/torch 版本匹配的 wheel）。未安装或初始化失败时，`remamba_blocks` 自动退化为 **TransformerEncoderLayer**。

### 8.2 训练 PEM

示例入口：`experiment=preproduction/baseline_reproduction/fima/paper_edges`（可用 Hydra 覆盖项调整 `scene.loading` 路径）。

```bash
cd /path/to/omniultra
python run.py experiment=preproduction/baseline_reproduction/fima/paper_edges task=train
```

- **`engine.iterations`**：PEM 迭代次数。
- **`engine.interval.test` / `save`**：训练中周期性 **val 评测** 与 **checkpoint**。
- **`engine.resume`**：填写迭代号字符串时，从 `output_dir/checkpoints/iteration_<n>_*.pth` 恢复（见下节）。

### 8.3 仅评测 PEM

```bash
python run.py experiment=preproduction/baseline_reproduction/fima/paper_edges task=test
```

可先训练得到 checkpoint，再在 yaml 或 CLI 覆盖中设置 `engine.resume`。

### 8.4 训练 PoseRefiner（PEM 已训好）

1. 准备 PEM 权重文件（通常为某次保存的 **`pose_estimator`** state dict）。
2. 复制或新建 yaml：`model.fima_component: refiner`，并设置  
   `model.pem_checkpoint_path: /绝对或相对路径/xxx.pth`
3. 配置 `engine.state`、`engine.needed_callbacks` 仍为 FiMA 离线五条 callback（与 PEM 实验相同）。
4. 执行：

```bash
python run.py experiment=preproduction/baseline_reproduction/fima/paper_edges task=train model.fima_component=refiner
```

### 8.5 评测 Refiner

同一 Hydra alias，`task=test`。指标见日志中的 `eval/refiner/...`（粗轨迹 vs refine 后轨迹相对 GT 的平移 RMSE、旋转均值等）。

---

## 9. Checkpoint 命名

由 `utils.io_manager.IOManager.save_checkpoint` 写入：

```
{system.output_dir}/checkpoints/iteration_{k}_pose_estimator.pth
{system.output_dir}/checkpoints/iteration_{k}_pem_optimizer.pth
```

Refiner 阶段对应：

```
iteration_{k}_pose_refiner.pth
iteration_{k}_pose_refiner_optimizer.pth
```

加载 **`pem_checkpoint_path`** 时，文件应为 **`torch.save(state_dict)`** 或可解包出 `state_dict` 的字典；与上述命名一致即可。

---

## 10. 常见问题

1. **`ValueError: Sequence length < pem.window_size`**  
   缩短 `pem.window_size`，或换更长 sweep。

2. **希望对照 MSE**  
   在 yaml 中设 `model.pem_loss.use_mse: true`，并可调 `lambda_mse`。

3. **无 Mamba**  
   设 `pem.prefer_mamba: false` 可强制走 Transformer；或为当前环境安装匹配的 `mamba-ssm`。

4. **Refiner 几乎不动**  
   检查 `pose_refiner.max_delta_rot_deg`、`max_delta_trans_mm`；ultrags 默认曾为 0° 导致旋转分支被压死，本仓库 `fima.yaml` 已给小型非零示例。

---

## 11. 扩展（未实现）

- **跨 sweep manifest**（论文式 train/val/test sweep 列表）当前未接；需要时在 `FiMASequenceSampler` 侧扩展数据源即可。

如需与更长策划文档对照，可参阅仓库外的计划稿（例如 `.cursor/plans/` 下 FiMA 相关 markdown）；**以本 README 与代码为准**随迭代更新。

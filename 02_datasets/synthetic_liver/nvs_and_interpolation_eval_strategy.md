# NVS 与插值评测双协议策略（当前代码基线）

## 背景结论

- 在 `interpolated_file`（中间帧插值测试）协议下，指标显著更高，适合体现高保真连续重建能力。
- 在 `holdout_interval`（留出法 NVS）协议下，尤其 SSIM 会明显下降，更符合超声真实泛化难度。
- 论文层面建议采用“双协议叙事”：
  - 主线：NVS（留出法）对齐社区标准。
  - 副线：插值协议展示高保真连续重建优势。

## 当前代码状态（已支持）

### 1) Scene 层测试协议切换

已在 `scene/scene.py` 支持：

- `scene.loading.test_mode: interpolated_file`
  - 读取 `test_images_name` / `test_poses_name` 文件作为测试集（原行为）。
- `scene.loading.test_mode: holdout_interval`
  - 从训练 sweep 内按规则自动划分测试帧：
    - `holdout_trim_each`：掐头去尾帧数
    - `holdout_num_samples`：中间均匀抽取测试帧数
  - 自动构建：
    - `scene.train_indices`（训练索引池）
    - `scene.test_indices`（测试索引池）
  - `build_init_gaussians()` 只用训练索引，避免测试泄漏。

### 2) Local Window 回调已适配留出索引

`methods/ultrags/offline_callbacks_local_window.py` 已改为优先使用 `scene.train_indices`：

- local window 划分基于训练索引池长度（而非全量原始帧长）。
- 训练阶段窗口采样与 seam joint-opt 采样均先采 local 索引，再映射到全局帧 id。
- 避免 `holdout_interval` 下把测试帧误采到训练流程。

### 3) 配置 schema 已扩展

`configs/base/scene.py` 的 `LoadingParams` 已包含：

- `test_mode`
- `holdout_trim_each`
- `holdout_num_samples`

保证 OmegaConf 能解析实验配置中的新字段。

### 4) 分辨率一致性与运行时自适配

- `scene/container/single_sweep_container.py`：
  - 新增 `images/poses` 形状校验；
  - 新增 runtime meta 分辨率对齐（根据实际 `images.npy` 的 H/W 更新 meta 与像素间距）。
- `scene/factory.py`：
  - offline 下增加 train/test 分辨率一致性检查，避免 `images_x2.npy` 与 `test_images.npy` 混配。

## 推荐实验协议（论文可用）

### Protocol A: NVS（主表）

- `test_mode: holdout_interval`
- 例如：`N=200, trim_each=10, num_samples=40`（train=160/test=40）
- 与 baseline 在同协议上比较（主结果）

### Protocol B: 插值高保真（副表）

- `test_mode: interpolated_file`
- 使用 `test_images*.npy` / `test_poses.npy` 中点插值测试集
- 作为“连续扫描高保真重建”补充指标，不替代主线 NVS

## 写作建议（避免审稿风险）

- 明确区分任务：
  - NVS（留出泛化）
  - 插值重建（连续性与细节保真）
- 避免把插值协议写成“通用 NVS”的主结论。
- 主文主表给 NVS；插值结果放补充主文副表或附录强化项。

## 最小配置示例

```yaml
scene:
  loading:
    test_mode: "holdout_interval"   # or "interpolated_file"
    holdout_trim_each: 10
    holdout_num_samples: 40
```

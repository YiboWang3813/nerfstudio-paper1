# UltraGS Renderer Debug 经验沉淀

## 1) 症状与误导点

- `pytest tests/ops/ultrags_renderer/test_batched_equivalence.py` 报错落在 Python 侧 `F.pad(...)`（`pack_vec3`）并不代表 `F.pad` 本身有问题。
- 这是典型 CUDA 异步错误：真正越界通常发生在更早的 custom kernel，直到下一次 CUDA API/算子调用才被“冲刷”出来。

## 2) 这次的真实根因

- 首个精准前缀是：
  - `[ultrags_renderer.shader] kernel_pixel_alpha_accum_forward failed: an illegal memory access was encountered`
- 进一步定位后，根因在 mapper 的单帧路径 `MapperForwardCUDA`：
  - `pixel_gauss_counts -> pixel_gauss_offsets` 的 CUB `ExclusiveSum` 复用了 `temp_storage`，但没有先 query `scan_pixel_bytes`。
  - 当所需空间超过已有 `temp_storage_bytes` 时，scan 可能写越界，随后 shader 读取损坏的 CSR（`offsets/counts/indices`）触发非法访问。

## 3) 已修复项

- 单帧 mapper 路径补齐了 `scan_pixel_bytes` 查询与按需扩容逻辑（与 batched 路径一致）。
- 移除了算子内的临时 debug 逻辑（`printf` 和 launch 后显式 `cudaGetLastError` 检查），避免影响后续性能测试。

## 4) 推荐调试流程（以后复用）

1. 强制同步：
   - `CUDA_LAUNCH_BLOCKING=1`
   - 可选：`TORCH_SHOW_CPP_STACKTRACES=1`
2. 先在可疑 kernel launch 后临时加 `cudaGetLastError`（仅调试分支），拿到“第一个炸点”。
3. 优先检查：
   - CSR 边界（`offset + count` 是否越界）
   - CUB scan/sort 的 `temp_storage_bytes` 是否先 query 再分配
   - 扁平化索引是否跨 batch/pixel 维度写错
4. 修复后移除临时 debug 逻辑，再做性能测试。

## 5) 验证命令

```bash
cd /home/boywang25/ultrasound-scene-reconstruction/omniultra/ops/ultrags_renderer
bash rebuild.sh

cd /home/boywang25/ultrasound-scene-reconstruction/omniultra
CUDA_LAUNCH_BLOCKING=1 TORCH_SHOW_CPP_STACKTRACES=1 pytest tests/ops/ultrags_renderer/test_batched_equivalence.py -q -s
```

---

## 6) 新增排障经验：per-sample active mask 改造

### 6.1 症状

新增 `active_gauss_mask_batched` 后，等价性测试出现明显偏差：

- `test_batched_with_active_mask_matches_serial_subsets` 失败
- 表现为大面积像素 mismatch，而非少量浮点误差

### 6.2 根因

在 batched mapper 的两个 kernel 中（`src/mapper/core.cu`）：

- `kernel_precompute_gauss_projected_radii_batched`
- `kernel_project_gauss_and_compute_tile_rects_batched`

最初把 mask 早退放在了 `__syncthreads()` 之前。
由于同一个 block 内线程对 mask 判断不同，导致一部分线程 return、另一部分线程继续执行并进入 barrier，触发未定义行为，最终造成结果错误。

### 6.3 修复方式

- 保持 shared 几何加载与 `__syncthreads()` 全线程一致执行
- 将 mask 早退移动到 barrier 之后
- 仅在同步后根据 `active_gauss_mask_batched[b, g]` 决定是否继续当前线程逻辑

### 6.4 复现/回归建议

1. 改完立即重编译：
   - `bash ops/ultrags_renderer/rebuild.sh`
2. 先跑等价性测试（优先）：
   - `pytest tests/ops/ultrags_renderer/test_batched_equivalence.py -q`
3. 再跑性能脚本：
   - Historical debug launchers were removed during script reorganization; inspect git history if the exact legacy launchers are needed.

### 6.5 经验总结

- CUDA kernel 中凡是出现 `__syncthreads()`，必须保证同一 block 中线程控制流一致到达 barrier。
- “看起来是数值误差”的 batch 偏差，很多时候是同步/越界类 bug，不是精度问题。

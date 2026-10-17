# OmniUltra Slurm Skills

本文档只约束服务器、GPU、Slurm、CUDA 编译和实验监控操作。实验生命周期和输出保留策略见 `pipeline_skills.md`；通用开发规范见 `development_skills.md`；文件范围见 `scope_skills.md`。

## 1. 基本原则

1. 训练、评估、CUDA 编译、GPU 测试和长时间 benchmark 必须通过 Slurm 跑在计算节点上。
2. 不在登录节点直接跑 heavy CUDA 任务。
3. 项目生产环境是 conda env `gaussian_splatting`。
4. CUDA extension / renderer 相关任务加载 `cuda/11.8.0`。
5. 新实验使用 Hydra alias：`python run.py experiment=<alias>`。
6. 不使用旧入口：`python run.py --config ...`、`CONFIG=config_files/...`。
7. 新实验输出遵守 `pipeline_skills.md` 的 `outputs/` 协议。
8. Slurm stdout/stderr 写入 `logs/<owner>/`，这些日志不进 git。

## 2. 查看资源

提交前先看资源：

```bash
sinfo
sinfo -N
sinfo -N -o "%N %P %t %G %C"
squeue -u boywang25
```

GPU 选择优先级：

```text
4090/gpu3 > A100 > 4080
```

优先向 4090 节点要卡，也就是优先考虑 `gpu3`。固定节点前必须确认节点有空闲 GPU；如果 `gpu3` 已满，不要一直等 4090，继续看 A100、4080，让实验尽快启动。

## 3. 提交和监控

提交：

```bash
mkdir -p logs/<owner>
sbatch scripts/methods/<owner>/<method_or_idea>/slurm/<job>.sh
```

生产对比脚本通常在：

```bash
sbatch scripts/production/comparisons/<comparison_name>/<job>.sh
```

只返回 job id：

```bash
sbatch --parsable <script>
```

矩阵实验默认并行提交，不要把一组独立训练串成只占一张卡的长队列。submitter 应一次提交多个独立 Slurm job，让调度器同时占用多张可用 GPU；只在任务有真实数据依赖、用户明确要求串行、或资源配额临时限制时才串行。

推荐提交策略：

```text
小矩阵或 smoke: 直接并行提交全部任务。
中等矩阵: 使用 MAX_PARALLEL 控制每波并发数量。
大矩阵: 使用 START_INDEX / END_INDEX 分批提交，每批内部并行。
有阶段依赖: 只在阶段之间加 dependency，阶段内部仍并行。
QOS 限制较紧的大矩阵: 优先考虑单个 Slurm job 申请多张 GPU，在 job 内部启动多个 worker 分片处理任务。
```

如果需要固定节点，优先申请 4090：

```bash
sbatch --nodelist=gpu3 <script>
```

固定 `gpu3` 前先看 `sinfo` / `squeue`。如果 `gpu3` 忙，移除 `--nodelist=gpu3` 或换到其他可用 GPU 节点，而不是把所有任务排成单卡串行。

查看队列：

```bash
squeue -u boywang25
```

取消用户明确认为过时的任务：

```bash
scancel <job_id>
```

读日志：

```bash
tail -n 80 logs/<owner>/<job_name>_<job_id>.out
tail -n 80 logs/<owner>/<job_name>_<job_id>.err
```

监控顺序：

1. `squeue -u boywang25` 看 running/pending。
2. `.out` 看配置 echo、Hydra run dir、训练 summary、final metrics。
3. `.err` 看 tqdm 速度、warning、traceback。
4. 任务完成后检查 `resolved_config.yaml`、`run_manifest.yaml`、`logs/scalars.tsv` 或最终 metric 文件。
5. 再决定是否提交下一批。

## 4. Slurm 脚本位置

预实验脚本：

```text
scripts/methods/baselines/<method>/slurm/**
scripts/methods/ours/<idea_name>/slurm/**
```

生产脚本：

```text
scripts/production/comparisons/<comparison_name>/**
scripts/production/ablations/<method_or_idea>/<ablation_name>/**  # when needed
```

推荐拆分：

```text
run_<task>.sh       # runner，一次跑一个实验
submit_<matrix>.sh  # submitter，批量提交 runner
```

规则：

1. runner 一次只跑一个实验。
2. submitter 负责 sweep、seed、dataset、ablation 矩阵。
3. submitter 用 `EXPERIMENT`、`OVERRIDES`、`EXP_NAME`、`OUTPUT_ROOT` 传参。
4. 不用旧 `CONFIG` 路径表示实验配置。
5. 支持 `START_INDEX` / `END_INDEX` 做局部提交。
6. 默认并行提交独立任务，避免只占一张卡串行跑完整矩阵。
7. 用 `MAX_PARALLEL` 或分批提交控制并发规模，避免一次性塞满 pending 队列。
8. 如果 Slurm array 或多 job 提交触发 `QOSMaxSubmitJobPerUserLimit`，可以把 runner 改成单个 multi-GPU job，在 job 内用 worker 分片跑完整矩阵。

## 5. Runner 最小约定

新 runner 应包含：

```bash
#!/bin/bash
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -J <short_job_name>
#SBATCH -o logs/<owner>/%x_%j.out
#SBATCH -e logs/<owner>/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/boywang25/ultrasound-scene-reconstruction/omniultra}"
CONDA_ENV="${CONDA_ENV:-gaussian_splatting}"
CUDA_MODULE="${CUDA_MODULE:-cuda/11.8.0}"
EXPERIMENT="${EXPERIMENT:?EXPERIMENT is required}"
OVERRIDES="${OVERRIDES:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
EXP_NAME="${EXP_NAME:-${SLURM_JOB_NAME:-run}_${SLURM_JOB_ID:-manual}}"

source ~/.bashrc
conda activate "${CONDA_ENV}"
module load "${CUDA_MODULE}"

cd "${PROJECT_DIR}"
python run.py \
  "experiment=${EXPERIMENT}" \
  "hydra.run.dir=${OUTPUT_ROOT}/${EXP_NAME}" \
  "system.output_root=${OUTPUT_ROOT}" \
  "system.exp_tag=${EXP_NAME}" \
  ${OVERRIDES}
```

具体矩阵、输出根和汇总表规则由 owning script 和 `pipeline_skills.md` 决定。

### 5.1 Multi-GPU / Single-GPU Multi-Process Runner

当任务显存占用较低、但 Slurm 提交数量受限时，可以用一个 Slurm job 申请多张卡，并在 job 内部启动多个 worker。典型用途：

```text
1 个 Slurm job
2 张 GPU
每张 GPU 2 个训练进程
总计 4 个 worker 并行处理 sweep/task 分片
```

这种模式适合 UltraGauss、UltraGray 等单 run 显存占用较低的 GS baseline；不适合显存接近上限、会动态膨胀高斯数、或 CUDA renderer 不支持同卡多进程的任务。使用前先用一两个 sweep 验证显存和速度。

最小模板：

```bash
#!/bin/bash
#SBATCH -c 8
#SBATCH --gres=gpu:2
#SBATCH -J <short_job_name>
#SBATCH -o logs/<owner>/%x_%j.out
#SBATCH -e logs/<owner>/%x_%j.err

set -euo pipefail

WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
GPUS_PER_JOB="${GPUS_PER_JOB:-2}"

TASKS=(task0 task1 task2 task3 task4 task5)

run_one_task() {
  local task_index="$1"
  local gpu_id="$2"
  local slot_id="$3"
  local task="${TASKS[$task_index]}"
  local output_dir="${OUTPUT_ROOT}/runs/${task}"

  mkdir -p "${output_dir}/logs" "${output_dir}/vis"
  echo "[WORKER ${slot_id}] start task_index=${task_index} gpu=${gpu_id} task=${task}"

  CUDA_VISIBLE_DEVICES="${gpu_id}" python run.py \
    "experiment=${EXPERIMENT}" \
    "hydra.run.dir=${output_dir}" \
    "system.output_root=${OUTPUT_ROOT}" \
    "system.exp_tag=${task}" \
    ${OVERRIDES}

  echo "[WORKER ${slot_id}] finish task_index=${task_index} gpu=${gpu_id} task=${task}"
}

run_worker() {
  local slot_id="$1"
  local gpu_id="$2"
  local total_slots="$3"
  local task_index

  for ((task_index = slot_id; task_index < ${#TASKS[@]}; task_index += total_slots)); do
    run_one_task "${task_index}" "${gpu_id}" "${slot_id}"
  done
}

total_slots=$((GPUS_PER_JOB * WORKERS_PER_GPU))
pids=()

for ((slot = 0; slot < total_slots; slot++)); do
  gpu=$((slot / WORKERS_PER_GPU))
  run_worker "${slot}" "${gpu}" "${total_slots}" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

exit "${failed}"
```

Operational rules for this mode:

1. Each worker must write to a unique `hydra.run.dir`; never let two processes share one run directory.
2. Use `CUDA_VISIBLE_DEVICES="${gpu_id}"` per process, where `gpu_id` is local to the Slurm allocation.
3. Do not use Slurm array variables inside this runner unless the job is intentionally also an array.
4. Shared summary TSV appends are acceptable only if the summary script writes complete single-line rows; if corruption appears, summarize into per-worker files and merge after completion.
5. Delete checkpoint payloads after each successful summary when storage is tight.
6. Echo worker slot, GPU id, task index, task name, output dir, and summary path at startup.
7. If any worker fails, the top-level job must exit non-zero after waiting for all workers, so Slurm reports the production batch as failed.
8. This mode reduces Slurm submit count, but it does not reduce GPU memory contention; use it only after confirming same-GPU multi-process training fits.

## 6. Submitter 最小约定

submitter 应：

1. 定义任务矩阵。
2. 导出 `EXPERIMENT`、`OVERRIDES`、`EXP_NAME`、`OUTPUT_ROOT`。
3. 支持 `START_INDEX` / `END_INDEX`。
4. 默认并行提交独立任务；不要用单个 dependency chain 把所有任务串成只用一张卡。
5. 支持 `MAX_PARALLEL` 或 wave/group dependency：每波内部并行，下一波等待上一波完成。
6. 避免触发 `QOSMaxSubmitJobPerUserLimit`；如果 pending 过多，用 `START_INDEX` / `END_INDEX` 或较小的 `MAX_PARALLEL` 分批。
7. echo 每个提交的 job id、任务名、alias、overrides、output root 和 dependency。

submitter 推荐模式：

```bash
MAX_PARALLEL="${MAX_PARALLEL:-4}"
current_group=()

for task in "${TASKS[@]}"; do
  job_id="$(sbatch --parsable --export=ALL,... "${RUNNER}")"
  current_group+=("${job_id}")

  if (( ${#current_group[@]} >= MAX_PARALLEL )); then
    group_dep="$(IFS=:; echo "${current_group[*]}")"
    current_group=()
  fi
done
```

这个模式表示“每组最多并行 `MAX_PARALLEL` 个任务”。只有下一阶段依赖上一阶段结果时，才把 `group_dep` 用到下一阶段的 `--dependency=afterok:<ids>`。

需要直接执行 submitter 时：

```bash
chmod +x scripts/.../submit_<matrix>.sh
```

mode change 应和脚本内容一起提交。

## 7. CUDA / Ops 作业

涉及 CUDA extension rebuild、renderer 测试、GPU reference test 时：

1. 通过 Slurm 跑在计算节点。
2. `conda activate gaussian_splatting`。
3. `module load cuda/11.8.0`。
4. 先 rebuild，再 pytest 或 smoke。
5. `.out` 中 echo `which python`、`CUDA_HOME`、`nvcc --version`、`nvidia-smi`。

示例命令：

```bash
bash ops/<renderer_name>/rebuild.sh
pytest -q -p no:cacheprovider ops/<renderer_name>/tests
pytest -q -p no:cacheprovider tests/ops/<renderer_name>
```

## 8. 常见失败处理

`Permission denied`：

```text
给 submitter 加 chmod +x，并提交 mode change。
```

`QOSMaxSubmitJobPerUserLimit`：

```text
减少一次提交数量，用 START_INDEX/END_INDEX 或较小的 MAX_PARALLEL 分批；不要反复 spam sbatch。
如果 array 仍然因为 submit job count 被拒绝，把任务改成单个 Slurm job 内部多 worker 分片。
例如申请 --gres=gpu:2，并用 4 个 worker 在两张卡上跑，每张卡 2 个进程。
这会让 Slurm 只看到 1 个 job，同时保持 4 个训练进程并行。
```

`QOSMaxCpuPerJobLimit`：

```text
降低 #SBATCH -c 或 --cpus-per-task。
多 GPU / 多 worker job 不一定需要按 worker 数线性增加 CPU；如果训练主要瓶颈在 GPU，可先用 -c 8。
确认 job 从 PENDING 变为 RUNNING 后再离开。
```

训练完成但没有 summary：

```text
从 run 目录的 logs/scalars.tsv、metric 文件或 final eval 输出重建 summary。
检查 .out 中是否有 env var 缺失、KeyError 或汇总脚本路径错误。
```

summary 匹配到错误 run：

```text
改成精确 EXP_NAME/run_dir 匹配，禁止模糊前缀匹配。
```

CUDA code 改了但结果像没变：

```text
重建对应 ops 包，确认 .out 里 import 的 extension 路径和 rebuild 时间。
```

## 9. 汇报要求

提交正式实验后必须记录：

```text
job id
Slurm script path
experiment alias
overrides
output root
Hydra run dir
stdout/stderr log path
final metrics path
```

最终汇报时说明：

1. 哪些 jobs 跑过、job ids 是什么。
2. 是否还有 running/pending jobs。
3. 关键指标或 metric 文件位置。
4. run dir 和 `run_manifest.yaml` 位置。
5. 失败任务的原因和处理决定。
6. 如果代码或配置变更已验证，给出 commit hash。

## 10. 禁止事项

1. 在登录节点直接跑重训练、GPU benchmark 或 CUDA-heavy pytest。
2. 新脚本使用 `CONFIG=config_files/...`。
3. 新命令使用 `python run.py --config ...`。
4. 默认把输出写回旧 `checkpoints/<method>/<exp>`。
5. 靠模糊前缀查找 run 目录。
6. 一次提交大量 pending jobs 导致配额拥塞。
7. 不看日志就反复重提同一个失败任务。
8. 把无依赖的矩阵实验串成单卡长队列，导致多张空闲 GPU 没有被使用。

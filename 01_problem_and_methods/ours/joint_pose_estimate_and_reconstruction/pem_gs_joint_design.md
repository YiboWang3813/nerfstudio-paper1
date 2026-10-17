需要。训练阶段层面的对比正好是立住「不是多跑两遍」的核心；再配合 模块内拆解，审稿人读起来才完整。

1. 必做的「主对角」对比（你说的 1 vs 2）
代号	设定	作用
B0：PEM→GS（无反馈）
PEM 离线训满 → 固定 pred pose → 只训 GS 一次
基线：无张量交换
B1：Alternating + 交换
多 phase：PEM↔GS，带 occupancy（及可选不确定性权重）
主方法：证明闭合有用
报告指标建议 两边都报：

Pose：留出集链式或与 GT 对齐的 trans/rot（同协议）。
重建：GT pose 上等价训练的 oracle 可先固定一条，再在 同一种 pred pose 条件 下比 PSNR/SSIM 等。
这样 训练阶段消融 回答的是：多出「GS→PEM 的辅助信息 + 交替」有没有收益，而不是只有结构差异。

2. 建议再做的「方法内」消融（在 B1 上拆）
在 Alternating + 交换 里拆.communication 形式，证明不是「多轮 schedule」占便宜：

代号	改什么
A1：交替但无 GS 反馈
仍多 phase，但 PEM 只吃 I，不吃 O（也不做
w
(
u
)
w(u)）；每 phase 仍用最新 pose 训 GS。与「只多轮换 pose」最接近。
A2：+ Occupancy 作输入
PEM 输入 concat(I, O)，O 来自上轮固定 GS + 当前 pose 前向。
A3：+ 不确定性加权（若采用）
在 A2 上加
w
(
u
)
w(u) 或只做 A3 不配 O（看谁有效）。
解释预期：

若 A1 ≈ B0（或略好仅因 schedule） 而 A2/A3 明显提升 → 贡献在 几何条件 / 加权，不单是「训久一点」。
若 A1 就明显提升 → 说明 迭代 pose 调度本身有用，要写进讨论，但仍可把 O 说成 incremental gain。
3. 要不要额外训「同等计算量」？
建议 至少一档Fair comparison：

预算匹配：例如总 PEM epochs × K、总 GS steps × K 与 B0+B0 接续的总步数对齐；或 Wall-clock 对齐。
避免被说「alternating 不过多算了」：B1 可与「加长 B0」（同样总步数只训 PEM 或只训 GS） 对比一下。
可作 表：Total PEM updates / Total GS iterations。

4. 可选但更漂亮的消融
Phase 数 K：
K
=
1
K=1（本质是加强版两遍）vs
K
>
1
K>1，看闭合是否饱和。
O 的构造：仅用 lack vs 仅用 T vs 拼接；或 GT pose 渲染的 O_teacher（仅合成）作上界teacher。
冷启动：首轮 GS 用 GT pose vs 首轮用 PEM coarse，看 交替是否修复冷启动。
5. 汇总成论文里一张表怎么列
表 1（训练范式）

B0 PEM→GS（一次）
B1 Alternating + 张量交换（full）
（可选）B0+：两次顺序 PEM→GS（无 O），与 B1 同总预算
表 2（通信机制，在 B1 或固定 K 下）

A1 交替无 O
A2 +O 输入
A3 +O +
w
(
u
)
w(u)
结论：

要做训练阶段层面的消融：1 vs 2（你列的）是 必须。
还要做「有交替但 无 GS→PEM 张量」的 A1，否则分不清是 通信 还是 多 phase schedule。
预算对齐 + 小网格 K 与 O/权重 各拆一刀，通常就够撑起一节 Ablation。
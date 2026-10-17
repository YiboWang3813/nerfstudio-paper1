
本文是考虑脏姿态的情况下 以及流式数据产生 + local_window本位下的 online 流程设计

**与 `docs/methods/baselines/ultrags_baseline/online_slam_design.md` 中模式的关系**：下文「独立优化 / 联合优化 + \(\Phi_p\) 洗姿态」描述的是 **模式 B（GT+Noise）与模式 C（PEM）的通用形态**——二者均有粗姿态 \(\mathbf{P}_i^{\mathrm{coarse}}\) 与 PRM \(\Phi_p\)，区别仅在于 \(\mathbf{P}_i^{\mathrm{coarse}}\) 来自 **加噪 GT** 还是 **PEM 链式预测**。**模式 A（GT）** 无粗/洗姿态：\(\mathbf{P}_i^{\mathrm{refined}}\equiv\mathbf{P}_i\)，不调用 \(\Phi_p\)，也不产生本文中的「pose 清洗」子流程。

---

## 0. 符号（正文叙事沿用）

$W$ 是窗口大小
$W_j$ 表示第$j$个Local Window
Local Window每次更新只向前走一个idx
$C_i^{W_j}$ 表示在$W_j$窗口内第$i$帧$I_i$可用于优化GS的优化次数预算
$C_i^{all}$ 表示在滑窗过程中第$i$帧可用于优化GS的优化次数预算
$C_i^{all} = sum(C_i^{W_j}) \ for \ W_j \ in \ W_{related} $ 其中 $W_{related}$ 就是$I_i$ 在整个滑窗过程中会接触到的所有Local Window
$\Phi_p$ 表示 Pose 清洗专用 MLP 它输入是一个脏 Pose 输出是一个干净 Pose 主要通过冻结GS的渲染损失优化 也可以增加 pose 相关的辅助损失

关于高斯的参与管理，就是如何实现拿到属于一帧的高斯，我建议引入 `membership` 属性，它的shape是 $G \times N_m$ 其中 $G$ 是当前所有高斯的数量 $N_m$ 是 `cfg` 设置的超参数，一般可以是3或者5. 一行 `membership` 初始化为一个长度为 $N_m$ 的张量，内部数值初始化为-1，表示这个高斯还不属于任何帧。
有这么几个关键节点：
1. 一个高斯刚被采样出生的时候，给把它的 `memebership` 的首位赋值为对应帧的id 得到 `membership = [id_born, -1, -1, -1, -1]`。
2. 当一个新帧输入，产生梯度，进行优化时，一个老高斯可能也会被影响到，这个影响不影响要根据 `render()` 函数来决定，这个函数可以生成一个当前帧 `frame_id` 影响了哪些高斯的张量，根据这个张量来修改老高斯的 `membership`。例如，帧1也可能影响帧0优化过的高斯，那这些高斯的 `membership = [0, 1, -1, -1, -1]`。每个iter优化完之后都要调用这个更新 membership的函数，保证高斯-图像帧的从属关系永远最新。这个新帧输入适用于单帧独立优化和多帧联合优化。

$G_i$ 表示 `membership` 只有 $i$ 也就是只和 $I_i$ 有关的高斯
$G_{ijk}$ 表示 `membership` 有 $i,j,k$ 也就是和 $I_i, I_j, I_k$ 都有关的高斯
$G_{global}$ 表示全局高斯池子 它里面存放着到目前为止所有出现、优化过的高斯

按照Local Window本位思想 + pose是脏的整体设计流程

每个窗口内，单帧独立优化的预算比例 $r_i^{front}$ 和 多帧联合优化的预算比例 $r_i^{rear}$ 满足 $r_i^{front} + r_i^{rear} = 1$。

涉及GS的每步优化完之后，都要刷新$G_{global}$中全部高斯的`membership`属性，因为一次优化对应一帧id，高斯也会移动，必须通过`render`函数返回的`frame_to_gauss_indices`张量来实时修改全部高斯的`membership`属性。

对$W_0$而言，这里我假设$W=3$，以保证泛化性，接着我一帧一帧的说流程
1. 独立优化帧0: 拿到 $I_0$ ，它的 pose 一定是干净的，记做 $P_0$ ，从 $I_0$ 上采样高斯得到 $G_0$，现在有 $G_{global} = \{ G_0 \}$，使用 $P_0$ 和 $G_{global}$ 产生 $I_0^{pred}$，用 $I_0$ 和 $I_0^{pred}$ 产生渲染损失，传递梯度给 $G_{global}$，优化 $C_0^{W_0} \cdot r_0^{front}$ 次，得到优化后的 $G_{global}$。
2. 独立优化帧1: 拿到 $I_1$，它的 pose 是脏的，记做 $P_1^{coarse}$。
   1. 先要进行 pose 清洗：冻结现在全部高斯的权重 $G_{global} \rightarrow G_{global}^{freeze}$，让 $P_1^{coarse}$ 经过 $\Phi_p$ 得到 $P_1^{pred}$，使用 $P_1^{pred}$ 以及 $G_{global}^{freeze}$ 产生 $I_1^{pseudo}$，用 $I_1^{pseudo}$ 和 $I_1$ 产生渲染损失，传递梯度给 $\Phi_p$，优化 $N_p$ 次，最后让 $P_1^{coarse}$ 经过优化后的 $\Phi_p$ 得到一个干净的 pose $P_1^{refined}$。
   2. 接着进行高斯采样和优化：恢复全局高斯池的权重 $G_{global}^{freeze} \rightarrow G_{global}$，从 $I_1$ 上采样高斯得到 $G_1$，把 $G_1$ 加入全局高斯池 $G_{global} = \{G_0, G_1\}$ ，使用$P_1^{refined}$ 和 $G_{global}$ 产生 $I_1^{pred}$，用 $I_1$ 和 $I_1^{pred}$ 产生渲染损失，传递梯度给 $G_{global}$，优化 $C_1^{W_0} \cdot r_1^{front}$ 次，得到优化后的 $G_{global}$。注意：在优化过程中，高斯会自己决定和哪帧发生关系，所以优化后的全局高斯池的构成如下：$G_{global} = \{G_0, G_1, G_{01}\}$。
3. 独立优化帧2: 拿到 $I_2$，它的 pose 是脏的，记做 $P_2^{coarse}$。
   1. 先要进行 pose 清洗：冻结现在全部高斯的权重 $G_{global} \rightarrow G_{global}^{freeze}$，让 $P_2^{coarse}$ 经过 $\Phi_p$ 得到 $P_2^{pred}$，使用 $P_2^{pred}$ 以及 $G_{global}^{freeze}$ 产生 $I_2^{pseudo}$，用 $I_2^{pseudo}$ 和 $I_2$ 产生渲染损失，传递梯度给 $\Phi_p$，优化 $N_p$ 次，最后让 $P_2^{coarse}$ 经过优化后的 $\Phi_p$ 得到一个干净的 pose $P_2^{refined}$。
   2. 接着进行高斯采样和优化：恢复全局高斯池的权重 $G_{global}^{freeze} \rightarrow G_{global}$，从 $I_2$ 上采样高斯得到 $G_2$，把 $G_2$ 加入全局高斯池 $G_{global} = \{G_0, G_1, G_{01}, G_2\}$ ，使用$P_2^{refined}$ 和 $G_{global}$ 产生 $I_2^{pred}$，用 $I_2$ 和 $I_2^{pred}$ 产生渲染损失，传递梯度给 $G_{global}$，优化 $C_2^{W_0} \cdot r_2^{front}$ 次，得到优化后的 $G_{global}$。注意：在优化过程中，高斯会自己决定和哪帧发生关系，所以优化后的全局高斯池的构成如下：$G_{global} = \{G_0, G_1, G_{01}, G_2, G_{12}, G_{02}, G_{012}\}$。
4. 联合优化帧012:
   1. 用 $P_0$ 和 $G_{global}$ 产生 $I_{0}^{pred}$，用 $I_0^{pred}$ 和 $I_0$ 产生渲染损失，传递梯度给 $G_{global}$，优化 $C_0^{W_0} \cdot r_0^{rear}$ 次。
   2. 用 $P_1^{refined}$ 和 $G_{global}$ 产生 $I_{1}^{pred}$，用 $I_1^{pred}$ 和 $I_1$ 产生渲染损失，传递梯度给 $G_{global}$，优化 $C_1^{W_0} \cdot r_1^{rear}$ 次。
   3. 用 $P_2^{refined}$ 和 $G_{global}$ 产生 $I_{2}^{pred}$，用 $I_2^{pred}$ 和 $I_2$ 产生渲染损失，传递梯度给 $G_{global}$，优化 $C_2^{W_0} \cdot r_2^{rear}$ 次。
   4. 上述联合优化可以用 $(P,G,I,T)$ 来包装，搞一个优化池，随机发生优化过程，不要死板的逐个进行。
   5. 经过联合优化，$G_{global}$ 中每个成分的比例会发生变化，但肯定还是由 $G_0, G_1, G_2, G_{01}, G_{02}, G_{12}, G_{012}$ 构成。

向 $G_{global}$ 里新增高斯的表示方法还有一种：$G_{global}^{cur} = G_{global}^{prev} + G_i$，下面我会用到，上面我展开写成分是为了具体描述高斯池中每个高斯的`membership`会发生改变。

接着，对 $W_1$ 而言，它对应的帧组合是 123，也就是在 $W_0$ 的基础上去掉0，增加3.
1. 独立优化帧3: 拿到 $I_3$，它的 pose 是脏的，记做 $P_3^{coarse}$。
   1. 先要进行 pose 清洗：冻结现在全部高斯的权重 $G_{global} \rightarrow G_{global}^{freeze}$，让 $P_3^{coarse}$ 经过 $\Phi_p$ 得到 $P_3^{pred}$，使用 $P_3^{pred}$ 以及 $G_{global}^{freeze}$ 产生 $I_3^{pseudo}$，用 $I_3^{pseudo}$ 和 $I_3$ 产生渲染损失，传递梯度给 $\Phi_p$，优化 $N_p$ 次，最后让 $P_3^{coarse}$ 经过优化后的 $\Phi_p$ 得到一个干净的 pose $P_3^{refined}$。
   2. 接着进行高斯采样和优化：恢复全局高斯池的权重 $G_{global}^{freeze} \rightarrow G_{global}$，从 $I_3$ 上采样高斯得到 $G_3$，把 $G_3$ 加入全局高斯池 $ G_{global} = G_{global}^{cur} = G_{global}^{prev} + G_3$ ，使用 $P_3^{refined}$ 和 $G_{global}$ 产生 $I_3^{pred}$，用 $I_3$ 和 $I_3^{pred}$ 产生渲染损失，传递梯度给 $G_{global}$，优化 $C_3^{W_1} \cdot r_3^{front}$ 次，得到优化后的 $G_{global}$。此时由于高斯的自由选择，$G_{global}$ 中的成分可能会增加出来 $G_{23}, G_{13}, G_{123}$。
2. 联合优化帧123：
   1. 用 $P_1^{refined}$ 和 $G_{global}$ 产生 $I_{1}^{pred}$，用 $I_1^{pred}$ 和 $I_1$ 产生渲染损失，传递梯度给 $G_{global}$，优化 $C_1^{W_1}$ 次。
   2. 用 $P_2^{refined}$ 和 $G_{global}$ 产生 $I_{2}^{pred}$，用 $I_2^{pred}$ 和 $I_2$ 产生渲染损失，传递梯度给 $G_{global}$，优化 $C_2^{W_1}$ 次。
   3. 用 $P_3^{refined}$ 和 $G_{global}$ 产生 $I_{3}^{pred}$，用 $I_3^{pred}$ 和 $I_3$ 产生渲染损失，传递梯度给 $G_{global}$，优化 $C_3^{W_1} \cdot r_3^{rear}$ 次。

在优化结束后，现在的 $G_{global}$ 的成分和比例如何呢，显然应该有如下规律：
1. $G_0$, $G_1$, $G_2$, $G_3$ 这种只和某一帧相关的高斯，比例适中。
2. $G_{01}$, $G_{12}$, $G_{23}$ 这种和相邻两帧相关的高斯，比例也不少。
3. $G_{02}$, $G_{03}$, $G_{13}$ 这种跳跃帧相关的高斯应该不存在，因为高斯椭球是连续占有空间的。
4. $G_{012}$, $G_{123}$ 这种和连续多帧相关的高斯也有一定比例。
5. $G_{0123}$ 这种横跨4帧的椭球也不是没有可能存在。

不管 $G_{global}$ 中高斯椭球的归属如何，它是 $G_0$ 还是 $G_{012}$ 我们总归是拿到了每一个高斯的归属 `membership`。
好，接着思考一个问题，随着窗口的前进，因为超声的扫描具有强烈的局部性，$W_8$ 还会影响到 $G_0$ 吗？显然不会，所以我们可以设置一个超参数，让后面的 Local Window 只从 $G_{global}$ 中拿它影响范围附近的一个子集，而不是从0积累到现在的全量高斯集合。例如，设 $l$ 表示这个数。我们让 $W_j$ 中最右边的帧 $I_i$ 只找 `membership` 内元素在 $[i-W+1-l+1, i]$ 的高斯。

举例，对于 $W_6$ 也就是 $I_7, I_8, I_9$ 这个组合，在 $l=5$ 的情况下有：
1. 独立优化帧9: 拿到 $I_9$，它的 pose 是脏的，记做 $P_9^{coarse}$。
   1. 选高斯子集：根据全部高斯的 `membership` 属性，从$G_{global}$ 排除`membership`不含[3,9]的高斯，获得一个缩小版的子集 $G_{global} \rightarrow G_{global}^{sub}$。这个 $G_{global}^{sub}$ 应该是 $G_{global}$ 的一个引用，它只让$G_{global}$的一部分参与计算，但产生的优化和修改还得同步到 $G_{global}$上。
   2. pose 清洗：冻结全局高斯池子集的权重 $G_{global}^{sub} \rightarrow G_{global}^{sub-freeze}$，让 $P_9^{coarse}$ 经过 $\Phi_p$ 得到 $P_9^{pred}$，使用 $P_9^{pred}$ 以及 $G_{global}^{sub-freeze}$ 产生 $I_9^{pseudo}$，用 $I_9^{pseudo}$ 和 $I_9$ 产生渲染损失，传递梯度给 $\Phi_p$，优化 $N_p$ 次，最后让 $P_9^{coarse}$ 经过优化后的 $\Phi_p$ 得到一个干净的 pose $P_9^{refined}$。
   3. 高斯采样和优化：恢复全局高斯池子集的权重 $G_{global}^{sub-freeze} \rightarrow G_{global}^{sub}$，从 $I_9$ 上采样高斯得到 $G_9$，把 $G_9$ 加入全局高斯池 $ G_{global}^{sub} = G_{global}^{sub} + G_9$ ，使用 $P_9^{refined}$ 和 $G_{global}^{sub}$ 产生 $I_9^{pred}$，用 $I_9$ 和 $I_9^{pred}$ 产生渲染损失，传递梯度给 $G_{global}^{sub}$，优化 $C_9^{W_6} \cdot r_9^{front}$ 次，得到优化后的 $G_{global}^{sub}$。
2. 联合优化帧789：
   1. 用 $P_7^{refined}$ 和 $G_{global}^{sub}$ 产生 $I_{7}^{pred}$，用 $I_7^{pred}$ 和 $I_7$ 产生渲染损失，传递梯度给 $G_{global}^{sub}$，优化 $C_7^{W_6}$ 次。
   2. 用 $P_8^{refined}$ 和 $G_{global}^{sub}$ 产生 $I_{8}^{pred}$，用 $I_8^{pred}$ 和 $I_8$ 产生渲染损失，传递梯度给 $G_{global}^{sub}$，优化 $C_8^{W_6}$ 次。
   3. 用 $P_9^{refined}$ 和 $G_{global}^{sub}$ 产生 $I_{9}^{pred}$，用 $I_9^{pred}$ 和 $I_9$ 产生渲染损失，传递梯度给 $G_{global}^{sub}$，优化 $C_9^{W_6} \cdot r_9^{rear}$ 次。

---

## 1. 数学抽象（位姿、高斯池与模式 B/C）

**位姿**。第 \(i\) 帧观测为 \(\mathbf{I}_i\)。重基准后相对位姿 \(\mathbf{P}_i\in\mathrm{SE}(3)\)（`probe→world`）见 `docs/methods/baselines/ultrags_baseline/online_slam_design.md` §1.1。

- **模式 A**：\(\mathbf{P}_i^{\mathrm{refined}}=\mathbf{P}_i\)，无 \(\mathbf{P}^{\mathrm{coarse}}\)，无 \(\Phi_p\)。
- **模式 B**：\(\mathbf{P}_i^{\mathrm{coarse}}=\mathbf{P}_i^{\mathrm{noise}}\)（GT 加噪），\(\mathbf{P}_i^{\mathrm{refined}}=\Phi_p(\mathbf{P}_i^{\mathrm{coarse}})\)。
- **模式 C**：PEM \(\mathcal{M}_p\) 给出边 \(\mathbf{T}_{i-1\to i}^{\mathrm{pred}}=\mathcal{M}_p(\mathbf{I}_{i-1},\mathbf{I}_i)\)，链式 \(\mathbf{P}_i^{\mathrm{pred}}=\mathbf{T}_{i-1\to i}^{\mathrm{pred}}\mathbf{P}_{i-1}^{\mathrm{refined}}\)；取 \(\mathbf{P}_i^{\mathrm{coarse}}\leftarrow \mathbf{P}_i^{\mathrm{pred}}\)，再 \(\mathbf{P}_i^{\mathrm{refined}}=\Phi_p(\mathbf{P}_i^{\mathrm{coarse}})\)。

**PRM 与主路径**。统一写 \(\mathbf{I}_i^{\mathrm{render}}=\mathrm{Render}(\mathbf{P}_i^{\mathrm{refined}},\mathcal{G}_{\ast})\)。**Pose 辅助路径**（可选，对应 slam 文档 §2.2）在 \(\mathcal{G}_{\mathrm{exclude}\,i}\) 上渲染，使 \(\Phi_p\) 主要从「非本帧主归属」几何解释 \(\mathbf{I}_i\)，减轻与高斯争抢：

\[
\mathbf{I}_i^{\mathrm{aux}}=\mathrm{Render}(\mathbf{P}_i^{\mathrm{refined}},\mathcal{G}_{\mathrm{exclude}\,i}).
\]

正文中的「冻结 \(G_{\mathrm{global}}\) 只训 \(\Phi_p\)」即：**主高斯参数本步不更新**（实现上可对 \(G\) 清零梯度或跳过 optimizer step），仅对 \(\Phi_p\) 反传；数学上与「在 \(\mathcal{G}_{\mathrm{exclude}\,i}\) 或冻结集合上构造监督」同目的，细节以实现为准。

**全局高斯与归属**。记全局高斯参数集合为 \(\mathcal{G}\)（文中 \(G_{\mathrm{global}}\)）。每点 \(j\) 有多槽 `membership` \(\mathcal{M}_j\subset\{0,\ldots,N-1\}\)（至多 \(N_m\) 个帧 id，不足填 \(-1\)）。**出生**：新点 \(j\) 自帧 \(i\) 播种时 \(\mathcal{M}_j\leftarrow\{i\}\)。**共视更新**：渲染返回「本帧 \(i\) 实际贡献梯度/覆盖」的点集 \(\mathcal{S}_i\)，对 \(j\in\mathcal{S}_i\) 将 \(i\) 并入 \(\mathcal{M}_j\)（FIFO 满则移位），对应正文「`render` 后更新 membership」。

**符号**（与上文叙事一致）：

\[
G_i = \{\, j : \mathcal{M}_j=\{i\}\,\},\quad
G_{i_1\cdots i_r}=\{\, j : \mathcal{M}_j \text{ 恰含 } i_1,\ldots,i_r \,\}.
\]

**单窗内 front / rear 拆分**。对窗口 \(W_j\) 内各帧 \(i\)，将「独立帧优化」与「窗内联合 BA」的步数预算拆开：

\[
N_i^{\mathrm{front}} = \bigl\lfloor C_i^{W_j}\, r_i^{\mathrm{front}} \bigr\rfloor,\qquad
N_i^{\mathrm{rear}} = \bigl\lfloor C_i^{W_j}\, r_i^{\mathrm{rear}} \bigr\rfloor,\qquad
r_i^{\mathrm{front}}+r_i^{\mathrm{rear}}=1.
\]

联合阶段可对 \((\mathbf{P},\mathcal{G},\mathbf{I},T)\) 做随机调度（mini-batch 抽帧/抽损失项），不必严格按 \(0\to1\to2\) 顺序。

---

## 2. Local Window Manager（LWM）

**规格以 `docs/methods/baselines/ultrags_baseline/online_update_spec.md` 为准**（该文 §「数学原理」+ §「架构设计」）。

- **预算**：\(C_i^{W_j}\)、\(C_i^{all}\) 及与 \(r_i^{\mathrm{front}},r_i^{\mathrm{rear}}\) 组合成 \(K_i^{\mathrm{front}},K_i^{\mathrm{rear}}\) 的公式，均在 `docs/methods/baselines/ultrags_baseline/online_update_spec.md` 中给出，**不**以 `iters_per_frame` 为唯一来源。
- **接口**：`initialize(cfg)` 预计算 \(\mathcal{C}_{i,j}=C_i^{W_j}\)；`get_optim_steps(frame_id, window_id)` 查表；`update()` 维护当前 `window_id` 与流式帧 \(t\) 对齐方式。
- **与 OGWM 现有实现**：`online_gaussian_window_manager.py` 中的 `settle`、`frame_budget` EMA、`get_frame_weight` 可作为 **辅助**（时间窗 active、启发式 \(w_i\)）；**硬预算步数**应对齐 `get_optim_steps`，见 `docs/methods/baselines/ultrags_baseline/online_update_spec.md` 文末「与现有代码」说明。

**窗口集合**：处理到第 \(t\) 帧时 \(\mathcal{W}_t=\{\max(0,t-W+1),\ldots,t\}\)，与叙事中「窗 \(W_j\) 覆盖 \(\{j,\ldots,j+W-1\}\)」在 \(t=j+W-1\) 时一致。

---

## 3. Gaussian State Manager（GSM）

**规格以 `docs/methods/baselines/ultrags_baseline/online_update_spec.md` §「Gaussian State Manager」为准**。

- **`memberships`** \((G, N_m)\)、出生与共视更新、\(\mathcal{G}_{\mathrm{global}}^{\mathrm{sub}}\) 由 membership 与时间区间筛选。
- **梯度屏蔽**、与 LWM 的 **只读** 双向契约（`window_id`、窗内帧列表、`active_mask` 等）。
- **Pose 辅助**：\(\mathcal{G}_{\mathrm{exclude}\,i}=\{g:\texttt{owner}[g]\ne i\}\)（`get_exclude_indices`）与 **子集 \(\mathcal{G}_{\mathrm{sub}}\)** 是不同查询，需并存；与 `docs/methods/baselines/ultrags_baseline/online_slam_design.md` §2.2 一致。

实现上可与 LWM 合并为 `OnlineGaussianWindowManager`，逻辑上仍按 GSM 分区。

---

## 4. Pipeline

**规格以 `docs/methods/baselines/ultrags_baseline/online_update_spec.md` §「Pipeline」为准**：在 **window 本位** 下将本文 §0 叙事拆为 **`WINDOW_SETTLE` / `LWM_SYNC` / `POSE_WASH` / `CONCAT_SEED` / `GS_INDEPENDENT` / `GS_JOINT` / `SEEN_UPDATE` / `MERGE`** 等原子；各 **`GS_*`** 的迭代次数由 **`get_optim_steps`（或 `get_optim_steps_split`）** 给出，对应 \(C_i^{W_j}\) 与 \(r^{\mathrm{front}}/r^{\mathrm{rear}}\)。

**与当前插件链（过渡）**：`online_callbacks.py` 中 `settle_and_pose` → `concat` → `optimize` → `seen_update` → `merge` → `log_eval_save` 是占位实现；应将 **`optimize`** 拆成独立帧与联合窗内两段，并用 **`get_optim_steps`** 替换固定 `iters_per_frame`，才能与 `docs/methods/baselines/ultrags_baseline/online_update_spec.md` 一致。

**模式 A/B/C**：与 `docs/methods/baselines/ultrags_baseline/online_slam_design.md` §4.0 一致；**A** 无 `POSE_WASH`；**B/C** 有 \(\Phi_p\)，粗姿态来源见 §1。

---

## 5. Gaussian Merge Manager（GMM）与局部子集 \(G_{\mathrm{global}}^{\mathrm{sub}}\)

**GMM**（`gaussian_merge_manager.py` + `merge` 插件）：在 \(\mathcal{G}\) 上做去冗余合并；**不**替代 LWM 的窗口逻辑。合并后必须 `sync_state(remap, ogwm)` 以统一索引与 OGWM 状态。

**\(\mathcal{G}_{\mathrm{global}}^{\mathrm{sub}}\) 由 GSM 按 `memberships` 筛选**（公式与实现要点见 `docs/methods/baselines/ultrags_baseline/online_update_spec.md` §「Gaussian State Manager」第 3 条）。

---

## 6. 协作关系小结

| 组件 | 回答的问题 |
|------|------------|
| **LWM** | 查表 \(\mathcal{C}_{i,j}=C_i^{W_j}\)；`get_optim_steps`；`update(window_id)`；与 \(K_i^{\mathrm{front}},K_i^{\mathrm{rear}}\) 对齐 |
| **GSM** | `memberships`、\(\mathcal{G}_{\mathrm{sub}}\)、exclude、梯度屏蔽；与 LWM 只读契约 |
| **Pipeline** | 原子序列见 `docs/methods/baselines/ultrags_baseline/online_update_spec.md` §「Pipeline」；当前插件链为过渡实现 |
| **GMM** | 全局点数与冗余；与窗口正交 |
| **模式 B/C** | 粗姿态来源不同；\(\Phi_p\) 与 slam 文档 §3 位姿损失共用一套接口 |

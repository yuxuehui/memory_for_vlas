2026-07-20 · 更新 2026-07-22

Status:

Tags: probe, keyframe-selection, patch-memory, tokendrop

# Patch 级 memory 选择 probe

**问题**：按 **patch**（不是整帧）选 512-token memory，选择信号该用什么？

**口径**：教师强制 stride-8 前向，front+wrist 各 9×9 = **162 cell/步**；竞争性 top-512 heap。通道：`nov`（pixel diff + frame-0 front sentinel）· `act`（DiT action→patch，L13）· `tail_L13/L15`（backbone post-image summary → patch）。GT 来自 `vlm_keyframe_labels/`。指标统一用**不同 cell 数**（上界 162）——同一格子在不同时刻是不同内容，所以集中 ≠ 冗余。

## 1. 通道组合方式：union（`gen_patch_labels.py`，20 个 model×episode 均值）

| 方法 | 不同 cell 数 | steps | demo_share |
|---|---:|---:|---:|
| td_diff | **145** | 38.1 | 0.300 |
| attn_L5 / L10 / L13 | 42 / 60 / 55 | **48.5** | 0.276–0.280 |
| **split（union 256+256）** | 130 | 47.6 | **0.308** |
| sum（z-score 相加） | 82 | 45.8 | 0.270 |

- **union 是唯一三项都接近最优的**；两个单通道各在一项上崩（diff 的 steps 38.1、attn 的 cell 数 42–60）。
- **sum 输**：z 标准化后 attention 尾巴更重 → 加和被 attention 主导，且丢 frame-0 锚点（span 从 t=16–32 才开始）。
- attention **无** demo 盲区（VPB 16/16 ref 全覆盖）；两通道失败模式互补（attn = 少 cell × 多 step，diff = 多 cell × 少 step）。
- coverage 指标在 512 预算下**饱和** —— 判别力在"存了什么"而非"事件旁有没有 cell"。
- 长 episode 是**变薄不是耗尽**（VPB T=1048 仍覆盖 120+ 步，早期 patch 活到结尾）。

## 2. ⚠️ 核心教训：选择必须 trained-in（eval-only 3-way，2026-07-20）

同一 exp-d@60k ckpt（训练分布 = acausal linspace 整帧），仅换 eval 侧选择（16 任务 × 50 eps）：

| | fifo（基线） | diff | patch_union |
|---|---:|---:|---:|
| **OVERALL** | **12.4** | 8.8 | 7.4 |

**性能排序 = 与训练 mem_seq 分布的距离排序**（"8 个完整近期帧" > "8 个完整事件帧" > "散点 patch 汤"）。**probe 代理不可信**：patch_union 的 probe 指标最好之一，eval 却最差。任何选择规则的改动都必须进训练才算数。

## 3. act vs tail：三模型对照（`gen_act_vs_tail_patches.py`，5 任务 × 3 模型）

三模型构成 note-22 的消费性对照：`vanilla`（`hamlet_mode=off`，无 moment token）· `exp-d`（moment 被 framesamp 顶替 = 不被消费）· `HAMLET`（moment 被消费）。

> ⚠️ 陷阱：三个 config 里都躺着残留字段 `n_moment_tokens=4`，真正的开关是 **`hamlet_mode`**；vanilla 按 4 砍会吃掉 4 个真 tail token。

**wrist 占比**（选中 patch 来自腕视角的比例）：

| 通道 | vanilla | exp-d | HAMLET | 读法 |
|---|---:|---:|---:|---|
| act_L13 | 0.26 | 0.39 | 0.34 | **front 偏好**，三模型一致 = 看"要去哪" |
| **tail_L13** | **0.44** | **0.41** | **0.78** | 两个无消费瓶颈的模型挤在一起，只有 HAMLET 跳走 |
| tail_L15 | 0.72 | 0.60 | 0.64 | 同向；微调把它往 front 拉 0.12，非完全模型无关 |
| nov | 0.71 | 0.71 | 0.71 | 逐像素，不过模型 |

**note-22"被消费瓶颈"的独立复核**（那次测注意力质量，这次测"最终选了哪些 patch"），三模型呈干净的**双水平**：tail_L13 与 act 的重叠在两个无消费模型上是 **0.311 / 0.311**，HAMLET 上是 **0.171**。图 `acttail_L13_circuit_ButtonUnmask.png`：HAMLET 腕视角橙框压在被操作物体上、front 行几乎全蓝；vanilla/exp-d 蓝橙混杂。

**通道重叠 @512**（越低越互补）：

| 配对 | vanilla | exp-d | HAMLET |
|---|---:|---:|---:|
| act ∩ tail_L13 | 0.311 | 0.311 | **0.171** |
| act ∩ tail_L15 | 0.230 | 0.336 | 0.224 |
| act ∩ nov | 0.162 | 0.153 | 0.125 |
| tail_L15 ∩ nov | 0.200 | 0.199 | 0.205 |

三通道两两重叠 ≤0.34（比 VLA-Pruner 的 ~0.5 更互补），**novelty 与两个 attention 通道最正交（0.13–0.21，三模型一致）**。

## 4. union 变体等预算对照（round-robin 回填到恰好 512）

| 变体 | 不同 cell 数↑ | demo↑ | steps↑ | wrist | REF |
|---|---:|---:|---:|---:|---:|
| nov∪act（已部署） | 131 / 131 / 130 | 0.387 / 0.384 / 0.377 | 51.4 / 51.0 / 48.6 | 0.48 / 0.55 / 0.50 | 全满 |
| **nov∪tail_L15** | 131 / 130 / **133** | **0.387 / 0.389 / 0.388** | 50.8 / 51.2 / 50.4 | 0.70 / 0.62 / 0.63 | 全满 |
| nov∪act∪tail_L15 | 124 / 124 / 126 | 0.383 / 0.382 / 0.385 | 51.6 / 51.6 / 51.2 | 0.57 / 0.56 / 0.50 | HAMLET **21/22** |
| nov∪act∪tail_L13 | 124 / 124 / 123 | 0.386 / 0.376 / 0.385 | 49.4 / 51.0 / 51.8 | 0.39 / 0.45 / 0.55 | 全满 |
| f0front∪act∪tail_L15 | 112 / 109 / 112 | 0.392 / 0.385 / 0.388 | 52.0 / 52.0 / 51.4 | 0.49 / 0.48 / 0.43 | HAMLET **21/22** |

（每格 `vanilla / exp-d / HAMLET`；两次独立重跑差 ≤0.01 = denoise 抖动，"噪声级"以此为标尺）

**① 三路不值**：多花 1/3 预算，cell 数反而从 130–133 掉到 123–126，demo/steps 持平。attention 通道空间集中（46–60 格 vs novelty 145 格），配额砍到 ~171 后贡献的多样性更低 —— 等于拿 novelty 的有效 slot 换 attention 的重复 slot。

**② frame-0 锚点替代不了 novelty**：锚点管 t=0，novelty 管 t>0 的事件（VPB 的 16 个 ref 事件散布在整段 demo 里）；且**锚点本来就在 novelty 通道里**（frame-0 sentinel），去掉 novelty 等于砍掉变化检测只留锚点。cell 数也更差（109–112）。锚点本身值得留（front 81 格 ≈ 16% 预算，TokenDrop sentinel 已有）。

**③ REF 证据**（弱，单事件级）：唯一的非满分点是同一个 HAMLET/VPB reference 事件，被**每一个 act+tail_L15 组合**丢掉（含 1/3 novelty 的三路）；novelty 份额 ≥1/2 的配置在三模型上从未丢过。novelty 该留的硬理由是正交性、cell 多样性和 demo 份额。

**④ nov∪tail_L15 vs nov∪act：probe 级打平**（差异全在噪声内）。但 tail 侧 **prefill 免费**（省两遍前向 + post-action 延迟）—— 打平就足以让它值得一训。**下一个 trained-in 变体选它，不是三路。**

**逐帧看**（HTML 每页 "Union comparison" 区 + `union_vanilla_*.png`，框色 = 买单通道）：novelty（红）在腕视角吃掉大部分 slot、压在被移动物体上；`nov+act` 的蓝框在前视角机械臂接近区，`nov+tail_L15` 的橙框在腕视角夹爪接触点 —— 与 wrist 表一致。三路是同一画面被三色摊薄，新增框大多落在已被红框覆盖的物体上。VPB（T=1048）三个变体的 12 列都从 t=0 铺到 t=1040，是"heap 竞争性而非消耗性"的画面证据。

**history 增删对比**（`cmp_navt_<task>.html`，model pu50k，第三行 DELTA：绿=tail_L15 插入 / 灰=被挤掉）：加 tail_L15 换掉约 **1/4 预算**（VPB 145/512、SwingXtimes 132、PatternLock 114）。**tail 插入的绿框系统性落在腕视角的 target 靶环 + 被操作物体上**——这些是 act 漏掉的**任务地标**；被挤掉的灰框多在 act 堆积的前视角机械臂区。即 tail_L15 把预算从"动作区特写"拉向"场景地标"，正是 §6 双探针诊断的"选择太集中"的对症方向。⚠️ 但这在等预算下 cell 数仍降（§4①），是否净收益要 trained-in eval 定。

## 5. 时间结构：永久内核 + 流动外围（`temporal_persistence.py`）

`overlap(lag) = |A_i ∩ A_{i+lag}| / |A_i|`，A = 该步分数前 K 的格子集合。**lag 单位是 scored step = 8 个控制步**。top-12.5%（20/162 格）的随机基线 = **0.12**。

| 通道 (top-12.5%) | lag1 | lag2 | lag4 | lag8 |
|---|---:|---:|---:|---:|
| act_L13 | 0.74 | 0.64 | 0.56 | 0.56 |
| tail_L15 | 0.76 | 0.66 | 0.55 | 0.53 |
| **nov** | **0.47** | **0.33** | **0.29** | **0.31** |

1. **两个 attention 通道 = 永久内核 + 流动外围**：衰减 2–4 步内停住后走平，约 **56% 的 top-K 是永久内核**（远高于随机 0.12），约 18% 几步内换掉。
2. **novelty 的内核只有一半（0.29–0.31）**，主体事件驱动 ⇒ 两类通道的**互补也在时间维度**（此前只知空间重叠 0.13–0.21）。
3. **EMA 残差方案不能无条件上**：残差在事件型信号上保住每次 onset，在持续型上只触发一次；attention 的 56% 永久内核会被整个抹掉，对 Counting（重复事件本身就是信息）有风险。可行形式 = **只对内核 cell（衰减走平后仍在集合里的那批）施加软衰减**，外围不动。
4. 只有 top-12.5%/25% 有判别力：top-50% 的随机基线就 ~0.5，且背景 diff≈0 时排序退化（`temporal_persistence.txt` 有全部三档）。

## 6. act 是"被消费的"，但不是"被训练的选择器"

**梯度没有流到选择上**：pass A 打分是 `no_grad`，top-k 不可导 —— **分数 → 选择 → 未来损失**这条路是断的。act 的训练目标是"当前 16 步动作做对"，不是"选出以后会用到的证据"；它和 tail_L15 一样是**借来的代理**（一个借自动作目标、一个借自预训练视觉摘要）。

症状：act 三模型都 front 偏好（0.26–0.39），而证据在腕视角（nov 0.71、HAMLET 训练出的读出 0.78）；**vanilla 的 act 最极端（0.26）**——从没被要求记住任何东西的模型，动作注意力最偏"要去哪"。

要真正训练选择器：① straight-through / Gumbel top-k 让未来动作损失回流到分数；② 用 **action→memory** 注意力（t′ 时刻 DiT 实际读了哪些 memory token）作为 t 时刻写入决策的 hindsight 信号；③ 保留一个被消费的 moment 通道（note-22 路线）。

## 下一步

- [ ] trained-in **`nov∪tail_L15`**（patch_union@60k 之后；L15、prefill 免费 → 去掉两遍前向）
- [ ] **hindsight 写入信号 probe**：action→**memory** 注意力 vs 写入时的 act/tail/nov 分数 —— "当场代理选的"和"事后真用的"重叠多少；只需把捕获的取列换成 memory 列
- [ ] 部署侧回填修复：`fs_patch_union.read()` 缺口目前用**当前帧 token** 补齐（重复 KV 已有内容），应沿通道排名回填历史 patch
- [ ] 软 refractory（只对内核 cell 衰减，§5.3）+ VLA-Pruner 的 **min-redundancy 过滤**（他们 union 之后按 mRMR 丢冗余，我们只有 Combine 没有 Filter）
- [ ] masking 层选择测试 + sink 剔除 —— 针对 act 的 DiT 层（tail 侧已定 framesamp 系用 **L15**）
- [ ] 架构缺口（最大）：存的 patch 仍无 **(t,y,x) pos-emb + state 标签**（note-16 v2b 教训）

## 文件

**脚本/绘图函数总览见 [SCRIPTS.md](SCRIPTS.md)**（每个 probe 脚本的作用、输入输出、重跑命令）。

- **主对照**：`acttail_<Task>_<model>.html` ×15（5 任务 × 3 模型，union 胶片按买单通道上色）· `acttail_summary.csv` · `acttail_overlap.csv` · `temporal_persistence.txt` · `proxy.log`
- **图**：`acttail_L13_circuit_ButtonUnmask.png`（3 模型 × front/wrist）· `union_vanilla_{ButtonUnmask,VideoPlaceButton,SwingXtimes}.png` · `union_f0_act_tail_*`（负结果）· `cmp_navt_*.html` ×5（history 增删）· `specprune_*.html` ×5（SpecPrune 移植）
- **早期**：`summary.csv` + `gen_patch_labels.py`（README §1 仍用）· `ep300_annotated/`（扩圈消融）· `relevance_ablation.py`（已被 `gen_act_vs_tail_patches.py` 取代）。⚠️ 旧的 2-模型 `<Task>_<model>.html` ×10 已删（被 3-模型 acttail 取代）
- 重跑：`cd HAMLET-Isaac-GR00T-N1d6 && CUDA_VISIBLE_DEVICES=<gpu> NO_ALBUMENTATIONS_UPDATE=1 .venv/bin/python .../<script>.py`

相关：[21_patch_union_memory.md](../21_patch_union_memory.md)（部署版方法）· [22_moment_readout_bottleneck.md](../22_moment_readout_bottleneck.md)（读出电路）· [20_tokendrop.md](../20_tokendrop.md)（帧级 diff）· [20_memory_theory.md](../20_memory_theory.md) §3.2（nov=innovation，attn=readout-relevance/VoI 代理）

# CoDA-CZSL 研究方案

**Conditional Disentanglement with Feasibility-Aware Augmentation for Compositional Zero-Shot Learning**

> 基于 CAMS (arXiv:2511.16378) 的延续性研究方案

---

## 0. 一句话故事

CAMS 把 attribute 和 object 投到三个独立空间里硬解耦,但解耦既没有显式信息论约束,也没考虑"red wine 的 red"与"red apple 的 red"在视觉上根本不是同一个东西。我们提出 **CoDA-CZSL**,用 **object 条件化的属性提取**来建模上下文依赖,用 **互信息上界(CLUB)** 显式强制属性—对象表征独立,再用 **LLM 引导的可行性感知组合增强** 把开放世界场景从 4.8% AUC 拉上来。

---

## 1. 研究动机

### 1.1 CAMS 的三大局限

| 问题 | 具体表现 | 证据 |
|---|---|---|
| **解耦无显式约束** | f^a 与 f^o 来自同一组 latent units,只靠 4 个 CE loss 隐式分离 | 论文 4.2 节,损失只有 L_a, L_o, L_c, L_g |
| **属性被当作上下文无关概念** | "caramelized" 在 tomato/chocolate 上视觉差异巨大,但用同一个 t^a 对齐 | Fig.6 失败案例:`Caramelized Tomato → Caramelized Vegetable` |
| **OW 性能塌陷** | C-GQA: CW AUC 17.4% → OW AUC 4.8%,MIT-States: 24.2% → 9.1% | Tab.1 vs Tab.2 |

### 1.2 我们的核心洞察

1. **属性是上下文依赖的(context-dependent)**。视觉上 "red wine" 的 red 是深红半透明液体,"red apple" 的 red 是表面反光固体,共享一个属性 embedding 是表达力瓶颈。
2. **解耦需要显式量化目标**。CE loss 只能让分类正确,不能保证 f^a ⊥ f^o。需要直接最小化 I(f^a; f^o)。
3. **OW 的核心问题是"无效组合干扰"**。`Yt = A × O` 里大量组合是物理上不可能的(`sliced sky`、`rusty water`),模型却必须对它们打分,引入噪声。需要 feasibility prior。

---

## 2. 方法概览

### 2.1 与 CAMS 的对比

```
CAMS:
  Image → CLIP_v → Latent Units (Q̃) → MSD(Trs_a, Trs_o, Trs_c) → f^a, f^o, f^c
                                              ↓
                                       CE alignment with t^a, t^o, t^c

CoDA-CZSL (本方案):
  Image → CLIP_v → Latent Units (Q̃)
            │
            ├──→ Object Stream:  f^o (with InfoNCE align t^o)
            │                       │
            │                       └──(条件化)──┐
            │                                     ↓
            ├──→ Attribute Stream: OCAD(Q̃ | f^o) → f^a (align t^a)
            │
            └──→ Composition Stream: f^c (align t^c)

  + L_MI = CLUB(f^a, f^o)              ← 显式解耦
  + L_aug = FACA(synthetic compositions) ← 开放世界增强
```

### 2.2 三个核心模块

- **OCAD** (Object-Conditioned Attribute Decoder):把对象特征作为条件输入,动态调制属性提取过程
- **MID** (Mutual Information Disentanglement):用 CLUB 估计并最小化 I(f^a; f^o)
- **FACA** (Feasibility-Aware Compositional Augmentation):LLM 评分 + 特征级合成,生成可信的 unseen compositions 用于训练

---

## 3. 方法详解

### 3.1 基础符号

沿用 CAMS:

- 图像 `x`,经 CLIP 图像编码器低层得到 `H_0 ∈ R^{P×d_v}`,高层 M 层 + GCA 得到 latent units `Q̃ ∈ R^{K×d_v}`
- 属性集 `A = {a_i}`,对象集 `O = {o_j}`,组合 `Y_s` (seen) ∪ `Y_u` (unseen)
- prompt 表征 `t^a_i, t^o_j, t^c_{i,j} ∈ R^d`(来自 CLIP 文本编码器)

新增:

- 对象特征 `f^o ∈ R^d`,属性特征 `f^a ∈ R^d`,组合特征 `f^c ∈ R^d`
- CLUB 估计网络 `q_φ(f^a | f^o)`

### 3.2 Module 1: Object-Conditioned Attribute Decoder (OCAD)

**动机**:属性的视觉表现依赖于对象类别。我们用 hypernetwork 的思路,让 object 特征 modulate attribute 提取器。

**步骤 A — 先提取对象表征**:

复用 CAMS 的 Trs_o 分支:

```
Q̃_o = Trs_o(Q̃)
f^o  = π_o(Mean(Q̃_o))
```

**步骤 B — 对象条件化调制**:

把 `f^o` 投影成 FiLM 风格的 (γ, β) 参数,作用在属性 latent units 上:

```
γ, β = MLP_film(f^o)         # γ, β ∈ R^{d_v}
Q̃_a_cond = γ ⊙ LayerNorm(Q̃) + β
```

然后送入属性 Transformer:

```
Q̃_a = Trs_a(Q̃_a_cond)
f^a  = π_a(Mean(Q̃_a))
```

**直觉**:同样一组 latent units,当对象是 "wine" 时,γ/β 会放大与液体、反光、深色相关的通道;当对象是 "apple" 时,会放大表面纹理通道。

**计算开销**:仅多了一个 2-layer MLP_film,几乎不增加显存。

---

### 3.3 Module 2: Mutual Information Disentanglement (MID)

**目标**:最小化 `I(f^a; f^o)`,保证两个表征在统计上独立(超出 CE loss 能保证的范畴)。

**方法选择**:用 **CLUB** (Cheng et al., ICML 2020) 估计 MI 上界,然后最小化。CLUB 对高维连续随机变量稳定。

**CLUB 估计网络**:

```python
class CLUB(nn.Module):
    """
    估计 I(f^a; f^o) 的上界
    q_phi(f^a | f^o) 用一个 MLP 参数化为高斯
    """
    def __init__(self, dim):
        super().__init__()
        self.mu_net     = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.logvar_net = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim),
                                        nn.Tanh())  # Tanh 防止 logvar 爆炸

    def forward(self, f_a, f_o):
        # f_a, f_o: [B, d]
        mu     = self.mu_net(f_o)
        logvar = self.logvar_net(f_o)

        # log q(f_a | f_o) for positive pairs
        pos = -((mu - f_a) ** 2) / 2 / logvar.exp() - logvar / 2

        # log q(f_a' | f_o) for negative pairs (shuffle f_a in batch)
        # CLUB 上界 = E[log q(pos)] - E[log q(neg)]
        f_a_shuf = f_a[torch.randperm(f_a.size(0))]
        neg = -((mu - f_a_shuf) ** 2) / 2 / logvar.exp() - logvar / 2

        return (pos.sum(dim=-1) - neg.sum(dim=-1)).mean()

    def learning_loss(self, f_a, f_o):
        """训练 q_phi 自身(最大化 log-likelihood)"""
        mu     = self.mu_net(f_o)
        logvar = self.logvar_net(f_o)
        return (((mu - f_a) ** 2) / 2 / logvar.exp() + logvar / 2).sum(dim=-1).mean()
```

**训练循环**(关键):CLUB 需要交替训练

```
每个 iteration:
  1. 固定主网络,更新 CLUB 的 q_phi(最大化 likelihood)
  2. 固定 q_phi,主网络优化 L_total = L_CE + λ_MI * I_upper(f^a, f^o) + ...
```

**为什么不用 InfoNCE 下界?** 我们要**最小化** MI,不是最大化,因此需要上界估计器。

---

### 3.4 Module 3: Feasibility-Aware Compositional Augmentation (FACA)

**目标**:解决 OW 性能塌陷。

#### 3.4.1 用 LLM 预计算 feasibility 矩阵

对每个候选组合 `(a_i, o_j) ∈ A × O`,用 LLM(如 Qwen2.5-72B / GPT-4o-mini)预先离线评分:

```
Prompt: "Rate the physical plausibility of the composition '{attribute} {object}'
on a scale from 0 to 1, where 0 = impossible (e.g., 'sliced sky'),
1 = common (e.g., 'red apple'). Only output a number."
```

得到 feasibility matrix `F ∈ [0,1]^{|A|×|O|}`。这一步**只需做一次**,缓存为 `.pt` 文件。

**优势**:相比 CGE / CLUSPRO 用 word2vec 余弦相似度估算 feasibility,LLM 显著更准。

#### 3.4.2 特征级组合合成

训练中,从可行(F > τ_high)但未出现在训练集的组合 `(a, o) ∉ Y_s` 中采样,合成伪样本:

```python
# 维护两个 memory bank
bank_attr = {a: deque(maxlen=128)}  # 每个属性的 f^a 池
bank_obj  = {o: deque(maxlen=128)}  # 每个对象的 f^o 池

# 训练中,每个 batch 后更新 bank
bank_attr[gt_attr].append(f_a.detach())
bank_obj[gt_obj].append(f_o.detach())

# 合成阶段(每 N 个 batch 触发一次)
for _ in range(n_synth):
    (a, o) = sample_feasible_unseen()           # F[a,o] > τ_high, 且 (a,o) ∉ Y_s
    f_a_syn = random.choice(bank_attr[a])
    f_o_syn = random.choice(bank_obj[o])
    f_c_syn = fuse(f_a_syn, f_o_syn)            # 简单 MLP 融合或拼接
    # 拿这个伪 f_c_syn 去对齐 t^c_{a,o}
    L_aug += CE(f_c_syn, t^c_{a,o}) * F[a, o]   # feasibility 加权
```

#### 3.4.3 可行性加权的开放世界推理

推理时,把 F 作为 logit 偏置:

```
score(a, o | x) = β·p(g_{a,o}|x) + (1-β)·p(c_{a,o}|x) + p(a|x)·p(o|x)
                + γ · log(F[a,o] + ε)            ← 新增
```

`γ` 控制 feasibility 的影响。这个一行改动在 OW 上对 AUC 有显著提升(参考 OADis、KG-SP 的相关结论)。

---

### 3.5 总损失

```
L_total = L_g + α_c·L_c + α_a·L_a + α_o·L_o          # CAMS 原损失
        + λ_MI · I_upper(f^a, f^o)                    # MI 解耦
        + λ_aug · L_aug                               # 合成增强
        + λ_orth · ||W_a^T W_o||_F^2                  # 投影头正交正则(轻量稳定项)
```

**推荐初始权重**:`α_a=α_o=0.5, α_c=α_g=1`(同 CAMS),`λ_MI=0.1`, `λ_aug=0.3`, `λ_orth=0.01`。

---

## 4. 算法伪代码

```python
# ===== 训练一个 iteration =====
def train_step(batch, model, club, optimizer_main, optimizer_club, feasibility_matrix):
    x, y_a, y_o, y_c = batch

    # ---------- Step 1: 主网络前向 ----------
    H_i = clip_v_low(x)                          # 低层 visual features
    Q_tilde = GCA(H_i, latent_units)             # CAMS 的 Gated Cross-Attention

    # 先算 object,再用它条件化 attribute
    Q_o = Trs_o(Q_tilde)
    f_o = pi_o(Q_o.mean(dim=1))

    gamma, beta = MLP_film(f_o)                  # FiLM 调制参数
    Q_a_cond = gamma.unsqueeze(1) * LayerNorm(Q_tilde) + beta.unsqueeze(1)
    Q_a = Trs_a(Q_a_cond)
    f_a = pi_a(Q_a.mean(dim=1))

    Q_c = Trs_c(Q_tilde)
    f_c = pi_c(Q_c.mean(dim=1))

    f_g = mlp_g(global_clip_feat(x))             # CAMS 的 global 分支

    # ---------- Step 2: 更新 CLUB(估计器自身) ----------
    optimizer_club.zero_grad()
    loss_club_learn = club.learning_loss(f_a.detach(), f_o.detach())
    loss_club_learn.backward()
    optimizer_club.step()

    # ---------- Step 3: 主损失 ----------
    optimizer_main.zero_grad()
    L_g = CE(f_g @ t_c.T / tau, y_c)
    L_c = CE(f_c @ t_c.T / tau, y_c)
    L_a = CE(f_a @ t_a.T / tau, y_a)
    L_o = CE(f_o @ t_o.T / tau, y_o)

    L_MI = club(f_a, f_o)                        # 上界,要最小化
    L_orth = (pi_a.weight @ pi_o.weight.T).pow(2).sum()

    # 合成增强(每 N 步触发一次,这里写在主 loss 里)
    L_aug = compositional_augmentation(model, bank_attr, bank_obj, feasibility_matrix)

    L_total = (L_g + alpha_c * L_c + alpha_a * L_a + alpha_o * L_o
               + lambda_MI * L_MI + lambda_aug * L_aug + lambda_orth * L_orth)
    L_total.backward()
    optimizer_main.step()

    # ---------- Step 4: 更新 memory bank ----------
    update_bank(bank_attr, bank_obj, f_a, f_o, y_a, y_o)

# ===== 推理 =====
def infer(x, model, feasibility_matrix, gamma_f=0.3):
    f_a, f_o, f_c, f_g = model(x)
    p_g  = softmax(f_g @ t_c.T / tau)            # over Y_s (CW) or A×O (OW)
    p_c  = softmax(f_c @ t_c.T / tau)
    p_a  = softmax(f_a @ t_a.T / tau)
    p_o  = softmax(f_o @ t_o.T / tau)

    score = (beta * p_g + (1 - beta) * p_c
             + outer(p_a, p_o).flatten()
             + gamma_f * log(feasibility_matrix.flatten() + 1e-6))    # OW only
    return score.argmax()
```

---

## 5. 实现细节(基于 CAMS 代码改动)

CAMS 仓库结构基于 Troika。预计你 fork 的目录长这样:

```
CAMS/
├── flags.py                  # 配置入口
├── models/
│   ├── cams.py              # 主模型 (改这个)
│   ├── gated_cross_attn.py  # GCA 模块 (基本不动)
│   └── multi_space.py       # MSD 模块 (改这个)
├── losses/                   # 新增
│   ├── club.py              # 新增:CLUB 估计器
│   └── augmentation.py      # 新增:FACA
├── utils/
│   ├── feasibility.py       # 新增:LLM 评分 + 矩阵加载
│   └── memory_bank.py       # 新增
├── data/
│   └── ...                  # 数据集
└── train.py                  # 训练入口 (改这个)
```

### 5.1 关键代码改动清单

| 文件 | 改动 |
|---|---|
| `models/multi_space.py` | 加 FiLM 调制层;先 forward Trs_o 再 forward Trs_a |
| `losses/club.py` | 新建,见 §3.3 代码 |
| `losses/augmentation.py` | 新建,见 §3.4.2 |
| `utils/feasibility.py` | 新建,离线脚本调用 LLM API 生成 `feasibility_{dataset}.pt` |
| `utils/memory_bank.py` | 新建,EMA 更新的 attribute/object feature pool |
| `train.py` | 加 `optimizer_club`,双循环交替更新;每 N 步触发 FACA |
| `inference.py` | OW 推理时加 feasibility 加权项 |
| `flags.py` | 加超参:`lambda_MI`, `lambda_aug`, `lambda_orth`, `gamma_feas`, `tau_high`, `n_synth_per_batch` |

### 5.2 离线 feasibility 生成脚本

```python
# scripts/compute_feasibility.py
import torch, json
from openai import OpenAI    # 或换成 vLLM 本地推理 Qwen

client = OpenAI()

def query(attr, obj):
    prompt = (f"Rate the physical plausibility of '{attr} {obj}' "
              f"from 0 (impossible) to 1 (common). Output only a number.")
    r = client.chat.completions.create(
        model="gpt-4o-mini", temperature=0,
        messages=[{"role":"user","content":prompt}])
    try:    return float(r.choices[0].message.content.strip())
    except: return 0.5

attrs = json.load(open("data/mit-states/attributes.json"))
objs  = json.load(open("data/mit-states/objects.json"))
F = torch.zeros(len(attrs), len(objs))
for i, a in enumerate(attrs):
    for j, o in enumerate(objs):
        F[i, j] = query(a, o)
    torch.save(F, "feasibility_mit.pt")  # 中途保存,防意外中断
```

**预算估算**:
- MIT-States: 115 × 245 = 28,175 调用,GPT-4o-mini 约 $3-5
- C-GQA: 413 × 674 = 278,362 调用,约 $30-50 → **强烈建议用 vLLM + Qwen2.5-32B 本地跑,免费**
- UT-Zappos: 16 × 12 = 192,可忽略

### 5.3 推荐超参起点

| 数据集 | λ_MI | λ_aug | λ_orth | γ_feas (OW) | τ_high | n_synth/batch |
|---|---|---|---|---|---|---|
| MIT-States | 0.1 | 0.3 | 0.01 | 0.3 | 0.7 | 8 |
| UT-Zappos | 0.05 | 0.2 | 0.01 | 0.1 | 0.8 | 4 |
| C-GQA | 0.15 | 0.5 | 0.01 | 0.4 | 0.6 | 16 |

UT-Zappos 因为属性—对象耦合天然就紧(都是鞋类),λ_MI 调小;C-GQA OW 空间巨大,λ_aug 调大。

---

## 6. 实验设计

### 6.1 数据集与指标

完全沿用 CAMS:**MIT-States / UT-Zappos / C-GQA**,指标 **S / U / HM / AUC**,CW 和 OW 两个设定。

### 6.2 对比方法

复现 CAMS 报告的所有 baseline:CSP、DFSP、CAILA、CDS-CZSL、Troika、PLID、RAPR、MSCI、Logic-CSP、Logic-Troika、CLUSPRO、CAMS 本身。**重点是把 CAMS 当作直接对照,所有改进都体现在 Δ 上**。

### 6.3 消融实验(必做)

| 消融配置 | 验证什么 |
|---|---|
| CAMS (复现) | 基线 |
| + OCAD (无 MID, 无 FACA) | 对象条件化是否单独有效 |
| + MID (无 OCAD, 无 FACA) | CLUB 解耦是否单独有效 |
| + FACA (无 OCAD, 无 MID) | OW 增强是否单独有效 |
| + OCAD + MID | 两个解耦机制是否互补 |
| + OCAD + MID + FACA (full) | 完整 CoDA |
| FACA 用 word2vec 余弦代替 LLM 评分 | feasibility 来源的重要性 |
| MID 用 InfoNCE 下界代替 CLUB | MI 估计器的选择 |
| OCAD 的 FiLM 换成简单 concat | 条件化方式的选择 |

### 6.4 必做可视化

1. **t-SNE 复刻 Fig.3**:CAMS vs CoDA 的 f^a / f^o 散点,展示更紧的类内 + 更远的类间
2. **MI estimate 曲线**:训练过程中 I(f^a; f^o) 随 epoch 下降的曲线,证明 MID 起作用
3. **属性—对象热图**:固定 attribute "red",对所有 object 计算条件化前后 γ 的余弦相似度矩阵,展示对象依赖性
4. **OW 错误分析**:画 feasibility-加权前后,top-5 错误组合的分布,展示 FACA 消除了多少"不可能组合"误报
5. **失败案例**:沿用 CAMS Fig.6 风格,展示哪些 case CoDA 救回来了

### 6.5 预期数字(目标 Δ)

基于 CAMS 当前数字保守估计:

| 数据集 | 设定 | CAMS | CoDA 目标 | Δ |
|---|---|---|---|---|
| MIT-States | CW AUC | 24.2 | **25.8–26.5** | +1.6 ~ +2.3 |
| UT-Zappos | CW AUC | 47.4 | **48.5–49.5** | +1.1 ~ +2.1 |
| C-GQA | CW AUC | 17.4 | **19.0–20.0** | +1.6 ~ +2.6 |
| MIT-States | **OW AUC** | 9.1 | **10.5–11.5** | +1.4 ~ +2.4 |
| UT-Zappos | OW AUC | 40.3 | **41.5–43.0** | +1.2 ~ +2.7 |
| **C-GQA** | **OW AUC** | 4.8 | **6.0–7.5** | **+1.2 ~ +2.7** |

C-GQA OW 是最大卖点——FACA 设计就是冲着这个场景去的。

---

## 7. 实施时间表(12 周)

| 周次 | 任务 | 交付物 |
|---|---|---|
| 1 | Fork CAMS,在本机复现论文 Tab.1 的 UT-Zappos 结果 | 复现日志 + AUC 数字误差 < 0.3 |
| 2 | 实现 OCAD(FiLM 模块),在 UT-Zappos 验证 | +OCAD ablation 结果 |
| 3 | 实现 CLUB 并跑通双 optimizer 循环 | MI 曲线下降验证 |
| 4 | OCAD + MID 联合在三个数据集 CW 跑通 | 主表 CW 列填充 |
| 5 | 跑 LLM feasibility 评分(C-GQA 用本地 vLLM) | `feasibility_*.pt` 文件 |
| 6 | 实现 memory bank + FACA | OW 设定下 ablation |
| 7 | 完整模型在三个数据集 CW + OW 全跑 | 主表完整 |
| 8 | 做完所有消融实验 | 消融表完整 |
| 9 | 跑可视化(t-SNE,MI 曲线,热图) | 所有 figure 草图 |
| 10 | 写论文初稿(introduction, related work, method) | 6–7 页 LaTeX |
| 11 | 写实验和分析部分,精修 figure | 完整初稿 |
| 12 | 内审、超参微调、补缺失实验 | 投稿版本 |

**关键里程碑**:第 4 周末必须看到 OCAD+MID 在至少一个数据集上有 > 1% AUC 提升,否则需要回退方案(见 §8)。

---

## 8. 风险与应对

| 风险 | 应对 |
|---|---|
| CLUB 训练不稳定,MI estimate 震荡 | (a) 加 logvar 的 tanh 限制;(b) 把 λ_MI 从 0.01 warmup 到 0.1;(c) 退化方案:换成简单的 Hilbert-Schmidt Independence Criterion (HSIC) |
| FACA 合成的伪样本质量差,反而拉低 CW 性能 | (a) feasibility 阈值 τ_high 调高到 0.8;(b) 只在 epoch > 5 后开启 FACA;(c) 用 EMA 更新的 momentum encoder 来生成更稳定的 bank 特征 |
| OCAD 让对象错误传播到属性 | (a) 训练前期(epoch < 3)用 ground-truth object embedding 做 teacher forcing;(b) 后期切换到 f^o |
| LLM 评分有偏(对小众组合判错) | (a) 用 3 个不同 LLM 投票;(b) 对训练集中出现过的组合强制 F = 1 |
| 复现 CAMS 失败 | 联系作者要预训练权重;Troika 仓库可以作为更可靠的起点 |
| 提升幅度不够发顶会 | 把 FACA 单独拎出来做一篇 OW-CZSL 专项 paper(切题 ECCV/AAAI workshop) |

---

## 9. 论文写作要点

### 9.1 标题候选

1. **"Conditional Disentanglement Meets Feasibility-Aware Augmentation for Compositional Zero-Shot Learning"**
2. "Beyond Single-Space Disentanglement: Object-Conditioned Attribute Modeling for CZSL"
3. "CoDA: Conditional Disentanglement and Augmentation for Open-World Compositional Recognition"

### 9.2 故事弧线(introduction 三段式)

- **第 1 段**:CZSL 任务介绍 + CLIP-based 方法的兴起。
- **第 2 段**:点出 CAMS 等方法的两个根本问题——属性的上下文依赖性(用 "red wine vs red apple" 举例)和 OW 设定下不可行组合的干扰。**强调这不是 CAMS 一家的问题,是整个 CLIP-CZSL 范式的问题**(避免显得只是在打 CAMS)。
- **第 3 段**:介绍 CoDA 的三个组件 + 一句话总结贡献。

### 9.3 核心 selling points

1. **第一个在 CZSL 中显式建模 object-conditioned attribute** (区别于所有现有方法的对称解耦)
2. **第一个把 CLUB 用于 CZSL 解耦** (信息论 grounding)
3. **第一个把 LLM 评分用作 feasibility prior** (区别于 CGE/KG-SP 的 word2vec 余弦)
4. **OW 上的显著提升**(尤其 C-GQA),证明方法不只是 CW 调参

### 9.4 投稿目标

- **首选**: CVPR 2026(11 月截稿)/ ICCV 2026(3 月截稿)
- **备选**: AAAI 2026(8 月截稿,时间紧)/ ACM MM 2026
- **保底**: NeurIPS 2026 workshop / Pattern Recognition / TCSVT

---

## 10. 参考实现资源

### 必读论文

- **CAMS** (arXiv:2511.16378) — 主基线
- **Troika** (CVPR 2024) — CAMS 代码祖宗
- **CLUB** (Cheng et al., ICML 2020) — MI 上界估计
- **FiLM** (Perez et al., AAAI 2018) — feature-wise linear modulation
- **CLUSPRO** (ICLR 2025) — OW 强 baseline
- **KG-SP** (CVPR 2022) — feasibility prior 在 CZSL 的早期应用
- **OADis** (CVPR 2022) — feasibility-aware OW 推理

### 代码参考

- CAMS: `https://github.com/ybyangjing/CAMS`
- Troika: `https://github.com/bighuang624/Troika` (含 CW/OW 测试 pipeline)
- CLUB 官方实现: `https://github.com/Linear95/CLUB`
- 数据集分割: CSP 仓库(三个 dataset 都有标准 split)

### LLM 离线推理工具

- **vLLM** + Qwen2.5-32B-Instruct (本地,免费,推荐)
- OpenAI API: gpt-4o-mini ($0.15/1M tokens,可接受)
- **不要用** llama-2-7b 之类的弱模型,feasibility 判断不准

---

## 附录 A:CoDA 的最小可行验证 (MVP) — 第 2 周必做

在写完整代码前,先做这个 sanity check:

1. **冻结整个 CAMS,只加 OCAD**:
   - 在 UT-Zappos 上微调 5 epoch
   - 期望:HM 从 58.5 提升到 ≥ 59.0(任何提升都算 OCAD 有效)
2. **冻结 CAMS,只加 CLUB 估计 MI**:
   - 不做最小化,只跑前向估计
   - 期望:看到 CAMS 训练好的模型的 I(f^a, f^o) 显著大于 0(证明现有解耦不充分,我们的动机成立)

如果 1 失败 → OCAD 设计需要重新思考(改成 cross-attention 而非 FiLM)
如果 2 失败(MI 已经接近 0)→ 整个 MID 模块没必要做,转而把精力全投到 FACA

---

**文档结束。**

如有任何模块需要展开成更详细的代码或推导,告诉我具体是哪一块。

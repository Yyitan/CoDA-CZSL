# CoDA-CZSL：Conditional Disentanglement with Feasibility-Aware Augmentation（组合零样本学习的条件解耦与可行性感知增强）

> 基于 [CAMS](https://github.com/ybyangjing/CAMS)（[arXiv:2511.16378](https://arxiv.org/abs/2511.16378)）的改进研究 · 个人研究项目

## 研究动机

CAMS 用 Gated Cross-Attention 与多空间解耦（MSD）把 attribute 与 object 投影到独立空间，但存在三点局限：

1. **解耦缺乏显式约束**：`f^a` 与 `f^o` 来自同一组 latent units，仅靠分类 CE loss 隐式分离，无法保证两个表征统计独立；
2. **属性被当作上下文无关概念**："red wine" 的 red（深红半透明液体）与 "red apple" 的 red（表面反光固体）在视觉上并不是同一个东西，共享同一个 attribute embedding 是表达力瓶颈；
3. **开放世界（Open-World, OW）性能塌陷**：`A × O` 中大量物理上不可行的组合（如 "sliced sky"、"rusty water"）必须参与打分，引入噪声——CAMS 论文中 C-GQA 的 CW AUC 17.4% 掉到 OW AUC 4.8%。

CoDA-CZSL 针对这三点分别引入 **OCAD / MID / FACA** 三个模块。

## 方法

| 模块 | 作用 | 代码位置 |
|---|---|---|
| **OCAD**（Object-Conditioned Attribute Decoder） | 用对象表征生成 FiLM 调制参数 (γ, β)，对属性分支的 latent units 做条件化调制，建模属性的上下文依赖；仅增加一个 2 层 MLP，初始为恒等映射 | `patches_from_upstream/clip_modules__gca.py.diff`（`FiLM` 类，应用到上游 `clip_modules/gca.py`） |
| **MID**（Mutual Information Disentanglement） | 用 CLUB（ICML 2020）估计并最小化互信息上界 `I(f^a; f^o)`，显式强制属性–对象表征独立；CLUB 的 `q_φ` 与主网络双 optimizer 交替训练，`λ_MI` 线性 warmup | `model/coda_modules.py`（`CLUB`）+ `patches_from_upstream/train.py.diff`、`model__configure_model.py.diff` |
| **FACA**（Feasibility-Aware Compositional Augmentation） | LLM 离线预计算组合可行性矩阵 `F ∈ [0,1]^{|A|×|O|}`；训练期从高可行未见组合中做 memory bank 特征级合成（`ComposeFuser` 融合 `f^a`、`f^o`），OW 推理时加软偏置 `γ_feas·log(F+ε)` | `model/coda_modules.py`（`FeatureBank`、`ComposeFuser`）+ `patches_from_upstream/train.py.diff`、`test.py.diff` |

总损失：`L_total = L_CAMS + λ_MI·I_upper(f^a, f^o) + λ_aug·L_aug + λ_orth·||W_a^T W_o||_F^2`。

三个模块均有独立开关（`use_ocad` / `use_mid` / `use_faca`，总开关 `use_coda`），全部关闭即严格回退到原 CAMS；OW 推理软偏置 `gamma_feas: 0` 同样回退纯 CAMS。完整方法设计、消融矩阵与超参推荐见 [`CoDA-CZSL_proposal.md`](CoDA-CZSL_proposal.md)。

## 仓库内容

**本仓库只包含新增模块、配置与文档，不包含上游 CAMS 原始代码。** 对上游文件的修改以 unified diff 形式记录在 `patches_from_upstream/`，复现时先克隆官方 CAMS 再应用补丁（见下文复现步骤）。

```text
CoDA-CZSL/
├── CoDA-CZSL_proposal.md            # 完整研究方案（动机 / 方法 / 实验设计 / 风险与应对）
├── model/
│   └── coda_modules.py              # 新增模块：CLUB（MID）/ FeatureBank / ComposeFuser（FACA）
└── patches_from_upstream/           # 对上游 CAMS 文件的修改记录
    ├── clip_modules__gca.py.diff        # OCAD：FiLM 调制（应用于 clip_modules/gca.py）
    ├── model__cams.py.diff              # 主模型集成 OCAD / MID / FACA
    ├── model__configure_model.py.diff   # 双 optimizer、可训练参数扩展
    ├── train.py.diff                    # 双更新循环、λ_MI warmup、memory bank 更新
    ├── test.py.diff                     # OW 推理加软可行性 log-bias
    ├── flags.py.diff                    # CoDA 命令行 / yml 参数
    ├── configs__mit-states.yml.diff     # 三个数据集配置加入 CoDA 超参
    ├── configs__utzappos.yml.diff
    ├── configs__cgqa.yml.diff
    └── README.md.diff
```

## 实验结果

**本仓库目前处于方案设计与实现阶段，尚未产出可公开引用的 benchmark 数字。**

注意：`CoDA-CZSL_proposal.md` §6.5 中的数字是**预期目标（target Δ），不是已取得的实验结果**，请勿当作结论引用。规划中的评估协议为：在 MIT-States / UT-Zappos / C-GQA 上以复现的 CAMS 为直接对照，按 proposal §6.3 的消融矩阵（+OCAD / +MID / +FACA / 两两组合 / Full CoDA）报告闭世界与开放世界的 **S / U / HM / AUC**，重点关注 C-GQA OW AUC。

## 复现步骤

1. **克隆上游 CAMS 并应用补丁**：

   ```bash
   git clone https://github.com/ybyangjing/CAMS.git
   cd CAMS
   git apply /path/to/CoDA-CZSL/patches_from_upstream/*.diff
   ```

2. **放入本仓库新增文件**：

   ```bash
   cp -r /path/to/CoDA-CZSL/model ./
   cp /path/to/CoDA-CZSL/CoDA-CZSL_proposal.md ./
   ```

3. **数据集与权重路径**：下载 MIT-States / UT-Zappos / C-GQA（CSP 标准划分 `compositional-split-natural`），放到 `data/` 下或任意位置并通过 `--dataset_path` 传入；CLIP ViT-L/14 权重路径在 yml 的 `clip_model` 字段配置。显存建议 ≥ 24GB。

4. **FACA 可行性矩阵**：训练与 OW 推理需要 `feasibilities/feasibility_mit.pt`、`feasibility_zp.pt`、`feasibility_cgqa.pt` 三个文件。生成方式见 proposal §5.2（GPT-4o-mini API 或本地 vLLM + Qwen2.5 均可，对每个 `(attribute, object)` 组合让 LLM 输出 0–1 的物理可行性分）。

5. **训练与评估**：

   ```bash
   # 闭世界训练（MIT-States / C-GQA 换对应 config 与数据路径）
   python train.py --config configs/utzappos.yml --dataset_path data/ut-zap50k

   # 开放世界评估（同时启用 CAMS 硬阈值过滤与 CoDA 软偏置；gamma_feas=0 即回退纯 CAMS 推理）
   python test.py --config configs/utzappos.yml --dataset_path data/ut-zap50k --open_world
   ```

   关键超参（均可在 yml / 命令行调整）：`lambda_MI`、`lambda_aug`、`lambda_orth`、`club_lr`、`mi_warmup_epochs`、`tau_high`、`n_synth_per_batch`、`bank_size`、`faca_warmup_epochs`、`gamma_feas`，推荐初值见 proposal §5.3。训练进度条中的 `mi`（CLUB 估计的 MI 上界）应随训练逐步下降。

## 引用

本工作基于 CAMS，如使用本仓库请同时引用：

```bibtex
@article{yang2025cams,
  title={CAMS: Towards Compositional Zero-Shot Learning via Gated Cross-Attention and Multi-Space Disentanglement},
  author={Yang, Pan and Deng, Cheng and Yang, Jing and Zhao, Han and Liu, Yun and Chen, Yuling and Ruan, Xiaoli and Chen, Yanping},
  journal={arXiv preprint arXiv:2511.16378},
  year={2025}
}
```

## License

本项目以 [MIT License](LICENSE) 发布（Copyright (c) 2025 Yyitan），仅覆盖本仓库新增的代码与文档。上游 [CAMS](https://github.com/ybyangjing/CAMS) 仓库当前未附带 LICENSE 文件，使用其代码请遵循上游仓库的声明。方法设计中参考的开放工作：Troika (CVPR 2024)、CLUB (ICML 2020)、FiLM (AAAI 2018)、CLUSPRO (ICLR 2025)、KG-SP (CVPR 2022)、OADis (CVPR 2022)。

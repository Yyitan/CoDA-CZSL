English | [简体中文](README_zh.md)

# CoDA-CZSL: Conditional Disentanglement with Feasibility-Aware Augmentation for Compositional Zero-Shot Learning

> Improvement research built on [CAMS](https://github.com/ybyangjing/CAMS) ([arXiv:2511.16378](https://arxiv.org/abs/2511.16378)) · Personal research project

## Motivation

CAMS uses Gated Cross-Attention and Multi-Space Disentanglement (MSD) to project attributes and objects into independent spaces, but it has three limitations:

1. **Disentanglement lacks an explicit constraint**: `f^a` and `f^o` come from the same set of latent units and are separated only implicitly by the classification CE loss, so statistical independence of the two representations is not guaranteed;
2. **Attributes are treated as context-free concepts**: the red of "red wine" (a deep-red, translucent liquid) and the red of "red apple" (a reflective solid surface) are not visually the same thing; sharing one attribute embedding is an expressiveness bottleneck;
3. **Open-world (OW) performance collapse**: many physically infeasible pairs in `A × O` (e.g., "sliced sky", "rusty water") must still be scored, injecting noise — in the CAMS paper, C-GQA falls from 17.4% closed-world AUC to 4.8% OW AUC.

CoDA-CZSL introduces one module for each of the three points: **OCAD / MID / FACA**.

## Method

| Module | Role | Code location |
|---|---|---|
| **OCAD** (Object-Conditioned Attribute Decoder) | Uses the object representation to produce FiLM modulation parameters (γ, β) that conditionally modulate the attribute branch's latent units, modeling attribute context-dependence; adds only a 2-layer MLP, initialized as the identity map | `patches_from_upstream/clip_modules__gca.py.diff` (the `FiLM` class, applied to upstream `clip_modules/gca.py`) |
| **MID** (Mutual Information Disentanglement) | Uses CLUB (ICML 2020) to estimate and minimize an upper bound on the mutual information `I(f^a; f^o)`, explicitly enforcing attribute–object independence; CLUB's `q_φ` and the main network are trained with alternating dual optimizers, and `λ_MI` follows a linear warmup | `model/coda_modules.py` (`CLUB`) + `patches_from_upstream/train.py.diff`, `model__configure_model.py.diff` |
| **FACA** (Feasibility-Aware Compositional Augmentation) | An LLM precomputes an offline pair feasibility matrix `F ∈ [0,1]^{|A|×|O|}`; during training, feature-level synthesis from highly feasible unseen pairs via a memory bank (`ComposeFuser` fuses `f^a` and `f^o`); at OW inference, a soft bias `γ_feas·log(F+ε)` is added | `model/coda_modules.py` (`FeatureBank`, `ComposeFuser`) + `patches_from_upstream/train.py.diff`, `test.py.diff` |

Total loss: `L_total = L_CAMS + λ_MI·I_upper(f^a, f^o) + λ_aug·L_aug + λ_orth·||W_a^T W_o||_F^2`.

All three modules have independent switches (`use_ocad` / `use_mid` / `use_faca`, master switch `use_coda`); turning them all off falls back strictly to the original CAMS, and the OW soft bias `gamma_feas: 0` likewise falls back to pure-CAMS inference. For the full method design, ablation matrix, and recommended hyperparameters, see [`CoDA-CZSL_proposal.md`](CoDA-CZSL_proposal.md).

## Repository contents

**This repository only contains new modules, configurations, and documentation.** The `patches_from_upstream/` directory records modifications made to upstream files as unified diffs; the upstream CAMS code itself is not included. To reproduce, first clone the official CAMS and then apply the patches (see the reproduction steps below).

```text
CoDA-CZSL/
├── CoDA-CZSL_proposal.md            # Full research proposal (motivation / method / experimental design / risks and mitigations)
├── model/
│   └── coda_modules.py              # New modules: CLUB (MID) / FeatureBank / ComposeFuser (FACA)
└── patches_from_upstream/           # Records of modifications to upstream CAMS files
    ├── clip_modules__gca.py.diff        # OCAD: FiLM modulation (applied to clip_modules/gca.py)
    ├── model__cams.py.diff              # Main-model integration of OCAD / MID / FACA
    ├── model__configure_model.py.diff   # Dual optimizers, extended trainable parameters
    ├── train.py.diff                    # Dual update loops, λ_MI warmup, memory-bank updates
    ├── test.py.diff                     # Soft feasibility log-bias at OW inference
    ├── flags.py.diff                    # CoDA command-line / yml arguments
    ├── configs__mit-states.yml.diff     # CoDA hyperparameters added to the three dataset configs
    ├── configs__utzappos.yml.diff
    ├── configs__cgqa.yml.diff
    └── README.md.diff
```

## Experimental results

**This repository is currently at the proposal-design-and-implementation stage and has not yet produced benchmark numbers suitable for public citation.**

Note: the numbers in §6.5 of `CoDA-CZSL_proposal.md` are **expected targets (target Δ), not obtained experimental results**; please do not cite them as findings. The planned evaluation protocol is: on MIT-States / UT-Zappos / C-GQA, with the reproduced CAMS as the direct baseline, report closed- and open-world **S / U / HM / AUC** following the ablation matrix in proposal §6.3 (+OCAD / +MID / +FACA / pairwise combinations / Full CoDA), with particular attention to C-GQA OW AUC.

## Reproduction

1. **Clone upstream CAMS and apply the patches**:

   ```bash
   git clone https://github.com/ybyangjing/CAMS.git
   cd CAMS
   git apply /path/to/CoDA-CZSL/patches_from_upstream/*.diff
   ```

2. **Drop in the new files from this repository**:

   ```bash
   cp -r /path/to/CoDA-CZSL/model ./
   cp /path/to/CoDA-CZSL/CoDA-CZSL_proposal.md ./
   ```

3. **Datasets and weight paths**: download MIT-States / UT-Zappos / C-GQA (the CSP standard split `compositional-split-natural`) and place them under `data/` or anywhere else, passed via `--dataset_path`; the CLIP ViT-L/14 checkpoint path is configured in the yml `clip_model` field. A GPU with ≥ 24GB of memory is recommended.

4. **FACA feasibility matrix**: training and OW inference require the three files `feasibilities/feasibility_mit.pt`, `feasibility_zp.pt`, and `feasibility_cgqa.pt`. See proposal §5.2 for how to generate them (the GPT-4o-mini API or a local vLLM + Qwen2.5 both work; for each `(attribute, object)` pair, ask the LLM for a 0–1 physical-feasibility score).

5. **Training and evaluation**:

   ```bash
   # Closed-world training (swap in the corresponding config and data path for MIT-States / C-GQA)
   python train.py --config configs/utzappos.yml --dataset_path data/ut-zap50k

   # Open-world evaluation (enables both the CAMS hard-threshold filter and the CoDA soft bias; gamma_feas=0 falls back to pure-CAMS inference)
   python test.py --config configs/utzappos.yml --dataset_path data/ut-zap50k --open_world
   ```

   Key hyperparameters (all adjustable in the yml / on the command line): `lambda_MI`, `lambda_aug`, `lambda_orth`, `club_lr`, `mi_warmup_epochs`, `tau_high`, `n_synth_per_batch`, `bank_size`, `faca_warmup_epochs`, `gamma_feas`; recommended initial values are in proposal §5.3. The `mi` value in the training progress bar (the CLUB-estimated MI upper bound) should decrease steadily over the course of training.

## Citation

This work is built on CAMS; if you use this repository, please also cite:

```bibtex
@article{yang2025cams,
  title={CAMS: Towards Compositional Zero-Shot Learning via Gated Cross-Attention and Multi-Space Disentanglement},
  author={Yang, Pan and Deng, Cheng and Yang, Jing and Zhao, Han and Liu, Yun and Chen, Yuling and Ruan, Xiaoli and Chen, Yanping},
  journal={arXiv preprint arXiv:2511.16378},
  year={2025}
}
```

## License

This project is released under the [MIT License](LICENSE) (Copyright (c) 2025 Yyitan), which covers only the new code and documentation added in this repository. The upstream [CAMS](https://github.com/ybyangjing/CAMS) repository currently ships no LICENSE file; follow the upstream repository's terms when using its code. Open work referenced in the method design: Troika (CVPR 2024), CLUB (ICML 2020), FiLM (AAAI 2018), CLUSPRO (ICLR 2025), KG-SP (CVPR 2022), OADis (CVPR 2022).

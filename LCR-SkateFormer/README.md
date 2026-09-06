# LCR-SkateFormer: Looping Concept Reasoning SkateFormer

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Architecture](https://img.shields.io/badge/Backbone-SkateFormer-brightgreen.svg)]()

Official implementation of **LCR-SkateFormer** (Looping Concept Reasoning SkateFormer), an interpretable, knowledge-regularized framework for skeleton-based human action recognition based on SkateFormer and iterative symbolic reasoning.

---

## 🌟 Key Highlights & Scientific Motivation

Existing Concept Bottleneck Models (CBMs) for action recognition perform a single forward pass: $\text{Feature} \to \text{Concept} \to \text{Action}$. However, human reasoning about complex human actions often involves **iterative hypothesis testing and verification against commonsense domain knowledge (Knowledge Graphs)**.

**LCR-SkateFormer** introduces a **KG-guided Looping Concept Reasoning engine (`IterativeKGReasoner`)**:
1. **Perception-Reasoning Separation**: SkateFormer backbone extracts initial concept logits $z^0 \in \mathbb{R}^{K_{\text{concept}}}$ without black-box shortcutting.
2. **Iterative Refinement in Concept Space**:
   $$z^{t+1} = z^t + \beta \Delta z^t, \qquad c^{t+1} = \sigma(z^{t+1})$$
   where $\Delta z^t = \text{UpdateMLP}([z^t, r^t])$ is driven by soft KG rule activations ($r^t$) and modulated by a learnable gate $\beta \in [0, 0.5]$.
3. **Multi-Loop Deep Supervision**: Anchors initial perception ($c^0$) to prevent drift, supervises refined concepts ($c^K$) with asymmetric confidence-weighted loss, applies auxiliary action supervision on intermediate loops, and enforces concept stability.
4. **Verifiable Explanations**: Produces a full step-by-step reasoning trace ($t = 0 \to 1 \to \dots \to K$) susceptible to counterfactual deletion and sanity audits.

---

## 🏗️ Architecture Overview

```text
                  Skeleton Sequence X
                          ↓
                SkateFormer Backbone
                          ↓
              Feature Map [B, C, T', V*M]
                    ↙           ↘
          Spatial Pool         Temporal Pool
               ↓                     ↓
         Spatial Head          Temporal Head
               ↘                     ↙
            Initial Concept Logits z⁰
                          ↓
  ┌────────────────────────────────────────────────────────┐
  │  IterativeKGReasoner Loop t = 0 ... K-1                │
  │   1. Literals:      ℓᵗ = [cᵗ, 1 - cᵗ], cᵗ = σ(zᵗ)     │
  │   2. KG Rule Bank:  hidden₁ᵗ, hidden₂ᵗ → Evidence rᵗ   │
  │   3. Action Output: yᵗ = ActionHead(hidden₂ᵗ) / Proto  │
  │   4. Refine State:  Δzᵗ = UpdateMLP([zᵗ, rᵗ])          │
  │   5. Update Logits: zᵗ⁺¹ = zᵗ + β Δzᵗ                  │
  └────────────────────────────────────────────────────────┘
                          ↓
        Final Action yᴷ + Refined Concepts cᴷ + Reasoning Trace
```

---

## 📊 Mathematical Formulation

### 1. Multi-Objective Looping Loss
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{action}}^K + \lambda_{\text{aux}}\sum_{t=0}^{K-1} w_t \mathcal{L}_{\text{action}}^t + \lambda_{\text{concept}}\mathcal{L}_{\text{concept}}^K + \lambda_{\text{early}}\mathcal{L}_{\text{concept}}^0 + \lambda_{\text{KG}}\mathcal{L}_{\text{KG}} + \lambda_{\text{stable}}\mathcal{L}_{\text{stable}} + \lambda_{\text{prior}}\mathcal{L}_{\text{prior}} + \lambda_{\text{sparse}}\mathcal{L}_{\text{sparse}} + \lambda_{\text{part}}\mathcal{L}_{\text{part}}$$

- $\mathcal{L}_{\text{early}} = \operatorname{AsymBCE}(z^0, q)$: Anchors perception so the visual backbone preserves raw concept fidelity.
- $\mathcal{L}_{\text{concept}} = \operatorname{AsymBCE}(z^K, q)$: Supervises the final refined concepts.
- $\mathcal{L}_{\text{stable}} = \frac{1}{K}\sum_{t=0}^{K-1} \|\sigma(z^{t+1}) - \sigma(z^t)\|_1$: Prevents runaway oscillations.
- $\mathcal{L}_{\text{KG}}$: Penalizes violations of domain `require`, `alongside`, and `excludes` rules.

---

## 📂 Project Directory Structure

```text
LCR-SkateFormer/
├── model/
│   ├── __init__.py
│   ├── SkateFormer.py          # Complete SkateFormer backbone
│   └── lcr_skateformer.py      # LCRSkateFormer, IterativeKGReasoner, KGUnionLayer
├── config/
│   ├── train/                  # YAML configurations for NTU60/120 & NW-UCLA
│   └── test/
├── concepts/
│   └── ntu_rgbd_60_120.csv     # 72 Ontology concepts & action prototypes
├── feeders/
│   ├── feeder_ntu.py
│   ├── feeder_ntu_inter.py
│   ├── feeder_ucla.py
│   └── tools.py
├── torchlight/                 # DictAction & IO helpers
├── main.py                     # Training, Evaluation & Reasoning Trace Processor
├── verify_model.py             # Diagnostic & Unit Test Suite
├── generate_configs.py         # Config generator for datasets & ablations
├── generate_notebook.py        # Colab Notebook Generator
└── README.md
```

---

## 🚀 Quick Start

### 1. Run Verification Suite
```bash
python verify_model.py
```

### 2. Generate Configuration Files
```bash
python generate_configs.py
```

### 3. Train LCR-SkateFormer (e.g. NTU-120 CSub, K=2 Prototype Mode)
```bash
python main.py --config config/train/ntu120_csub/LCR_SkateFormer_j_K2_proto.yaml
```

### 4. Evaluate & Extract Reasoning Traces
```bash
python main.py --config config/test/ntu120_csub/test_LCR_SkateFormer_j.yaml --save-trace True
```

---

## 🔬 Ablation Experiments Matrix

| ID | Configuration | Purpose |
|---|---|---|
| **L0** | SkateFormer Baseline | Performance baseline |
| **L1** | ECR Baseline ($K=1$) | Single-step reasoning |
| **L2** | LCR-SkateFormer ($K=2$, Linear) | Linear action head with 2 loops |
| **L3** | **LCR-SkateFormer ($K=2$, Prototype)** | **Proposed Main Architecture** |
| **L4** | LCR-SkateFormer ($K=3$, Prototype)** | 3rd refinement step analysis |
| **L5** | LCR-SkateFormer ($K=2$, Unshared) | Shared vs independent parameters |
| **L6** | LCR-SkateFormer (No KG prior) | Measure contribution of KG priming |
| **L7** | LCR-SkateFormer (No concept update $\beta=0$) | Prove necessity of concept refinement |



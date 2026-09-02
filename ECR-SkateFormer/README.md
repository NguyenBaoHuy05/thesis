<div align="center">
<h2>ECR-SkateFormer: Explainable Concept Reasoner SkateFormer</h2>

<p>
An explainable neuro-symbolic extension of SkateFormer for skeleton-based human action recognition.
</p>
<p>
Based on <a href="https://github.com/KAIST-VICLab/SkateFormer">SkateFormer</a> by Jeonghyeok Do and Munchurl Kim.
</p>
</div>

---

## Overview

This repository extends the SkateFormer skeleton perception backbone with an explicit semantic concept bottleneck and a Knowledge Graph-grounded differentiable neuro-symbolic reasoner.

The original SkateFormer is the official PyTorch implementation of “SkateFormer: Skeletal-Temporal Transformer for Human Action Recognition” (ECCV 2024). It provides the skeleton perception backbone; the concept bottleneck and reasoning components described below are the ECR extension.

The intended decision path is:

```text
Skeleton sequence
      |
SkateFormer perception backbone
      |
Spatial and temporal concept prediction
      |
Positive/negative concept literals
      |
Knowledge Graph-primed differentiable reasoning
      |
Action prediction and reasoning trace
```

## Contribution Scope and Attribution

The concept vocabulary and class-level concept matrix used by this repository are adopted from the REASON work. ECR-SkateFormer does not claim to introduce a new concept ontology or a new concept annotation matrix. The adopted concept annotations are used as semantic supervision and are reorganized into the spatial/temporal concept interface required by this model.

The contributions of ECR-SkateFormer are instead:

1. **Explainable spatial-temporal concept bottleneck.** We adapt the adopted concept space to the SkateFormer feature map through separate spatial and temporal pooling and concept prediction branches, creating an explicit interface between neural skeleton perception and semantic reasoning.

2. **Knowledge-grounded differentiable neuro-symbolic reasoning.** We integrate concept evidence with learnable soft logical rules. Action prototypes and Knowledge Graph relations, including `require`, `alongside`, and `excludes`, are used as symbolic priors for action reasoning.

3. **Sparse and consistency-aware concept learning.** We train the model with a sparse asymmetric confidence-aware concept loss together with Knowledge Graph consistency, prior-retention, rule-sparsity, and part-diversity regularization.

Accordingly, the adopted REASON concept vocabulary/matrix should be cited as prior work in any paper using this repository. The novelty claimed for ECR-SkateFormer is the proposed spatial-temporal grounding, neuro-symbolic integration, and consistency-aware optimization over that adopted semantic space.

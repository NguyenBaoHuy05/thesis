"""
Unified Neurosymbolic Framework for Skeleton Action Recognition
===============================================================
Integrates Spatio-Temporal Backbones (SkateFormer, CTR-GCN, ST-GCN, HyperGCN),
9-Group Anatomical & Temporal Concept Bottleneck,
LoRA CLIP Semantic Grounding,
Knowledge Graph Regularization (S_KG, C_KG, A_KG),
Differentiable First-Order Logic Reasoning,
and Residual Neural-Symbolic Fusion.
"""

from __future__ import annotations
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from concepts.concept_utils import (
    GROUP_SPECS,
    build_concept_groups,
    load_concept_annotations,
)
from knowledge_graph import (
    build_concept_coactivation_static,
    build_semantic_similarity_from_concepts,
)
from model.backbones import build_backbone, BaseBackboneAdapter
from model.clip_alignment import LoRATextEncoder, MultiPositiveClipLoss
from model.crl.components import ConceptLogicLayers
from model.concept_decoder import GroupAwareConceptDecoder


class GRAFTSAR(nn.Module):
    """
    GRAFT-SAR: Group-aware Residual Adaptive Fusion with Temperature-Annealed Logic
    for Skeleton Action Recognition.
    """

    def __init__(
        self,
        backbone_name: str = "skateformer",
        backbone_cfg: Optional[Dict[str, Any]] = None,
        pretrained_backbone: Optional[str] = None,
        freeze_backbone: bool = False,
        num_classes: int = 60,
        concepts_csv: str = "concepts/ntu_rgbd_60_120.csv",
        num_rules: int = 128,
        use_not_logic: bool = True,
        alpha_init: float = 0.5,
        learnable_alpha: bool = True,
        use_clip_alignment: bool = True,
        clip_backbone: str = "ViT-B/32",
        lora_r: int = 4,
        concept_decoder_dim: int = 256,
        concept_decoder_heads: int = 8,
        concept_decoder_layers: int = 1,
        concept_queries_per_group: int = 4,
        loss_weights: Optional[Dict[str, float]] = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_clip_alignment = use_clip_alignment
        self.device = device

        # Default loss weights
        default_loss_weights = {
            "final_ce": 1.0,
            "neural_ce": 0.5,
            "logic_ce": 0.5,
            "concept_bce": 1.0,
            "clip": 0.2,
            "kg": 0.1,
            "logic_reg": 0.01,
        }
        self.loss_weights = default_loss_weights
        if loss_weights is not None:
            self.loss_weights.update(loss_weights)

        # 1. Initialize Neural Skeleton Backbone Adapter
        self.backbone: BaseBackboneAdapter = build_backbone(
            backbone_name=backbone_name,
            backbone_cfg=backbone_cfg or {},
            num_class=num_classes,
        )
        self.feat_dim = self.backbone.feat_dim

        # 2. Branch 1: Direct Neural Classification Head
        self.neural_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feat_dim, num_classes),
        )

        # 3. Load Concept Annotations & Group Partitions
        self.concept_matrix, self.concept_names, self.class_names = load_concept_annotations(
            concepts_csv, num_classes=num_classes
        )
        self.num_concepts = len(self.concept_names)
        self.concept_groups = build_concept_groups(self.concept_names)

        # 4. Knowledge Graph Ontological Priors (S_KG and C_KG)
        S_KG = build_semantic_similarity_from_concepts(self.concept_matrix)
        C_KG = build_concept_coactivation_static(self.concept_matrix)
        self.register_buffer("S_KG", S_KG)
        self.register_buffer("C_KG", C_KG)

        # Load pretrained weights after creating the direct head so an
        # optional pretrained classifier can safely initialize it.
        if pretrained_backbone and os.path.exists(pretrained_backbone):
            missing, unexpected = self.backbone.load_pretrained(pretrained_backbone, device=device)
            print(f"[Backbone] Loaded pretrained weights from '{pretrained_backbone}' (missing: {len(missing)}, unexpected: {len(unexpected)})")
            cls_weights = self.backbone.get_classifier_weights()
            if cls_weights is not None and cls_weights[0] is not None:
                w, b = cls_weights
                if w.shape == self.neural_head[1].weight.shape:
                    self.neural_head[1].weight.data.copy_(w.data)
                    if b is not None and self.neural_head[1].bias is not None:
                        self.neural_head[1].bias.data.copy_(b.data)

        # 5. Branch 2: Group-aware Transformer Concept Decoder & Projections
        self.concept_decoder = GroupAwareConceptDecoder(
            memory_dim=self.feat_dim,
            concept_groups=self.concept_groups,
            decoder_dim=concept_decoder_dim,
            num_heads=concept_decoder_heads,
            num_layers=concept_decoder_layers,
            queries_per_group=concept_queries_per_group,
        )
        self.group_projections = nn.ModuleDict()
        
        for g_name, indices in self.concept_groups.items():
            g_dim = len(indices)
            if g_dim == 0:
                continue
            # CLIP projection if enabled (project to 512-d CLIP space)
            if use_clip_alignment:
                self.group_projections[g_name] = nn.Sequential(
                    nn.Linear(concept_decoder_dim, 512),
                    nn.BatchNorm1d(512),
                )

        # Fallback full concept predictor (for joint projection)
        self.concept_proj_all = nn.Sequential(
            nn.Linear(self.feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.num_concepts),
        )

        # CLIP Text Encoder & Loss
        if use_clip_alignment:
            self.text_encoder = LoRATextEncoder(
                clip_backbone=clip_backbone,
                lora_r=lora_r,
                use_lora=True,
                device=device,
            )
            self.clip_loss_fn = MultiPositiveClipLoss()
        else:
            self.text_encoder = None
            self.clip_loss_fn = None

        # 6. Branch 3: Differentiable First-Order Logic Engine
        self.logic_engine = ConceptLogicLayers(
            num_concepts=self.num_concepts,
            num_classes=num_classes,
            num_rules=num_rules,
            use_not=use_not_logic,
            use_linear_classifier=True,
            initial_k=3.0,
            k_max=20.0,
        )

        # 7. Residual Fusion Weight alpha
        if learnable_alpha:
            self.alpha_param = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))
        else:
            self.register_buffer("alpha_param", torch.tensor(float(alpha_init), dtype=torch.float32))

        if freeze_backbone:
            frozen, trainable = self.backbone.freeze_backbone()
            print(f"[Backbone] Frozen {frozen:,} parameters in backbone for concept & logic tuning.")

    @property
    def alpha(self) -> torch.Tensor:
        """Clamps alpha into valid [0, 1] range."""
        return torch.clamp(self.alpha_param, 0.0, 1.0)

    def set_temperature(self, epoch: int, total_epochs: int):
        """Anneals logic binarization temperature."""
        self.logic_engine.set_temperature(epoch, total_epochs)

    def extract_features(
        self, x: torch.Tensor, index_t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extracts pooled embedding (B, feat_dim) from backbone."""
        return self.backbone.extract_feat(x, index_t=index_t)

    def extract_tokens(
        self, x: torch.Tensor, index_t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Return decoder memory tokens when the backbone supports them."""
        if hasattr(self.backbone, "extract_tokens"):
            return self.backbone.extract_tokens(x, index_t=index_t)
        feat = self.backbone.extract_feat(x, index_t=index_t)
        return feat.unsqueeze(1)

    def predict_concepts(
        self, feat: torch.Tensor, tokens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Predicts concept activations across all 9 groups.
        Returns:
            concept_probs: (B, num_concepts) in [0, 1]
            group_logits: Dict of {group_name: (B, group_dim)}
        """
        if tokens is None:
            tokens = feat.unsqueeze(1)
        concept_logits, group_logits, group_features = self.concept_decoder(tokens)

        # Add residual from global concept projector
        concept_logits = concept_logits + 0.1 * self.concept_proj_all(feat)
        concept_probs = torch.sigmoid(concept_logits)
        return concept_probs, group_logits, group_features

    def forward(
        self,
        x: torch.Tensor,
        index_t: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        concept_gt: Optional[torch.Tensor] = None,
        prompts_by_group: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """
        Unified Forward Pass.
        Args:
            x: Input skeleton tensor of shape (B, C, T, V, M)
            index_t: Optional temporal index tensor
            labels: Optional (B,) ground truth action labels
            concept_gt: Optional (B, num_concepts) ground truth concept vectors
            prompts_by_group: Dict of prompt lists for CLIP grounding
        Returns:
            Dictionary containing final/neural/logic logits, concept probs, and loss breakdown.
        """
        # 1. Backbone Feature Extraction
        tokens = self.extract_tokens(x, index_t=index_t)
        feat = tokens.mean(dim=1)
        if hasattr(self.backbone, "backbone") and getattr(self.backbone.backbone, "dropout", None) is not None:
            feat = self.backbone.backbone.dropout(feat)

        # 2. Branch 1: Direct Neural Logits
        y_neural = self.neural_head(feat)

        # 3. Branch 2: Concept Probabilities & Group Features
        concept_probs, group_logits, group_features = self.predict_concepts(feat, tokens=tokens)

        # 4. Branch 3: Differentiable Logic Reasoning
        y_logic, clauses, literals = self.logic_engine(concept_probs)

        # 5. Residual Fusion
        alpha = self.alpha
        y_final = (1.0 - alpha) * y_neural + alpha * y_logic

        outputs = {
            "y_final": y_final,
            "y_neural": y_neural,
            "y_logic": y_logic,
            "concept_probs": concept_probs,
            "clauses": clauses,
            "literals": literals,
            "alpha": alpha.item(),
        }

        # 6. Compute Comprehensive Losses if labels are provided
        if labels is not None:
            losses = {}
            # Action classification losses
            loss_final_ce = F.cross_entropy(y_final, labels)
            loss_neural_ce = F.cross_entropy(y_neural, labels)
            loss_logic_ce = F.cross_entropy(y_logic, labels)

            losses["loss_final_ce"] = loss_final_ce
            losses["loss_neural_ce"] = loss_neural_ce
            losses["loss_logic_ce"] = loss_logic_ce

            total_loss = (
                self.loss_weights["final_ce"] * loss_final_ce
                + self.loss_weights["neural_ce"] * loss_neural_ce
                + self.loss_weights["logic_ce"] * loss_logic_ce
            )

            # Concept Supervision BCE Loss
            if concept_gt is not None:
                loss_concept_bce = F.binary_cross_entropy(
                    concept_probs, concept_gt.to(concept_probs.device).float()
                )
                losses["loss_concept_bce"] = loss_concept_bce
                total_loss = total_loss + self.loss_weights["concept_bce"] * loss_concept_bce

            # CLIP Multi-Positive Contrastive Loss
            if (
                self.use_clip_alignment
                and self.text_encoder is not None
                and prompts_by_group is not None
                and concept_gt is not None
            ):
                clip_loss_total = torch.tensor(0.0, device=feat.device)
                count = 0
                for g_name, indices in self.concept_groups.items():
                    if (
                        g_name in self.group_projections
                        and g_name in prompts_by_group
                        and len(prompts_by_group[g_name]) > 0
                    ):
                        g_prompts = prompts_by_group[g_name]
                        g_proj = self.group_projections[g_name](group_features[g_name])  # (B, 512)
                        text_embeds = self.text_encoder.encode_prompts(g_prompts, device=feat.device)  # (C_g, 512)
                        pos_mask = concept_gt[:, indices].to(feat.device)  # (B, C_g)

                        if pos_mask.sum() > 0:
                            c_loss, _ = self.clip_loss_fn(g_proj, text_embeds, pos_mask)
                            clip_loss_total = clip_loss_total + c_loss
                            count += 1

                if count > 0:
                    clip_loss_avg = clip_loss_total / count
                    losses["loss_clip"] = clip_loss_avg
                    total_loss = total_loss + self.loss_weights["clip"] * clip_loss_avg

            # Knowledge Graph Regularization (C_KG and S_KG)
            # Concept co-activation correlation loss
            B = concept_probs.size(0)
            if B > 1:
                batch_corr = torch.matmul(concept_probs.t(), concept_probs) / B  # (num_concepts, num_concepts)
                loss_c_kg = F.mse_loss(batch_corr, self.C_KG.to(feat.device))
                # Action semantic consistency loss
                neural_probs = F.softmax(y_neural, dim=-1)
                batch_act_sim = torch.matmul(neural_probs, neural_probs.t())  # (B, B)
                target_sim = self.S_KG[labels][:, labels].to(feat.device)  # (B, B)
                loss_s_kg = F.mse_loss(batch_act_sim, target_sim)

                loss_kg = loss_c_kg + loss_s_kg
                losses["loss_kg"] = loss_kg
                total_loss = total_loss + self.loss_weights["kg"] * loss_kg

            # Logic Rule L1 Sparsity Regularization
            loss_logic_reg = self.logic_engine.get_regularization_loss()
            losses["loss_logic_reg"] = loss_logic_reg
            total_loss = total_loss + self.loss_weights["logic_reg"] * loss_logic_reg

            outputs["loss"] = total_loss
            outputs["losses"] = losses

        return outputs

    @torch.no_grad()
    def explain(
        self,
        x: torch.Tensor,
        index_t: Optional[torch.Tensor] = None,
        top_k_concepts: int = 5,
        threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Generates symbolic, interpretable natural language explanations for a batch of inputs.
        """
        self.eval()
        tokens = self.extract_tokens(x, index_t=index_t)
        feat = tokens.mean(dim=1)
        concept_probs, _, _ = self.predict_concepts(feat, tokens=tokens)
        y_logic, clauses, literals = self.logic_engine(concept_probs)
        y_neural = self.neural_head(feat)
        y_final = (1.0 - self.alpha) * y_neural + self.alpha * y_logic

        pred_classes = torch.argmax(y_final, dim=-1).cpu().numpy()
        rules = self.logic_engine.extract_rules(self.concept_names, self.class_names)

        explanations = []
        for i in range(x.size(0)):
            c_probs = concept_probs[i].cpu().numpy()
            pred_c = pred_classes[i]
            pred_name = self.class_names[pred_c] if pred_c < len(self.class_names) else f"Class_{pred_c}"

            # Top active concepts
            top_concept_idx = np.argsort(c_probs)[::-1][:top_k_concepts]
            active_concepts = [
                {"name": self.concept_names[idx], "prob": round(float(c_probs[idx]), 3)}
                for idx in top_concept_idx
                if c_probs[idx] >= threshold
            ]

            # Matching logic rules
            matched_rules = rules.get(pred_name, [])

            explanations.append({
                "sample_idx": i,
                "predicted_action": pred_name,
                "predicted_id": int(pred_c),
                "neural_confidence": round(float(F.softmax(y_neural[i], dim=-1)[pred_c]), 4),
                "logic_confidence": round(float(F.softmax(y_logic[i], dim=-1)[pred_c]), 4),
                "fusion_alpha": round(float(self.alpha), 3),
                "active_concepts": active_concepts,
                "logic_rules": matched_rules,
            })

        return explanations


# Alias for backward compatibility
ThesisNeurosymbolicModel = GRAFTSAR

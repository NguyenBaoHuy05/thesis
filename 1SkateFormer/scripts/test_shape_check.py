"""
Shape-check test for concept model pipeline (no data files needed).
Tests that the entire concept model forward pass works correctly:
  input [B, 3, T, V, M] -> concept_logits [B, 74] + action_logits [B, num_classes]
"""
import sys
sys.path.insert(0, '.')

import torch
import numpy as np

# ---- Test ConceptDecoder standalone ----
print("=== Testing ConceptDecoder ===")
from model.concept_decoder import ConceptDecoder, masked_bce_loss, soft_bce_loss

dec = ConceptDecoder(backbone_dim=192, d_model=256, num_concepts=74, num_heads=8, num_layers=2)
dec.eval()

# Test with 4D feature map [B, C, T', V']
feat4d = torch.randn(2, 192, 8, 48)   # B=2, C=192, T'=8, V'=48 (8*6=48 joints*2persons)
logits = dec(feat4d)
print(f"  Input (4D): {feat4d.shape} -> concept logits: {logits.shape}")
assert logits.shape == (2, 74), f"Expected (2, 74), got {logits.shape}"

# Test with 3D token sequence [B, L, C]
feat3d = torch.randn(2, 384, 192)   # B=2, L=384, C=192
logits2 = dec(feat3d)
print(f"  Input (3D): {feat3d.shape} -> concept logits: {logits2.shape}")
assert logits2.shape == (2, 74)

# Test masked BCE loss
targets = torch.randint(0, 2, (2, 74)).float()
targets[0, :10] = -1.0   # unknown positions
mask = (targets >= 0)
loss = masked_bce_loss(logits, targets, mask)
print(f"  Masked BCE loss: {loss.item():.4f}")
assert not torch.isnan(loss), "NaN in BCE loss!"
print("ConceptDecoder OK\n")

# ---- Test SymbolicDecoder ----
print("=== Testing SymbolicDecoder ===")
from model.symbolic_decoder import SymbolicDecoder

template = torch.from_numpy(
    np.random.randint(0, 2, (60, 74)).astype(np.float32)
)

# Learned mode
sym_learned = SymbolicDecoder(num_concepts=74, num_classes=60,
                               template_matrix=template, mode="learned")
concept_probs = torch.sigmoid(torch.randn(2, 74))
action_logits = sym_learned(concept_probs)
print(f"  Learned mode: concept_probs {concept_probs.shape} -> action_logits {action_logits.shape}")
assert action_logits.shape == (2, 60)

# Symbolic mode
sym_symbolic = SymbolicDecoder(num_concepts=74, num_classes=60,
                                template_matrix=template, mode="symbolic")
action_logits2 = sym_symbolic(concept_probs)
print(f"  Symbolic mode: -> {action_logits2.shape}")
assert action_logits2.shape == (2, 60)
print("SymbolicDecoder OK\n")

# ---- Test ConceptModel (no backbone ckpt - random init) ----
print("=== Testing ConceptModel (random backbone) ===")
from model.concept_model import ConceptModel

template_np = np.random.randint(0, 2, (60, 74)).astype(np.float32)
template_t = torch.from_numpy(template_np)

model = ConceptModel(
    backbone_ckpt=None,
    num_classes=60,
    num_concepts=74,
    backbone_dim=192,
    backbone_kwargs={
        "num_classes": 60,
        "num_people": 2,
        "num_points": 24,
        "kernel_size": 7,
        "num_heads": 32,
        "attn_drop": 0.0,
        "head_drop": 0.0,
        "rel": True,
        "drop_path": 0.0,
        "type_1_size": [8, 8],
        "type_2_size": [8, 12],
        "type_3_size": [8, 8],
        "type_4_size": [8, 12],
        "mlp_ratio": 4.0,
        "index_t": True,
    },
    decoder_kwargs={
        "d_model": 128,  # smaller for speed
        "num_heads": 4,
        "num_layers": 1,
        "d_ff": 256,
        "dropout": 0.0,
        "use_self_attn": False,
        "mode": "learned",
    },
    freeze_backbone=True,
    use_symbolic=True,
    template_matrix=template_t,
)
model.eval()

print(f"  Trainable params: {model.count_trainable():,}")
print(f"  Total params:     {model.count_total():,}")

# Minimal forward pass
# NOTE: partition=True in feeder maps 25 joints → 24 joints before model input
B, C, T, V, M = 2, 3, 64, 24, 2   # V=24 because of partition=True in feeder
data = torch.randn(B, C, T, V, M)
index_t = torch.arange(T).float().unsqueeze(0).expand(B, -1)

with torch.no_grad():
    concept_logits, action_logits = model(data, index_t)

print(f"  Input:          {data.shape}")
print(f"  concept_logits: {concept_logits.shape}")
print(f"  action_logits:  {action_logits.shape}")
assert concept_logits.shape == (B, 74), f"Expected ({B}, 74), got {concept_logits.shape}"
assert action_logits.shape  == (B, 60), f"Expected ({B}, 60), got {action_logits.shape}"
assert not torch.isnan(concept_logits).any(), "NaN in concept_logits!"
assert not torch.isnan(action_logits).any(),  "NaN in action_logits!"

print("\n=== ALL SHAPE CHECKS PASSED! ===")

# ---- Test concept probs range ----
probs = model.decoder.get_concept_probs(model.extract_features(data, index_t))
assert (probs >= 0).all() and (probs <= 1).all(), "Concept probs out of [0,1]!"
print("Concept probs in [0,1]: OK")

print("\nAll checks complete!")

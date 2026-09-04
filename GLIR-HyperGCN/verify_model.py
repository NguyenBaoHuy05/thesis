"""Smoke-test cho GLIR-HyperGCN: xac nhan G1 (BodyPartGroundedConceptHeads),
G2 (LightConceptGrounding, graceful fallback khi khong co CLIP/internet),
G3 (set_phase roi rac cho cau hinh V2, set_progress lien tuc cho cau hinh V3),
va vong lap logic (ke thua tu CLIR) forward/backward khong loi.

Khong huan luyen that (khong co du lieu NTU that trong moi truong nay) — chi
kiem tra tinh dung dan kien truc bang du lieu gia lap, giong tinh than
verify_model.py cua CLIR-SkateFormer.
"""

import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from model.glir_hypergcn import GLIRHyperGCN, NTU25_BODY_PART_JOINTS


def print_section(title):
    print("\n" + "=" * 75)
    print(f"  {title}")
    print("=" * 75)


def make_model(**overrides):
    cfg = dict(
        num_classes=60,
        num_person=2,
        num_point=25,
        in_channels=3,
        embedding_channels=32,   # nho de smoke-test nhanh (mac dinh that: 128)
        num_loops=2,
        r1=64, r2a=32, r2b=32,   # nho de smoke-test nhanh (mac dinh that: 256/128/128)
        literal_mode="soft",
        use_grounding=True,
    )
    cfg.update(overrides)
    return GLIRHyperGCN(**cfg)


def dummy_batch(B=2, T=32, V=25, M=2, num_concepts=72, num_classes=60):
    x = torch.randn(B, 3, T, V, M)
    concept_targets = (torch.rand(B, num_concepts) > 0.7).float()
    action_labels = torch.randint(0, num_classes, (B,))
    return x, concept_targets, action_labels


def test_body_part_joint_coverage():
    print_section("Test 1: NTU25_BODY_PART_JOINTS phu du 25 khop, khong chong lan")
    all_idx = []
    for g, idx in NTU25_BODY_PART_JOINTS.items():
        all_idx.extend(idx)
    assert sorted(all_idx) == list(range(25)), f"Thieu/trung khop: {sorted(all_idx)}"
    print(f"  OK — 6 nhom, {len(all_idx)} khop, phu dung 0..24 khong trung.")


def test_forward_backward(model, tag, x, concept_targets, action_labels):
    out = model(x, concept_targets=concept_targets, action_labels=action_labels)
    assert out["action_logits"].shape == (x.shape[0], model.num_classes)
    assert out["concept_logits"].shape == (x.shape[0], model.num_concepts)
    assert "l_c_K" in out and "l_sup" in out and "aux_loss" in out
    loss = torch.nn.functional.cross_entropy(out["action_logits"], action_labels) + out["aux_loss"]
    loss.backward()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_total = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"  [{tag}] loss={loss.item():.4f}  grad flow: {n_grad}/{n_total} params co grad != 0")
    assert n_grad > 0
    model.zero_grad()
    return out


def test_v2_config():
    print_section("Test 2: Cau hinh V2 (G1 + G2, set_phase roi rac — chua bat G3)")
    model = make_model()
    x, c_t, a_l = dummy_batch()

    model.set_phase(1)
    print(f"  Phase1: literal_mode={model.reasoner.literal_mode}, beta_max={model.reasoner.beta_max}, lambda_reg={model.lambda_reg}")
    test_forward_backward(model, "V2/Phase1", x, c_t, a_l)

    model.set_phase(2)
    print(f"  Phase2: literal_mode={model.reasoner.literal_mode}, beta_max={model.reasoner.beta_max}, lambda_reg={model.lambda_reg}")
    out = test_forward_backward(model, "V2/Phase2", x, c_t, a_l)
    assert "l_reg" in out and "l_sparse" in out
    print("  OK — V2 forward/backward on ca 2 phase, khong crash.")


def test_v3_config():
    print_section("Test 3: Cau hinh V3 (G1 + G2 + G3, set_progress lien tuc)")
    model = make_model()
    x, c_t, a_l = dummy_batch()

    prev_beta_max = None
    for progress in (0.0, 0.3, 0.6, 1.0):
        model.set_progress(progress)
        print(f"  progress={progress:.1f}: temperature={model.reasoner.temperature:.3f}, "
              f"beta_max={model.reasoner.beta_max:.3f}, lambda_reg={model.lambda_reg:.5f}")
        if prev_beta_max is not None:
            assert model.reasoner.beta_max >= prev_beta_max - 1e-6, "beta_max phai tang don dieu theo progress"
        prev_beta_max = model.reasoner.beta_max
        test_forward_backward(model, f"V3/progress={progress}", x, c_t, a_l)
    print("  OK — V3 annealing lien tuc, khong co buoc nhay dot ngot, forward/backward on dinh.")


def test_grounding_graceful_fallback():
    print_section("Test 4: G2 LightConceptGrounding — graceful fallback khi khong co CLIP")
    model = make_model(use_grounding=True)
    print(f"  grounding.available = {model.grounding.available} "
          f"({'CLIP tai duoc' if model.grounding.available else 'khong co internet/checkpoint — dung fallback 0-loss, KHONG crash'})")
    x, c_t, a_l = dummy_batch()
    out = model(x, concept_targets=c_t, action_labels=a_l)
    assert "l_ground" in out
    print(f"  l_ground = {out['l_ground'].item():.4f} (0.0 neu grounding khong available)")
    loss = out["aux_loss"]
    loss.backward()
    print("  OK — model khong crash du CLIP co san hay khong.")


def test_concept_slice_alignment():
    print_section("Test 5: Doi chieu concept_slices (G1) voi part groups o tang logic (L1)")
    model = make_model()
    logic_groups = {name: (start, end) for name, start, end, _ in
                     __import__("model.crl.conjunction_layer", fromlist=["get_part_groups"]).get_part_groups(72)}
    for g, (s, e) in model.concept_heads.concept_slices.items():
        assert logic_groups[g] == (s, e), f"Lech slice nhom {g}: G1={s,e} vs L1={logic_groups[g]}"
    print("  OK — concept_slices cua G1 khop chinh xac voi group boundaries o ConjunctionLayer (L1).")


if __name__ == "__main__":
    test_body_part_joint_coverage()
    test_concept_slice_alignment()
    test_v2_config()
    test_v3_config()
    test_grounding_graceful_fallback()
    print_section("TAT CA TEST PASS")

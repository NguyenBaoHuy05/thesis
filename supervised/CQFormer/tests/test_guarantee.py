"""Check the guarantee and the shapes. Run on Colab, from the CQFormer folder:

    python tests/test_guarantee.py

For every joint set: moving joints outside the set leaves its concepts exactly unchanged, and
moving joints inside the set does change them.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.cqformer import CQFormer  # noqa: E402
from model.joint_sets import JOINT_SETS, load_concept_matrix  # noqa: E402

CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "concepts", "ntu_rgbd_60_120.csv")


def sample(device, two_person):
    x = torch.randn(4, 3, 64, 25, 2, device=device)
    if not two_person:
        x[1:3, :, :, :, 1] = 0   # samples 1 and 2 have one person
    index_t = torch.linspace(-1, 1, 64, device=device).repeat(4, 1)
    return x, index_t


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for num_classes in (60, 120):
        S = load_concept_matrix(CSV, num_classes)
        model = CQFormer(num_classes, S, dim=48, depth=2, heads=4, free_per_set=4, motifs=16, order_filters=16).to(device).eval()
        x, index_t = sample(device, two_person=False)
        with torch.no_grad():
            out = model(x, index_t)
        assert out["logits"].shape == (4, num_classes)
        assert out["concept_logits"].shape == (4, S.shape[1])
        assert out["timeline"].shape == (4, 16, model.num_timeline)

    for two_person in (False, True):
        x, index_t = sample(device, two_person)
        for i, s in enumerate(model.sets):
            inside = list(s.joints)
            outside = [j for j in range(25) if j not in inside]
            with torch.no_grad():
                base = model.run_set(x, index_t, i)
                if outside:
                    moved = x.clone()
                    moved[:, :, :, outside, :] = torch.randn_like(moved[:, :, :, outside, :])
                    assert torch.equal(base, model.run_set(moved, index_t, i)), "leak in " + s.name
                moved = x.clone()
                moved[:, :, :, inside, 0] += 1.0 + torch.randn_like(moved[:, :, :, inside, 0])
                assert not torch.equal(base, model.run_set(moved, index_t, i)), "dead set " + s.name
            print("ok {:12s} {:2d} joints, two_person sample={}".format(s.name, len(inside), two_person))

    # gradient form of the guarantee, on one set
    i = next(k for k, s in enumerate(model.sets) if s.name == "legs")
    x, index_t = sample(device, two_person=True)
    x.requires_grad_(True)
    model.run_set(x, index_t, i).sum().backward()
    outside = [j for j in range(25) if j not in JOINT_SETS[i].joints]
    assert x.grad[:, :, :, outside, :].abs().sum().item() == 0
    assert x.grad[:, :, :, list(JOINT_SETS[i].joints), :].abs().sum().item() > 0
    print("all good on", device)


if __name__ == "__main__":
    main()

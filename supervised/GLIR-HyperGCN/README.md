# GLIR-HyperGCN

> **Tên đầy đủ:** Grounded Logic Iterative Refinement với backbone HyperGCN
> **Nguồn gốc:** Nâng cấp trực tiếp từ [`CLIR-SkateFormer`](../CLIR-SkateFormer) theo
> [`Architecture/10_CLIR-SkateFormer_v2_Proposal.md`](../Architecture/10_CLIR-SkateFormer_v2_Proposal.md),
> đổi backbone sang HyperGCN (tham khảo [`Hyper-GCN`](../Hyper-GCN), dùng bản đã làm sạch tại
> `GRAFT-SAR/model/backbones/hypergcn.py`), triển khai đủ **G1 + G2 + G3** (chưa gồm G4 —
> adaptive halting, xem ablation V4 trong proposal nếu muốn làm tiếp).

## Vì sao đổi sang HyperGCN?

`extract_feat` của SkateFormer giữ nguyên chiều khớp `V'` suốt backbone (đã xác nhận:
`PatchMergingTconv` chỉ downsample theo T), nhưng `num_points=24` không khớp 1-1 rõ ràng
với layout NTU-25 chuẩn. HyperGCN dùng đúng `num_point=25` (layout NTU-25 chuẩn, cùng
`inward_ori_index` với `HyperGraph` trong `model/hypergcn.py`) và cũng giữ `V` không đổi
qua các tầng — khớp thẳng với `NTU25_BODY_PART_JOINTS` không cần đoán/ánh xạ lại.

## 3 khoảng trống đã vá (so với CLIR-SkateFormer gốc)

| | Khoảng trống ở CLIR gốc | Cách vá ở GLIR-HyperGCN |
|---|---|---|
| **G1** | `spatial_head` dự đoán cả 52 concept từ 1 vector pooled toàn cục — không có ràng buộc "theo bộ phận" ở tầng nhận thức | `BodyPartGroundedConceptHeads` ([model/glir_hypergcn.py](model/glir_hypergcn.py)) — mỗi nhóm (Head/Hand/Arm/Hip/Leg/Foot) pool RIÊNG trên đúng tập khớp NTU của mình (`NTU25_BODY_PART_JOINTS`), khớp 1-1 với `get_part_groups()` ở tầng logic L1 |
| **G2** | Không có ngữ nghĩa ngôn ngữ nào cho concept | `LightConceptGrounding` ([model/crl/light_grounding.py](model/crl/light_grounding.py)) — CLIP **đóng băng hoàn toàn**, text anchor encode 1 lần lúc init, chỉ học 1 projection nhỏ. Graceful fallback (`l_ground=0`) nếu không tải được CLIP (offline) |
| **G3** | `set_phase()` đổi `temperature`/`beta_max`/`lambda_reg` đột ngột tại 1 epoch cố định | `set_progress(progress)` nội suy liên tục theo `epoch/num_epoch` — không có bước nhảy. `set_phase()` vẫn giữ lại (dùng cho cấu hình V2) |

Vòng lặp logic (`ConceptUpdateCell`, `IterativeLogicReasoner`) và toàn bộ `model/crl/`
(`literal_layer.py`, `conjunction_layer.py`, `disjunction_layer.py`, `crl_reasoner.py`,
`rule_extractor.py`) **kế thừa nguyên trạng** từ CLIR-SkateFormer — các module này vốn đã
backbone-agnostic (chỉ tiêu thụ concept logits), không cần sửa khi đổi backbone.

## Cấu trúc

```
GLIR-HyperGCN/
├── model/
│   ├── hypergcn.py          # Backbone (kế thừa GRAFT-SAR, thêm extract_feature_map())
│   ├── glir_hypergcn.py      # GLIRHyperGCN + BodyPartGroundedConceptHeads (G1) + set_progress (G3)
│   ├── clip/                 # CLIP vendored (kế thừa GRAFT-SAR/REASON, dùng cho G2)
│   └── crl/                  # Kế thừa nguyên trạng từ CLIR-SkateFormer + light_grounding.py (G2) mới
├── concepts/ntu_rgbd_60_120.csv
├── feeders/                  # Kế thừa nguyên trạng từ CLIR-SkateFormer
├── config/ntu60_cs/glir_hypergcn.yaml
├── train.py                  # GLIRTrainer — hỗ trợ cả 2 cấu hình V2/V3 qua --variant
├── scripts/extract_rules.py  # Kế thừa từ CLIR, chỉ đổi lớp model khi load checkpoint
└── verify_model.py           # Smoke test — ĐÃ CHẠY PASS (xem bên dưới)
```

## Notebook Colab/Kaggle (giống kiểu CLIR-SkateFormer)

`generate_notebook.py` sinh ra `GLIR_HyperGCN_Colab.ipynb` — notebook tự chứa 100%
(embed thẳng source từ `model/`, `feeders/`, `concepts/`, không cần clone repo), theo
đúng khuôn mẫu `CLIR-SkateFormer/generate_notebook.py`:

```bash
python generate_notebook.py   # sinh lại GLIR_HyperGCN_Colab.ipynb sau khi sửa model/
```

12 block: Dependencies (kèm `pip install CLIP`) → Concept CSV & Feeder → HyperGCN backbone
→ Tầng logic CRL → **G2 LightConceptGrounding** (block riêng) → ConceptUpdateCell/
IterativeLogicReasoner/**G1 BodyPartGroundedConceptHeads**/GLIRHyperGCN → RuleExtractor →
Smoke test (chạy thử cả V2 lẫn V3 ngay trong notebook) → Cấu hình + chọn `VARIANT=v2|v3` →
`GLIRTrainer` (hỗ trợ cả 2 lịch huấn luyện) → Rule extraction + Fidelity/Coverage/Precision →
CLI thay thế.

Đã dry-run toàn bộ code cell tới hết Block 8 (smoke test) trong 1 process Python thường
(bỏ qua các dòng `!pip install`) — không lỗi cú pháp/runtime, V2 và V3 forward/backward
đều chạy đúng ngay trong notebook.

## Đã kiểm chứng (verify_model.py, không cần dữ liệu NTU thật)

```
python verify_model.py
```

5/5 test pass trên môi trường này (`D:\Thesis\venv`, torch 2.13.0+cpu):
1. `NTU25_BODY_PART_JOINTS` phủ đúng 25 khớp, không chồng lấn.
2. `concept_slices` của G1 khớp chính xác với group boundary ở `ConjunctionLayer` (L1).
3. Cấu hình **V2** (`set_phase` rời rạc) — forward/backward OK ở cả Phase 1 và Phase 2.
4. Cấu hình **V3** (`set_progress` liên tục) — `beta_max` tăng đơn điệu theo progress,
   không có bước nhảy, forward/backward ổn định ở mọi mốc progress.
5. G2 graceful fallback — đã thử cả 2 nhánh: khi CLIP tải được (môi trường này có internet,
   `l_ground` được tính thật) và khi giả lập không có `clip` module (fallback `l_ground=0`,
   không crash).

**Chưa chạy training thật** — cần dữ liệu NTU RGB+D đã tiền xử lý (`.npz`, xem
`train_feeder_args.data_path` trong config) không có sẵn trong môi trường phát triển này.

## Chạy training (khi có dữ liệu)

```bash
# V2 — G1+G2, lịch huấn luyện 2-phase rời rạc (baseline so sánh)
python train.py --variant v2 --config config/ntu60_cs/glir_hypergcn.yaml

# V3 — G1+G2+G3, lịch annealing liên tục (cấu hình chính đề xuất)
python train.py --variant v3 --config config/ntu60_cs/glir_hypergcn.yaml
```

Sau khi có `best_model.pt`:

```bash
python scripts/extract_rules.py \
    --checkpoint work_dir/ntu60_cs/glir_hypergcn/best_model.pt \
    --config config/ntu60_cs/glir_hypergcn.yaml \
    --eval-fidelity
```

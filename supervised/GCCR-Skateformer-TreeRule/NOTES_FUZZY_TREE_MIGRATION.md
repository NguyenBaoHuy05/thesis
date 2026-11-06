# Ghi chú: Migration sang Fuzzy AND/OR Logic Tree (để tiếp tục ở session khác)

- **Ngày**: 15-16/09/2026
- **Plan gốc**: `C:\Users\Admin\.claude\plans\idempotent-toasting-hippo.md` (đã approve và thực hiện)
- **So sánh**: bản cũ trước khi redesign nằm ở `D:\Thesis\GCCR-Skateformer-bandacaitien` (có `JointSymbolicActionCompatibility` + `ConceptGCN` + fusion α-blend — kiến trúc bị coi là "yếu" về mặt luận điểm symbolic thật, xem lý do ở mục 1).

---

## 1. Quyết định thiết kế (đã chốt, đã code xong)

Sau nhiều vòng thảo luận (Naive Bayes → TAN/KG-Bayes → Prototype/ProtoPNet → Rule-Tree → cuối cùng chọn **Fuzzy AND/OR Logic Tree**), đã:

- **Bỏ hẳn**: `JointSymbolicActionCompatibility` (attention học tự do bằng gradient — không phải luật thật), `ConceptGCN` (gây leakage), fusion `α`-blend neural+symbolic.
- **Giữ**: backbone-agnostic feature extraction, 9 concept-group heads (`encode_group_concepts`), ontology Jaccard-based (`concept_matrix` từ CSV), concept BCE loss, `direct_action` (giữ làm mốc neural tham chiếu, KHÔNG fuse vào output).
- **Thay bằng**: `FuzzyLogicTree` — cây nhị phân, mỗi node là AND/OR gate thật (t-norm product / t-conorm probabilistic-sum), lá là literal concept có NOT-gate mềm, train bằng loss riêng (CE + entropy regularizer ép rời rạc hoá), nhiệt độ anneal dần 1.0→0.05 qua các epoch, sau đó **harden** thành công thức Boolean tường minh.

## 2. File đã sửa / tạo mới

- **`fuzzy_logic_tree.py`** (mới) — class `FuzzyLogicTree`: `forward(hard=False/True)`, `loss()`, `harden()`, `hardening_report()`, `explain_hard()` (trace literal cho case study), `export_formula()` (xuất công thức Boolean, có dedupe `X op X -> X`).
- **`joint_symbolic_model.py`** — bỏ 2 class cũ, đơn giản hoá `encode_group_concepts`/`forward`, gắn `self.fuzzy_tree`.
- **`train_joint.py`** — bỏ lịch trình alpha/symbolic-weight cũ, bỏ `dump_explanations()`, thêm anneal nhiệt độ mỗi epoch (`model.fuzzy_tree.temperature = ...`), fix bug resume (xem mục 3).
- **`harden_tree.py`** (mới) — script standalone: load checkpoint, in `hardening_report()`, so `soft/hard/hybrid/unique` accuracy + coverage, xuất `rule_tree_hardened.txt`.
- **`eval_rulebook.py`** — so sánh 4 điểm (neural/tree_soft/tree_hard/kmeans_rb) + coverage/hybrid accuracy + ensemble `tree_km` (sweep gamma) + `case_study()` (literal trace + true/predicted) + **chẩn đoán concept head collapse ở Step 2** (mean/std per concept, cảnh báo nếu std < 0.02).
- **`config/ntu/skateformer/ntu_cs_j.yaml`** — bỏ key cũ (`use_concept_gcn`, `kg_s_weight`, `fusion_*`, `symbolic_*`...), thêm `tree_depth: 4`, `tree_entropy_weight: 0.01`, `tree_temp_start: 1.0`, `tree_temp_end: 0.05`.
- **`run_eval_rulebook.sh`** — cập nhật flag (`--case-study --tree-km-sweep`).
- **Chưa sửa (out of scope, sẽ lỗi nếu chạy)**: `verify_model.py` (lỗi import `JointSymbolicActionCompatibility`), `evaluate_branches.py`, `dump_rulebook.py`, `eval_ensemble.py` (lỗi runtime gọi `model.symbolic_action`) — đều là script phụ/debug, không thuộc pipeline train/eval chính.

## 3. Bug đã tìm và fix trong quá trình chạy thật trên Kaggle

1. **`KeyError: initial_lr`** khi resume training (`--resume-checkpoint --start-epoch >1`) — PyTorch mới yêu cầu `initial_lr` có sẵn trong param_groups khi `last_epoch>=0`. Đã fix trong `train_joint.py` (set `group.setdefault("initial_lr", group["lr"])` trước khi tạo scheduler).
2. **Bug lớn: `forward(hard=True)` không nhị phân hoá giá trị literal**, chỉ cứng hoá cấu trúc (chọn concept/AND-OR/NOT) còn giá trị vẫn là xác suất liên tục → AND (product t-norm) nhân dồn nhiều số <1 khiến **class có công thức càng sâu (nhiều AND lồng) càng bị co giá trị theo cấp số nhân**, gây lệch so sánh giữa các class không liên quan gì tới input thật. Đã fix: ép `values = (values > 0.5).float()` sau khi áp polarity, trước khi combine AND/OR ở chế độ hard.

## 4. Kết quả training 30 epoch (checkpoint SkateFormer, stream `j`)

- `neural` (mốc, backbone frozen): ~92.4-92.5%, plateau ngay từ epoch 1 (do warm-start từ backbone pretrained + backbone đóng băng).
- `tree_soft`: 92.35% — bám sát neural, KHÔNG yếu hơn bản cũ (`-bandacaitien` fused đạt ~90-92%).
- `tree_hard` (sau khi fix bug nhị phân hoá): **13.40%** (naive argmax, gây hiểu lầm).
- **Coverage thật** (sau fix): `unique-fire=3.3%`, `no-fire=0.0%`, **`multi-fire=96.7%`**, **`accuracy-when-unique=97.99%`**, `hybrid_accuracy=92.35%`.
- → Diễn giải: khi luật quyết định RÕ RÀNG (chỉ 1 class thoả), gần như luôn đúng (97.99%) — chứng tỏ literal học được có ý nghĩa thật. Nhưng đa số công thức là AND của nhiều `NOT concept` (điều kiện phủ định), quá dễ thoả mãn cho rất nhiều sample khác nhau → nhiều class cùng "khớp" → naive argmax (tie-break về class chỉ số nhỏ) cho ra accuracy thấp giả tạo.

## 5. VẤN ĐỀ ĐANG MỞ — cần làm tiếp ở session sau

**Nghi vấn chính**: một số concept head (`arm_raise`, `hip_extend`, `temporal_direction_motion_downward`, `leg_static_sit`, `temporal_dynamics_speed_slow`...) có thể đã **collapse** — luôn output xác suất gần 0.00 bất kể input (không phải tín hiệu thật theo từng sample), khiến các literal `NOT concept_đó` trở thành "ăn gian" luôn đúng.

**Bằng chứng đã thấy** (từ case study, chưa có con số thống kê đầy đủ):
- `class_0` (drink_water), `class_1` (eat_meal): mọi literal đều `=T(0.00)` giống hệt nhau qua nhiều sample khác nhau — rất giống collapsed head.
- `brush_teeth` bị đoán sai thành `drink_water` — khớp đúng cơ chế: rule `drink_water` (toàn NOT-của-concept-nghi-collapse) gần như luôn đúng cho MỌI sample → khi tie, argmax chọn class chỉ số nhỏ nhất (0).
- Ngược lại `class_2` (`foot_jump`=0.76), `class_3` (`leg_kick`=0.71, `leg_jump`=0.91...) có giá trị dao động thật — KHÔNG collapse, chỉ là chọn concept không liên quan ngữ nghĩa (vấn đề khác, nhẹ hơn).

**Việc cần làm ngay khi tiếp tục**:
1. Đã thêm đoạn code chẩn đoán vào `eval_rulebook.py` Step 2 (tính mean/std từng concept trên toàn bộ `concept_vecs` đã thu thập, cảnh báo nếu `std < 0.02`) — **CHƯA CÓ kết quả từ user**, cần chạy lại `eval_rulebook.py` (không cần train lại) và lấy đúng đoạn `[WARN] N/74 concept(s) have near-zero variance...`.
2. Tuỳ số lượng concept bị flag:
   - **Ít** (vài concept): sửa không cần train lại — loại các concept variance thấp ra khỏi ứng viên `argmax` của leaf-selector khi hardening (thêm mask vào `FuzzyLogicTree.harden()`/`export_formula()`, ép chọn trong số concept "còn sống").
   - **Nhiều** (vấn đề hệ thống): phải sửa từ gốc lúc train — tăng trần `pos_weight` clamp (hiện `[1,20]` trong `joint_symbolic_model.py`, có thể chưa đủ cho concept quá hiếm), hoặc thêm regularizer chống collapse (phạt concept có variance thấp trên batch), rồi train lại.

**Insight quan trọng cho luận văn** (dù hướng nào cũng nên giữ): kiến trúc cũ (soft blending toàn bộ 74 concept qua attention) che giấu được concept head yếu vì trung bình hoá; kiến trúc mới (Fuzzy Tree, bắt buộc `argmax` chọn cứng 1 concept/lá) **phơi bày** điểm yếu này ra ngoài — đây là cái giá thật của việc theo đuổi interpretability dạng luật rời rạc, và bản thân phát hiện "hardening lộ ra concept nào yếu mà soft accuracy không thấy được" là 1 luận điểm đáng đưa vào phần thực nghiệm.

## 6. Lệnh tham khảo nhanh

```bash
# Train (resume nếu bị ngắt, nhớ --start-epoch đúng epoch tiếp theo)
python train_joint.py --config config/ntu/skateformer/ntu_cs_j.yaml \
    --resume-checkpoint <checkpoint_dang_co> --start-epoch <N+1> --epochs 30

# Harden + so sánh soft/hard/coverage
python harden_tree.py --config config/ntu/skateformer/ntu_cs_j.yaml \
    --checkpoint <joint_best_hoac_final.pt> --data <NTU60_CS.npz> --eval-split test

# So sánh đầy đủ 4 điểm + ensemble + case study + chẩn đoán concept collapse
python eval_rulebook.py --config config/ntu/skateformer/ntu_cs_j.yaml \
    --checkpoint <checkpoint> --data <NTU60_CS.npz> \
    --kmeans-k 3 --max-per-class 100 --tree-km-sweep --case-study --n-case-per-class 2
```

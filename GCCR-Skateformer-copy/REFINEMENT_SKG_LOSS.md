# Ghi chú Cải tiến: Áp dụng $S_{KG}$ Loss trực tiếp lên Nhánh Symbolic

- **Ngày ghi chú**: 12/09/2026
- **Trạng thái**: Chờ thực nghiệm (Pending Implementation & Ablation)
- **Tập tin liên quan**: [`joint_symbolic_model.py`](file:///d:/Thesis/GCCR-Skateformer/joint_symbolic_model.py)

---

## 1. Bối cảnh & Vấn đề Hiện tại

Trong kiến trúc **GCCR-Skateformer** (`JointSpatialTemporalCBM`), mô hình gồm hai nhánh ra:
- **Nhánh Neural (`neural_logits`)**: Do backbone SkateFormer + Linear head đảm nhiệm. Nhánh này thường được load từ pretrained weights sắc nét (được huấn luyện trước với one-hot Cross-Entropy).
- **Nhánh Symbolic (`symbolic_logits`)**: Do các Concept Group Heads kết hợp với bộ chấm điểm tương thích `symbolic_action(group_probs)` đảm nhiệm.
- **Dự đoán kết hợp**: 
  $$\text{action\_logits} = (1 - \alpha) \cdot \text{neural\_logits} + \alpha \cdot \text{symbolic\_logits}$$

### Vấn đề trong code hiện tại ([`joint_symbolic_model.py:L356-L360`](file:///d:/Thesis/GCCR-Skateformer/joint_symbolic_model.py#L356-L360)):
```python
# HIỆN TẠI: loss_s đang áp lên action_logits (fused)
loss_s = F.kl_div(
    F.log_softmax(action_logits.float(), dim=-1),
    s_target,
    reduction="batchmean",
)
```

1. **Xung đột mục tiêu (Gradient Conflict) ở nhánh Neural**:
   - $S_{KG}$ là phân phối xác suất mềm (soft distribution) mô tả độ tương đồng ngữ nghĩa giữa các lớp.
   - Nhánh neural vốn đã học dự đoán one-hot sắc nét. Việc ép $S_{KG}$ vào `action_logits` vô tình cưỡng ép nhánh neural phải "nhòe" (soften) dự đoán, có thể làm giảm Top-1 accuracy vốn có của backbone.
2. **Lệch pha trong chiến lược Curriculum Warm-up**:
   - Ở các epoch đầu, `fusion_alpha = 0.10` (90% neural, 10% symbolic).
   - Do đó, **90% gradient của $S_{KG}$ bị đổ vào nhánh neural**, trong khi các Concept Heads của nhánh Symbolic (nơi thực sự cần tri thức ngữ nghĩa này để định hình concept) lại nhận được rất ít gradient.

---

## 2. Đề xuất Cải tiến (Refinement)

Chuyển mục tiêu giám sát của $S_{KG}$ từ `action_logits` sang **`symbolic_logits`**:

$$\mathcal{L}_{S\_KG} = D_{KL}\Big(\text{Softmax}(\text{symbolic\_logits}) \parallel \overline{S}_{KG}[\text{label}]\Big)$$

### Lợi ích khoa học:
1. **Phân định rạch ròi trách nhiệm (Disentanglement)**:
   - Nhánh Neural: Chuyên trách phân loại tối ưu độ chính xác (`CrossEntropy`).
   - Nhánh Symbolic: Chuyên trách biểu diễn quan hệ ngữ nghĩa ontology và suy luận logic.
2. **Bổ trợ cho `ranking_loss` của Symbolic**:
   - Trong `symbolic_loss`, hàm ranking margin đối xử với mọi lớp âm như nhau. Áp $S_{KG}$ giúp nhánh Symbolic hiểu được cấu trúc ưu tiên: nhầm `drink_water` sang `eat_meal` (cùng cử chỉ tay-miệng) được chấp nhận tốt hơn nhầm sang `jump_up`.
3. **Tối ưu 100% Gradient cho Concept Heads**:
   - Toàn bộ gradient của $S_{KG}$ sẽ truyền trực tiếp qua `symbolic_action` về 9 `group_heads`, buộc các concept không gian và thời gian phải kích hoạt hài hòa.

---

## 3. Thay đổi Code Cụ thể

### File: `d:\Thesis\GCCR-Skateformer\joint_symbolic_model.py`

```diff
-        loss_s = F.kl_div(
-            F.log_softmax(action_logits.float(), dim=-1),
-            s_target,
-            reduction="batchmean",
-        )
+        loss_s = F.kl_div(
+            F.log_softmax(symbolic_logits.float(), dim=-1),
+            s_target,
+            reduction="batchmean",
+        )
```

*(Chỉ cần thay `action_logits.float()` thành `symbolic_logits.float()` tại dòng 357).*

---

## 4. Kế hoạch Thực nghiệm & Đánh giá (Ablation Study)

Khi triển khai sửa, sẽ thực hiện bài đo so sánh:

| Cấu hình | Mục tiêu $S_{KG}$ | Top-1 Neural Acc | Top-1 Symbolic Acc | Top-1 Fused Acc |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (Hiện tại)** | `action_logits` | ... | ... | ... |
| **Refined (Đề xuất)** | `symbolic_logits` | ... | ... | ... |

- **Kỳ vọng**: 
  - Accuracy của nhánh `symbolic` tăng rõ rệt.
  - Nhánh `neural` giữ được độ sắc nét và ổn định hơn.
  - Kết quả `fused` và độ chính xác phân loại bằng `kmeans_rb` (Rule Book) tốt hơn.

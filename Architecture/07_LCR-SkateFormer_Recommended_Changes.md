# LCR-SkateFormer — Các thay đổi cần thực hiện

Tài liệu này ghi rõ các thay đổi được đề xuất cho **LCR-SkateFormer** dựa trên mã nguồn hiện tại trong `LCR-SkateFormer/model/lcr_skateformer.py`.

## 1. Ưu tiên rất cao — cần sửa trước khi dùng làm mô hình chính

### 1.1. Sửa stability loss để đúng với mục tiêu multi-loop

Hiện tại `LCR-SkateFormer` chỉ tính:

```python
stability_loss = (c_final - c0).abs().mean()
```

Như vậy không kiểm soát được dao động giữa các vòng trung gian. Cần lưu toàn bộ các trạng thái concept `z^0, z^1, ..., z^K` rồi tính trung bình sai khác giữa từng cặp liên tiếp. Nên bổ sung metric `concept_delta_per_loop` để kiểm tra reasoning có hội tụ hay không.

### 1.2. Đảm bảo prototype mode thật sự là đường quyết định chính

Mô hình chính nên dùng:

```yaml
action_mode: prototype
```

Prototype phải được kiểm tra về đúng thứ tự 72 concept, đúng số action của NTU-60/NTU-120 và đúng ngữ nghĩa ontology. `linear` mode nên giữ làm baseline/ablation. Nếu dùng `linear` làm kết quả chính thì không nên tuyên bố quyết định action hoàn toàn dựa trên logic có thể diễn giải.

### 1.3. Sửa ngữ nghĩa của các quan hệ KG

Trong `kg_loss`, quan hệ `require` hiện được diễn giải theo dạng `b` cần `a`. Cần ghi rõ quy ước này trong ontology và kiểm thử từng quan hệ bằng unit test.

Phải phân biệt rõ:

- `require(a, b)`: concept nào là điều kiện và concept nào là hệ quả;
- `alongside(a, b)`: đồng xuất hiện mềm, không phải bắt buộc;
- `excludes(a, b)`: phạt đồng thời xuất hiện, không phải phủ định tuyệt đối.

Nên báo cáo `KG violation rate` trước và sau huấn luyện để chứng minh model học tri thức thay vì chỉ học lại tương quan dữ liệu.

## 2. Ưu tiên cao — tăng tính giải thích và độ tin cậy

### 2.1. Thêm action prototype có trọng số require/support/exclude

Prototype hiện chủ yếu là ma trận concept nhị phân. Nên thay bằng schema cho từng action gồm ba nhóm:

```text
required concepts
supporting concepts
excluded concepts
```

Điểm action nên tách được thành `positive evidence - contradiction penalty`, để phần giải thích chỉ rõ concept nào ủng hộ và concept nào mâu thuẫn với dự đoán.

### 2.2. Thêm kiểm tra counterfactual có định lượng

Test hiện tại chỉ kiểm tra rằng can thiệp concept làm output thay đổi. Cần bổ sung action drop của lớp liên quan khi xóa concept bắt buộc, action increase khi bật concept hỗ trợ, contradiction penalty khi bật concept loại trừ và so sánh với random concept intervention.

Explanation chỉ nên được xem là có giá trị khi can thiệp vào concept quan trọng tạo ra thay đổi có hướng và có ý nghĩa, không chỉ tạo ra delta khác 0.

### 2.3. Sửa reasoning trace để không giữ computation graph

Khi `return_trace=True` trong lúc training, trace có thể giữ tensor cùng computation graph và làm tăng VRAM. Nên chỉ lưu trace đầy đủ khi evaluation, hoặc dùng `detach()`/CPU khi logging; trường hợp thông thường chỉ lưu top-k concept/rule.

## 3. Ưu tiên trung bình — cải thiện kiến trúc

### 3.1. Thay Product T-norm bằng phép tính ổn định hơn

`KGUnionLayer` dùng phép tích cho AND/OR. Với nhiều literal, tích nhiều giá trị nhỏ có thể làm gradient yếu. Nên làm ablation giữa Product T-norm, log-space product, Łukasiewicz/soft-min-soft-max và temperature annealing.

Các cấu hình cần được so sánh cùng accuracy, concept calibration và KG violation rate.

### 3.2. Bổ sung nhánh neural residual tùy chọn

Hiện `fused_feature` được tạo trong `LCR-SkateFormer` nhưng chưa được nối vào action output. Có thể bổ sung biến thể:

```text
y_final = y_prototype + alpha * y_neural
```

Phải tách rõ hai cấu hình:

- **Strict prototype LCR-SkateFormer:** action chỉ đi qua prototype reasoning;
- **Residual LCR-SkateFormer:** có neural residual, dùng để đo trade-off accuracy–explainability.

Không nên trộn hai cấu hình khi báo cáo kết quả.

### 3.3. Cân nhắc action feedback như một ablation riêng

`use_action_feedback` mặc định là `false`. Nếu bật, action logits tác động ngược vào concept update, có thể tạo vòng lặp tự củng cố và làm giảm tính độc lập của concept evidence. Cần báo cáo riêng cấu hình có/không action feedback cùng concept delta và counterfactual fidelity.

## 4. Ưu tiên về huấn luyện và thực nghiệm

### 4.1. Huấn luyện theo giai đoạn

Nên dùng ba giai đoạn:

1. warm-up backbone và concept head với `K=1` hoặc beta nhỏ;
2. bật KG reasoner và tăng dần `beta`;
3. fine-tune end-to-end với learning rate nhỏ cho backbone.

Không nên bật đồng thời tất cả loss ở cường độ lớn ngay từ epoch đầu.

### 4.2. Ablation bắt buộc

| ID | Cấu hình | Mục đích |
|---|---|---|
| L0 | SkateFormer | Baseline perception |
| L1 | ECR/one-step reasoning | Đo hiệu quả suy luận một vòng |
| L2 | LCR, K=2, không KG prior | Đo ảnh hưởng của KG |
| L3 | LCR, K=2, prototype | Mô hình chính đề xuất |
| L4 | LCR, K=3, prototype | Đo lợi ích của vòng thứ ba |
| L5 | LCR, shared/unshared | Đo ảnh hưởng của chia sẻ tham số |
| L6 | LCR, beta=0 | Kiểm tra sự cần thiết của refinement |
| L7 | LCR + random prototype/KG | Kiểm tra model có thực sự dùng tri thức hay không |

### 4.3. Metric cần báo cáo

Ngoài Top-1/Top-5 action accuracy, **LCR-SkateFormer** cần báo cáo concept macro-F1, micro-F1, AUPRC, Brier score hoặc ECE, KG violation rate, action drop dưới concept intervention, số tham số, FLOPs, latency, VRAM và accuracy/stability theo từng loop `K`.

## 5. Thứ tự triển khai khuyến nghị

1. Sửa stability loss và trace memory.
2. Kiểm tra ontology, prototype và hướng quan hệ KG.
3. Chạy ablation K=1/2/3, shared/unshared và beta=0.
4. Bổ sung counterfactual fidelity và KG violation rate.
5. Sau đó mới thử temperature annealing hoặc neural residual.

## Kết luận cho LCR-SkateFormer

**LCR-SkateFormer** nên được giữ làm hướng nghiên cứu chính vì iterative concept reasoning có tính mới và dễ kiểm chứng hơn ECR một bước. Tuy nhiên, trước khi dùng làm mô hình chính của luận văn, cần sửa stability loss, kiểm chứng prototype/KG, định lượng counterfactual explanation và tách rõ strict prototype khỏi neural-residual variant.

# Đề xuất nghiên cứu

## ECR-SkateFormer: Evidence-Calibrated Reasoning SkateFormer

### 1. Quyết định chiến lược

**Không nên đổi sang một kiến trúc hoàn toàn mới.** Trục SkateFormer + concept bottleneck + differentiable reasoning đã đủ khác biệt để làm nền tảng nghiên cứu. Vấn đề không nằm ở việc thiếu thêm một module, mà ở chỗ cần biến kiến trúc hiện tại từ một chuỗi module hợp lý thành một **luận đề khoa học có thể bác bỏ và kiểm chứng**.

Định hướng nên được tái định vị là:

> **Concept-grounded, uncertainty-aware và KG-regularized reasoning cho skeleton action recognition, với explanation có thể kiểm chứng bằng phản thực (counterfactual).**

Đóng góp trung tâm không phải là “thêm nhiều hidden nodes”, mà là chứng minh rằng dự đoán hành động:

1. đi qua các concept có ý nghĩa;
2. sử dụng các quan hệ KG có thể truy nguyên;
3. ổn định khi thay đổi người thực hiện, dữ liệu hoặc một phần khớp;
4. thay đổi có lý do khi một concept quan trọng bị loại bỏ.

### 2. Ba thay đổi quyết định

#### 2.1. Prototype-constrained reasoning

`action = Linear(hidden)` vẫn có thể học một ánh xạ black-box từ hidden nodes tới lớp hành động. Vì vậy, action classifier không nên là một lớp tuyến tính tự do trong phiên bản chính.

Mỗi action nên có một **action prototype** hoặc schema được xây dựng từ các concept liên quan. Điểm của action được tính từ độ tương thích giữa concept evidence và schema của action:

$$
s_y = \operatorname{Compat}(c, \mathcal{S}_y, W_{KG})
$$

Trong đó $\mathcal{S}_y$ là tập concept bắt buộc, hỗ trợ và loại trừ của action $y$. Có thể giữ linear action head như một baseline hoặc residual ablation, nhưng không dùng nó làm nguồn quyết định chính nếu muốn tuyên bố interpretability.

**Giải thích phải là một phần của cơ chế dự đoán**, không chỉ là visualization được tạo sau khi mô hình đã dự đoán.

#### 2.2. Uncertainty-aware concept supervision

Các concept hiện được suy ra ở cấp action class, trong khi một video cụ thể có thể không biểu hiện đầy đủ mọi concept của lớp đó. Vì vậy không nên dùng nhãn nhị phân cứng như chân lý tuyệt đối.

Đề xuất dùng:

- target mềm $q_{ic} \in [0,1]$ theo độ tin cậy của concept;
- mask hoặc trọng số tin cậy cho concept chưa chắc chắn;
- asymmetric BCE hoặc focal-style loss cho mất cân bằng;
- báo cáo riêng các concept có độ tin cậy cao và thấp.

Nếu chưa có annotation theo từng video, phải gọi đây là **weakly supervised concept learning**, không gọi là concept ground truth đầy đủ.

#### 2.3. KG regularization thay cho chỉ KG initialization

KG-priming là cần thiết nhưng chưa đủ. Sau khi khởi tạo, cần bảo vệ cấu trúc KG bằng regularization:

$$
\mathcal{L}_{total} = \mathcal{L}_{action}
 + \lambda_c \mathcal{L}_{concept}
 + \lambda_{cons} \mathcal{L}_{KG-consistency}
 + \lambda_{prior} \|W-P\|_1
 + \lambda_{sparse}\|W\|_1.
$$

`require`, `alongside` và `excludes` phải được định nghĩa riêng về hướng và ngữ nghĩa. Không được xem cạnh `excludes` có prior thấp là một luật loại trừ cứng. Cần đo **KG violation rate** trước và sau huấn luyện để biết mô hình học tri thức hay chỉ quên prior.

### 3. Kiến trúc đề xuất

Không nên thêm một hierarchy bốn tầng lớn khi chưa chứng minh ba tầng hoạt động tốt:

1. **Atomic concepts:** 72 concept có tên và nguồn gốc ontology.
2. **Named functional schemas:** Upper-body, Lower-body, Posture, Dynamics; mask đầu vào cố định hoặc được kiểm soát.
3. **Action prototypes:** schema của từng action và điểm tương thích cuối cùng.

Các hidden node không có tên chỉ nên được dùng trong ablation. Nếu giữ hidden node, phải có group mask, sparsity và kiểm tra counterfactual để ngăn concept bypass. `fused_feature` cũng không được nối trực tiếp tới action output trong mô hình chính, nếu không sẽ tồn tại đường tắt giải thích.

### 4. Hàm mất mát

Không nên bắt đầu bằng “hard penalty” hoặc các ngưỡng slack cố định $0.15/0.70$. Các ngưỡng này dễ tạo vùng không được huấn luyện và làm calibration kém.

Phiên bản đầu nên dùng:

$$
\mathcal{L}_{concept}
 = \frac{1}{Z}\sum_{i,c} r_{ic}\,
\operatorname{AsymBCE}(z_{ic},q_{ic}),
$$

với $r_{ic}$ là độ tin cậy của nhãn. Sau đó thêm dần:

- KG consistency loss dạng mềm;
- prior-retention loss để chống prior drift;
- sparsity/entropy loss để luật ngắn và dễ đọc;
- calibration loss nếu concept probability được dùng như evidence.

Nên huấn luyện theo ba giai đoạn: concept warm-up, bật reasoning, rồi fine-tune end-to-end với learning rate nhỏ cho backbone. Điều này cho phép phân biệt lỗi do perception, concept hay reasoning.

### 5. Bộ thí nghiệm bắt buộc

| Mô hình | Mục đích |
|---|---|
| SkateFormer gốc | Baseline hiệu năng |
| SkateFormer + concept head | Đo chi phí của concept bottleneck |
| Concept + MLP reasoner | Kiểm tra KG có thực sự cần thiết không |
| Concept + rule bank không prior | Đo tác dụng của KG priming |
| Concept + KG regularization | Đo tác dụng của việc bảo toàn tri thức |
| Prototype-constrained full model | Phiên bản đề xuất chính |
| Full model + free residual/action head | Ablation về đường tắt black-box |

Các phép đo không được chỉ gồm action accuracy. Cần có:

- action accuracy/top-1 và top-5;
- concept macro-F1, micro-F1, AUPRC, Brier score hoặc ECE;
- tỷ lệ vi phạm `require`/`excludes`;
- rule sparsity, coverage và stability giữa các seed;
- deletion/insertion fidelity của explanation;
- action drop khi xoá concept quan trọng;
- robustness dưới occlusion theo từng nhóm khớp;
- few-shot performance và chi phí tính toán.

### 6. Tiêu chuẩn thành công và tiêu chuẩn dừng

Đề xuất chỉ được gọi là thành công nếu đồng thời đạt ba điều kiện:

1. Không giảm đáng kể action performance so với SkateFormer gốc, hoặc có trade-off được định lượng rõ.
2. Explanation faithful: xoá evidence được nêu phải làm giảm score tương ứng, còn xoá evidence không liên quan không được làm giảm mạnh.
3. KG giúp cải thiện ít nhất một trong các chỉ số consistency, few-shot hoặc occlusion mà không làm concept calibration suy giảm nghiêm trọng.

Nếu prototype-constrained reasoning làm accuracy giảm mạnh và không cải thiện fidelity, phải giữ action head hiện tại như baseline và kết luận trung thực rằng KG chủ yếu hỗ trợ regularization, không tuyên bố symbolic decision-making đầy đủ.

### 7. Những hướng chưa nên làm

- Không dùng LLM để tự sinh KG trong đóng góp chính.
- Không tăng số tầng hidden node chỉ để tạo cảm giác “hierarchical”.
- Không cho phép một dense residual bypass concept bottleneck.
- Không dùng prior drift tự do mà không đo khoảng cách $W-P$.
- Không dùng accuracy concept tổng thể làm bằng chứng interpretability.
- Không khẳng định Q1 trước khi hoàn thành ablation, fidelity và robustness.

### 8. Lộ trình thực hiện ưu tiên

1. Khóa ontology 72 concept và kiểm tra mapping với CSV/code.
2. Viết rõ hướng của từng quan hệ KG và tạo action prototype.
3. Sửa action output để prototype-constrained reasoning có thể chạy độc lập.
4. Thay BCE cứng bằng confidence-weighted asymmetric concept loss.
5. Thêm prior-retention, sparsity và KG consistency loss.
6. Chạy ma trận ablation tối thiểu ở mục 5 với nhiều random seed.
7. Xây dựng counterfactual explanation và bảng fidelity.
8. Chỉ sau đó mới cân nhắc adaptive KG hoặc LLM-assisted ontology như future work.

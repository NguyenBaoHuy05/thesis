## 15. Ghi chú kết quả ablation NTU60-CS v2.3

Các run đều dùng cùng protocol `full_train_then_test_once`, train đến epoch 15,
và kết quả dưới đây lấy từ test split.

| Run | Thành phần | Top-1 (%) | Micro-F1 (%) | Macro-F1 (%) | Exact match (%) |
|---|---|---:|---:|---:|---:|
| A0 | Baseline | 92.2302 | 95.6186 | 86.3179 | 85.8919 |
| A1 | Region pooling | 92.2788 | 95.6169 | 86.3550 | 85.5826 |
| A2 | Motion branch | 90.6108 | 95.4714 | 86.0990 | 84.5151 |
| A3 | Region pooling + motion branch | 90.5623 | 95.4771 | 86.1328 | 84.2482 |
| A4 | Bounded scope fusion | **92.3152** | **95.6864** | **86.4009** | **86.0254** |

### Nhận xét tạm thời

- `A1_region` gần như không cải thiện so với baseline; region pooling riêng lẻ
  chưa cho thấy lợi ích rõ ràng.
- `A2_motion` làm giảm Top-1 khoảng 1.62 điểm phần trăm và giảm exact match
  khoảng 1.38 điểm phần trăm. `A3_region_motion` cũng không khắc phục được
  hiện tượng này, nên motion branch hiện chưa chứng minh được hiệu quả trong
  thiết lập hiện tại.
- `A4_bounded_scope` đạt kết quả action-level tốt nhất. Cải tiến đến từ việc
  thay fixed scope fusion bằng bounded, learnable per-concept gate:
  `sigmoid(gate) * tanh(scope_logits) / temperature`. Cách này giới hạn ảnh
  hưởng của scope logits và giữ scope term cùng thang đo với prototype logits.
- Mức cải thiện của A4 còn nhỏ và mới được kiểm tra trên một seed. Vì vậy cần
  chạy thêm nhiều seed trước khi kết luận về ý nghĩa thống kê.

### Lưu ý cần kiểm tra

`A2_motion` và `A3_region_motion` có `parent_micro_f1` khoảng 88.24%,
`parent_macro_f1` khoảng 83.88% và `parent_exact_match` bằng 0%, trong khi
những run còn lại có parent micro-F1 khoảng 98.72% và parent exact match khoảng
86%. Cần kiểm tra lại parent prediction/threshold, checkpoint loading và metric
pipeline trước khi sử dụng các parent metrics này trong luận văn.

## 16. Kế hoạch decoder V3 trên NTU120-CSub

Từ V3, parent branch chỉ còn auxiliary supervision, hierarchy regularization và
đóng góp trực tiếp trong symbolic score; parent probability không còn nhân vào
concept probability. Artifact mới chỉ dùng `concept_probs` và
`parent_for_child`, còn evaluator vẫn đọc được `raw_concept_probs` của artifact
v2 cũ.

Ma trận chạy được khóa trong `config/train/ablation_ntu120_csub_v31/`.
Tất cả run dùng ontology NTU120 v2.4 gồm 120 action/102 concept, checkpoint
SkateFormer NTU120-CSub và official CSub train/test split:

| Run | Temporal pooling | Scale fusion | Parent auxiliary |
|---|---|---|---|
| V2-R | legacy noisy-OR | single final scale | bật |
| V3-A | adaptive MIL | single final scale | bật |
| V3-B | legacy noisy-OR | per-concept multi-scale | bật |
| V3-C | adaptive MIL | per-concept multi-scale | bật |
| V3-D | adaptive MIL | per-concept multi-scale | tắt |

Các run dùng cùng seed, loss weights, stage schedule và final-epoch protocol;
chưa ghi kết quả cho đến khi có môi trường PyTorch cùng dữ liệu/checkpoint
NTU120-CSub. `V3-D` chỉ chạy sau khi `V3-C` hoàn tất.

Chạy toàn bộ ma trận theo đúng thứ tự:

```bash
python scripts/train_ntu120_csub_v31.py
```

Script tự bỏ qua run đã có `final_model.pt`, ghi log riêng và dừng ngay
khi một run lỗi. Có thể chạy riêng một phần ma trận, kiểm tra lệnh hoặc
train lại bằng:

```bash
python scripts/train_ntu120_csub_v31.py --runs V3_A V3_B V3_C
python scripts/train_ntu120_csub_v31.py --dry-run
python scripts/train_ntu120_csub_v31.py --runs V3_C --force
```

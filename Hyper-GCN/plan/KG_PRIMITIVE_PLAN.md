# Hướng 1 — Knowledge Graph hành động + primitive chuyển động học được

> Trạng thái: **kế hoạch, chưa thực nghiệm** (ưu tiên sau hướng 2).
> Liên quan: [SEMANTIC_HYPERGRAPH_PLAN.md](SEMANTIC_HYPERGRAPH_PLAN.md) — dùng chung bank synergy,
> chú thích LLM `annotations/class_synergy.json` và hạ tầng đánh giá trung thực.

## 0. Ý tưởng

Tri thức đi từ trên xuống, bằng chứng đi từ dưới lên:

- **Bằng chứng (neural):** với mỗi bộ phận / synergy và mỗi segment, mẫu được gán mềm vào một
  trong K **primitive chuyển động** học được (prototype trên đặc trưng backbone). Primitive có
  **tên** nhờ đo các đoạn gần nó nhất bằng teacher hình học (teacher đổi vai: từ giám sát sang
  đặt tên).
- **Tri thức (symbolic):** một **knowledge graph** (KG) nối primitive → cử động nguyên tử → hành
  động, cùng quan hệ thời gian giữa các cử động. KG do LLM đề xuất, người kiểm tra.
- **Suy luận:** message passing trên KG với trọng số cạnh khởi tạo từ tri thức và được học có ràng
  buộc. Lời giải thích = **đường đi** từ primitive quan sát được tới hành động dự đoán.

Khác hướng 2: hướng 2 hỏi *"nhóm khớp nào, lúc nào"*; hướng 1 trả lời thêm *"cử động gì, theo
tri thức nào"* và hỗ trợ zero-shot.

## 1. Câu hỏi nghiên cứu

| | Câu hỏi | Kiểm chứng |
|---|---|---|
| RQ1 | Primitive học được có thuần nghĩa không? | Độ thuần mô tả hình học, độ nhất quán giữa seed |
| RQ2 | Suy luận qua KG có giữ độ chính xác? | Top1 so với linear trên primitive và backbone |
| RQ3 | Đường giải thích có trung thực? | Deletion các nút trên đường so với ngẫu nhiên |
| RQ4 | Tri thức có giúp ích? | KG thật so với KG xáo trộn; zero-shot trên lớp chưa thấy |
| RQ5 | Mô hình sửa tri thức thế nào? | Tỷ lệ cạnh KG được giữ / bị bỏ / cạnh mới được học |

## 2. Kiến trúc

```
skeleton → backbone pretrained (ĐÓNG BĂNG; tùy chọn fine-tune l9 ở cuối)
  → pooling theo synergy × segment (bank của hướng 2)          f[e,s] ∈ R^C
  → Primitive layer: K prototype / synergy, cosine + softmax     p[e,s,k] ∈ Δ^K
  → KG reasoning:
        primitive (e,k) ──realizes──▶ movement m   (ma trận R, học, prior từ đặt tên)
        movement m ──indicates──▶ action c         (ma trận D, prior từ LLM)
        movement m ──precedes──▶ movement m'        (quan hệ thời gian giữa segment)
  → logit_c = Σ_m D[m,c]·EV_s(u[m,s]) + Σ_{m→m'} T[m,m',c]·SEQ(u[m], u[m'])
        với u[m,s] = Σ_{e,k} R[(e,k),m]·p[e,s,k]
```

### 2.1 Primitive layer (kiểu ProtoPNet, theo synergy)

- K = 16 (ablation 8 / 32) prototype cho mỗi synergy → ~24 × 16 = 384 primitive.
- Khởi tạo bằng k-means trên đặc trưng đã cache; huấn luyện với CE + cluster loss + separation loss.
- **Chiếu prototype** về đoạn huấn luyện thật gần nhất sau mỗi vài epoch → mỗi primitive có một
  "ví dụ minh họa" thật, điều kiện cần để giải thích bằng ví dụ có nghĩa.
- **Đặt tên tự động:** với 200 đoạn gần mỗi primitive nhất, tính các mô tả hình học của teacher
  hiện có (hướng di chuyển, gập/duỗi, gần đầu/chân, tốc độ). Tên = các mô tả có tần suất cao vượt
  ngưỡng, vd. `hand_R–head.p7 = "tay phải đi lên, gần đầu, chậm"`. Độ thuần = tần suất của mô tả
  chính. Primitive có độ thuần thấp được gắn nhãn "chưa đặt tên".

### 2.2 Knowledge graph

| Loại nút | Số lượng | Nguồn |
|---|---|---|
| Synergy / bộ phận | 24 | Bank của hướng 2 |
| Primitive | ~384 | Học (2.1) |
| Cử động nguyên tử (movement) | ~60–100 | LLM đề xuất từ 120 tên lớp, gộp trùng, người duyệt. Vd. "đưa tay lên miệng", "cúi người", "hai tay chạm nhau", "bước lùi" |
| Hành động | 120 | NTU120 |

| Loại cạnh | Ý nghĩa | Nguồn | Trong mô hình |
|---|---|---|---|
| movement–involves→synergy | cử động dùng nhóm khớp nào | LLM + duyệt | Mặt nạ cho R (primitive chỉ hiện thực cử động của đúng synergy) |
| primitive–realizes→movement | primitive hiện thực cử động | Học, prior từ tên tự động | R, mềm, thưa |
| movement–indicates→action | hành động gồm cử động nào | LLM + duyệt (~3–6 / lớp) | D, khởi tạo từ KG, học với phạt lệch khỏi prior |
| movement–precedes→movement (theo lớp) | thứ tự điển hình | LLM ("ngồi xuống: gập gối → hạ hông") | T, thưa, chỉ trên các cặp có trong KG |

- KG lưu dạng `knowledge/ntu120_kg.json` (nút, cạnh, nguồn, người duyệt), có script kiểm tra:
  mỗi lớp có ≥ 2 cử động, mỗi cử động thuộc ≥ 1 synergy, không có cạnh trỏ tới nút không tồn tại.
- Phạt lệch khỏi prior: `λ_kg·‖D − D_prior‖₁` + L1 trên phần ngoài KG → cạnh mới chỉ xuất hiện
  khi dữ liệu đủ mạnh, và có thể đếm được (RQ5).

### 2.3 Suy luận và giải thích

- `EV_s` / `SEQ` dùng lại toán tử thời gian đã có trong `model/logic_reasoner.py` (log-space, ổn
  định số) — nhưng trên ~100 cử động thay vì 1443 literal.
- Lời giải thích của mẫu = đường có đóng góp lớn nhất:
  `arm_R–head.p7 (seg 2) → "đưa tay lên miệng" → drink_water`, kèm ví dụ minh họa của p7.

## 3. Đánh giá

| Nhóm | Metric |
|---|---|
| Độ chính xác | Top1/Top5 X-Sub, X-Set |
| Primitive | Độ thuần, tỷ lệ primitive đặt tên được, Jaccard giữa 3 seed |
| Trung thực | Deletion / insertion trên nút của đường giải thích so với ngẫu nhiên |
| Tri thức | Tỷ lệ cạnh KG giữ / bỏ / thêm; precision@k cử động theo lớp so với KG |
| Zero-shot | Chia 110 / 10 và 96 / 24 lớp của NTU120 (theo các bài zero-shot skeleton phổ biến — kiểm tra lại giao thức chính xác trước khi báo cáo): lớp chưa thấy chỉ có cạnh KG, không có dữ liệu |

## 4. Baseline và ablation

| Mã | Cấu hình | Mục đích |
|---|---|---|
| B0 | Backbone gốc | Trần |
| B1 | Linear trên p[e,s,k] (bỏ KG) | Trần của primitive |
| B2 | Mô hình hướng 2 (M) | So sánh hai hướng |
| A1 | KG xáo trộn (giữ bậc của nút) | RQ4 — tri thức có giúp thật? |
| A2 | Bỏ cạnh precedes | Giá trị của thứ tự thời gian |
| A3 | D học tự do (λ_kg = 0) | Giữ tri thức vs độ chính xác |
| A4 | K ∈ {8, 16, 32} | Độ mịn primitive |
| A5 | Không chiếu prototype | Ảnh hưởng tới độ thuần và trung thực |

## 5. Lộ trình

| Pha | Việc | Ước tính |
|---|---|---|
| P0 | Cache đặc trưng l9 gộp synergy × segment cho train/test (một lần chạy backbone) | ~1 giờ GPU, ~vài GB đĩa (fp16) |
| P1 | Primitive layer + B1 trên cache; đặt tên tự động; đo độ thuần | 1–2 ngày, GPU rất ít |
| P2 | Xây KG: prompt LLM, gộp cử động, duyệt tay, script kiểm tra | 2–3 ngày (chủ yếu công người) |
| P3 | KG reasoning, huấn luyện trên cache | 1–2 ngày |
| P4 | Đánh giá trung thực, zero-shot, ablation | 2–3 ngày |
| P5 | (Tùy chọn) fine-tune l9 end-to-end | ~3 giờ GPU |

Tiêu chí dừng sớm: nếu B1 (linear trên primitive) < 70% thì xem lại K / lớp đặc trưng trước khi
xây KG; nếu A1 (KG xáo trộn) ngang mô hình đầy đủ thì tri thức không đóng góp — chuyển trọng tâm
sang zero-shot, nơi KG là bắt buộc.

## 6. Rủi ro

| Rủi ro | Xử lý |
|---|---|
| KG từ LLM sai / thiếu | Duyệt tay, ghi nguồn từng cạnh, A1 đo mức đóng góp thật |
| Primitive không thuần, khó đặt tên | Chiếu prototype, tăng K, báo cáo tỷ lệ "chưa đặt tên" |
| Giao thức zero-shot không khớp tài liệu | Đọc lại bài gốc trước khi chạy; báo cáo rõ giao thức đã dùng |
| Khối lượng công việc lớn | Dùng chung bank, chú thích và công cụ đánh giá với hướng 2; làm sau khi hướng 2 ổn |

## 7. Quan hệ với hướng 2

Hướng 2 cung cấp: bank synergy, `class_synergy.json`, công cụ deletion/insertion, và (nếu
fine-tune) đặc trưng l9 đã "thấm" synergy — có thể dùng làm đầu vào cho primitive layer.
Luận văn có thể trình bày hướng 2 là kết quả chính và hướng 1 là phần mở rộng "từ *ở đâu, khi nào*
sang *cái gì, vì sao*".

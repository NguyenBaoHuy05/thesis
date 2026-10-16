# A1 — Retrieve → Verify → Fuse: kế hoạch triển khai

> Trạng thái: **kế hoạch, chưa triển khai**.
> Tài liệu cha: [KG_RETRIEVAL_PLAN.md](../KG_RETRIEVAL_PLAN.md) (mục A1, §1 thành phần dùng chung, §4 ablation K0–K6).
> Xây KG: [KG_CONSTRUCTION_LLM_PLAN.md](../KG_CONSTRUCTION_LLM_PLAN/KG_CONSTRUCTION_LLM_PLAN.md).
> Liên quan: [KG_PRIMITIVE_PLAN.md](../../KG_PRIMITIVE_PLAN.md),
> [OLD_SEMANTIC_HYPERGRAPH_PLAN.md](../../OLD_SEMANTIC_HYPERGRAPH_PLAN.md) (P1: nhóm ngẫu nhiên ≈ nhóm ngữ nghĩa).
>
> Vai trò: A1 là kiến trúc **đầu tiên** — dựng toàn bộ hạ tầng dùng chung (KG, embedding, đầu thuộc
> tính, retrieval, bộ ablation) mà A2–A5 tái sử dụng, và trả lời câu hỏi gác cổng:
> **KG thật có hơn KG xáo trộn không?** (K2 — đúng điểm P1 đã thất bại).

## 0. Câu hỏi nghiên cứu của A1

| | Câu hỏi | Kiểm chứng | Ablation |
|---|---|---|---|
| Q1 | Head A1 có giữ độ chính xác của backbone? | Top1 so với K0 (backbone + FC) | K0 |
| Q2 | Tri thức KG có đóng góp thật? | KG thật > KG xáo trộn giữ bậc | **K2** |
| Q3 | Bước VERIFY có đóng góp? | Bỏ `v_c`, bỏ `d_attr` | K3, A1-d |
| Q4 | Chiều symbolic → neural có cần? | Bỏ `fc(h)` | K4 |
| Q5 | Giải thích có trung thực? | Can thiệp thuộc tính → dự đoán đổi đúng hướng | K5 |
| Q6 | KG cho khả năng neural không có? | Zero-shot 110/10, 96/24 | K6 |

## 1. Kiến trúc chi tiết

```
skeleton ─► Hyper-GCN large (ĐÓNG BĂNG, tới l9) ─► mean(T,V), max qua actor hợp lệ ─► z ∈ R^C   (C = 512)
                                                                                    (cache một lần)
z ─► q = P z  (Linear C→D, D = chiều text embedding)
     s_c = cos(q, e_c) / τ                    với mọi c ∈ 120 lớp, e_c cố định (offline)
     Cand = top-k(s)                          k = 10 (ablation 5 / 20)

G_q = Cand ∪ {attr: cạnh has/lacks tới Cand} ∪ {láng giềng similar/opposite của Cand}

a = sigmoid(W_a z + b_a) ∈ [0,1]^A            (đầu thuộc tính, A ≈ 50)

VERIFY
  W[c,attr] ∈ {−1, −½, 0, ½, 1}               has_attr: +w/2, lacks_attr: −w/2, w ∈ {1,2}
  π_c      = softmax_{c∈Cand}(s_c)            (trọng số ứng viên)
  d_attr   = std_π( W[·,attr] ) / max_attr std      ∈ [0,1]   (mọi ứng viên giống nhau → 0)
  v_c      = Σ_attr W[c,attr] · d_attr · (2a_attr − 1)

FUSE
  node(G_q): action  → U_a e_c + type_emb[action]
             attr    → U_t e_attr + a_attr · u_val + type_emb[attr]
             neighbor→ U_a e_c + type_emb[neighbor]
  h        = CrossAttn(query = U_z z, keys/values = node(G_q))      1 lớp, 4 head, d = 256, có mặt nạ G_q
  f_c      = ⟨W_h h, U_a e_c⟩ / √d               (song tuyến với embedding lớp → vẫn zero-shot được)

logit_c = s_c + λ·v_c + f_c        c ∈ Cand          (λ = softplus(λ̃), khởi tạo 1)
logit_c = −∞                       c ∉ Cand
```

Quyết định thiết kế:

- **`fc(h)` phải song tuyến với `e_c`**, không phải `Linear(d, 120)`: một FC 120 lớp không mở rộng được
  sang lớp chưa thấy và phá Q6.
- **Mặt nạ trong CrossAttn** chỉ cho chú ý vào node của `G_q`, không vào toàn KG → `h` thật sự phụ thuộc
  vào subgraph được truy xuất (điều kiện R1 của tài liệu cha).
- **Tính toán theo batch**: KG nhỏ (≈ 120 + 50 node) → lưu dạng ma trận dày `W ∈ R^{120×A}`,
  `S_sim, S_opp ∈ {0,1}^{120×120}`; `G_q` là mặt nạ `[N, 120 + A]`, không cần thư viện đồ thị.
- **Phân rã logit** là chính xác: `logit_c = s_c + λ v_c + f_c`, và `λ v_c` tách tiếp theo từng
  thuộc tính → đây là lời giải thích, không cần post-hoc.

### 1.1 Huấn luyện

```
L = CE_cand(logit, y)                      ← mục tiêu chính, trên tập ứng viên
  + α · CE_all(s, y)                       ← giữ retrieval tốt (α = 0.5)
  + β · BCE_masked(a, y_attr(y))           ← giám sát yếu thuộc tính (β = 0.5)
  + γ · mean(|f|) / mean(|s| + |λ v|)      ← tùy chọn, chống fc(h) lấn át (γ = 0 mặc định, xem rủi ro)
```

- `y_attr(y)`: 1 nếu `has_attr`, 0 nếu `lacks_attr`, **bỏ qua** (mặt nạ) nếu KG không nói gì về cặp đó.
- **GT ngoài top-k khi train**: ép GT vào `Cand` (thay ứng viên điểm thấp nhất). Khi test không ép →
  trần của Top1 là **Recall@k**, phải báo cáo cạnh Top1. Biến thể đối chứng: không ép, mất mẫu
  khỏi `CE_cand` (chỉ `CE_all` học).
- `τ` học được, kẹp `[0.01, 1]`. Optimizer Adam, lr 1e-3, cosine, 30 epoch trên cache, batch 512.
- Gradient của top-k: chọn tập là rời rạc (không cần gradient), gradient đi qua `s_c` của ứng viên
  và qua `π` trong `d_attr`.

### 1.2 Huấn luyện trên đặc trưng cache

Backbone đóng băng hoàn toàn ⇒ `z` cố định cho mỗi mẫu → **cache một lần** rồi train head trên cache:

- ~63k train + ~50k test (X-Sub) × 512 × fp16 ≈ 115 MB. Train một head: vài phút, chạy được nhiều seed
  và toàn bộ ablation trên Kaggle.
- Mất augmentation `random_rot`: cache **V = 5 view** xoay ngẫu nhiên cho tập train (≈ 320 MB),
  mỗi epoch lấy ngẫu nhiên một view. Test: một view, transform như `test_feeder_args`.
- Cache thêm logit của `fc` gốc → K0 có sẵn, không chạy lại.

## 2. Knowledge graph `annotations/action_kg.json`

> Quy trình xây KG bằng LLM, prompt và lược đồ đầy đủ (thay cho lược đồ rút gọn dưới đây):
> [KG_CONSTRUCTION_LLM_PLAN.md](../KG_CONSTRUCTION_LLM_PLAN/KG_CONSTRUCTION_LLM_PLAN.md).

```json
{
  "status": "LLM draft + hand-check, xem 'reviewed'",
  "attributes": {"touch_face": {"text": "the hand touches the face", "group": "contact"}, "...": {}},
  "actions": {
    "drink_water": {
      "text": "a person drinks water from a cup",
      "has":  {"touch_face": 2, "hold_object": 2, "tilt_head_back": 2},
      "lacks": {"repetitive": 1, "lower_body_motion": 2},
      "similar": ["eat_meal", "brush_teeth"], "opposite": [],
      "reviewed": true
    }
  },
  "attribute_relations": {"implies": [["touch_face", "upper_body_motion"]], "excludes": [["two_person", "..."]]}
}
```

- Tên lớp dùng đúng khóa của `class_synergy.json` / `NTU_ACTION_NAMES` (`model/concept_schema.py`).
- **Thuộc tính không phải tên synergy**: tránh trùng lặp với bank synergy để K2 không bị lẫn với
  kết quả P1.
- Cặp đảo thời gian (put_on/take_off, sit_down/stand_up) sẽ có tập thuộc tính gần giống nhau —
  **chấp nhận**: A1 không được kỳ vọng tách chúng (đó là việc của A5). Đánh dấu `opposite` để đo riêng.
- `tools/validate_kg.py`: mỗi lớp ≥ 3 `has` và ≥ 1 `lacks`; mọi thuộc tính dùng ≥ 2 lớp và ≤ 100 lớp;
  không cạnh trỏ node lạ; `similar` đối xứng; báo cáo các cặp lớp có **vector thuộc tính trùng nhau**
  (không phân biệt được bằng VERIFY).
- `tools/shuffle_kg.py --seed s`: xáo trộn cạnh `has/lacks` **giữ bậc** của cả lớp và thuộc tính
  (edge swap), giữ trọng số; cùng cho `similar`. Sinh 3 KG xáo trộn cho K2.

## 3. Embedding text offline

`tools/build_kg_embeddings.py --kg annotations/action_kg.json --encoder mpnet --out annotations/kg_emb_mpnet.npz`

- Chứa `action[120, D]`, `attribute[A, D]` (chuẩn hóa L2), cùng thứ tự khóa của JSON + hash của KG
  để head kiểm tra khớp khi load.
- Encoder chính: sentence-transformer `all-mpnet-base-v2` (D = 768) — văn bản là câu mô tả hành động
  và định nghĩa thuộc tính, cần độ tương đồng ngữ nghĩa giữa các câu; không bị giới hạn 77 token của
  CLIP nếu sau này muốn nhúng cả trường `motion`.
  CLIP text ViT-B/32 (D = 512) chạy làm ablation A1-enc, để so được với các bài skeleton zero-shot dùng CLIP.
- Chạy ở máy local / Colab, commit file `.npz` → Kaggle không cần text encoder.
- **K1** (chỉ text, không KG) dùng cùng file: head chỉ có `s_c`.

## 4. Mã nguồn

| File | Nội dung |
|---|---|
| `model/kg.py` | Load + validate KG, dựng `W`, `S_sim`, `S_opp`, nhãn thuộc tính `y_attr`, mặt nạ "biết/không biết"; split lớp seen/unseen |
| `model/kg_retrieval.py` | `KGRetrievalHead`: retrieve, `G_q`, verify, fuse; trả về dict `logits, s, v, f, a, d, cand, contributions[N,k,A]` (như `SynergyOutput`) + `explain()` + `audit()` |
| `tools/cache_features.py` | Chạy `Model.features(x, 'l9')` đóng băng → `z`, `fc_logits`, nhãn; `--views 5` cho train |
| `tools/build_kg_embeddings.py`, `tools/validate_kg.py`, `tools/shuffle_kg.py` | §2, §3 |
| `train_kg_head.py` | Train/eval head trên cache; config YAML; ghi `history.json`, `kg_metrics.json`, `explanations.jsonl` vào `work_dir` |
| `evaluate_kg.py` | Can thiệp (K5), zero-shot (K6), phân rã đóng góp, Recall@k |
| `config/kg/a1*.yaml` | Mặc định + một file / ablation (cùng khuôn với `config/large/.../synergy*.yaml`) |
| `tests/test_kg_retrieval.py` | Xem §4.1 |

Không sửa `main.py` ở giai đoạn này: head train trên cache nên script riêng gọn hơn. Chỉ khi cần
fine-tune l9 end-to-end (P6) mới gắn head vào `hypergcn_large.Model` theo mẫu `synergy`
(thêm tiền tố `kg.` vào `is_extension_key` trong `model/training.py`).

### 4.1 Test đơn vị

- `logit_c == s_c + λ v_c + f_c` trên ứng viên; `−∞` ngoài ứng viên.
- `d_attr = 0` khi mọi ứng viên cùng giá trị thuộc tính → thuộc tính đó không đổi `v`.
- Can thiệp đúng dấu: tăng `a_attr` mà `W[c,attr] > 0` thì `v_c` tăng.
- Mặt nạ CrossAttn: đổi embedding của node ngoài `G_q` không đổi `h`.
- Thêm lớp mới vào KG (không train lại) → head trả logit 121 chiều, không lỗi shape.
- `shuffle_kg` giữ đúng bậc mọi node.

## 5. Đánh giá

| Nhóm | Metric |
|---|---|
| Độ chính xác | Top1 / Top5 X-Sub (X-Set nếu kịp); Recall@k của retrieval; trung bình ± std 3 seed |
| Đóng góp | Tỷ lệ `|s|`, `|λv|`, `|f|` trung bình trên lớp dự đoán; % mẫu mà bỏ `λv` đổi dự đoán |
| Thuộc tính | AUC trên nhãn suy từ KG (**chỉ để chẩn đoán**, không phải bằng chứng grounding — nhãn đến từ nhãn lớp) |
| Trung thực (K5) | Với top-2 ứng viên, lấy thuộc tính có `|W[c1]−W[c2]|·d` lớn nhất, đảo `a` (0↔1): tỷ lệ dự đoán đổi **về đúng lớp kia**; so với đảo một thuộc tính ngẫu nhiên và với KG xáo trộn |
| Zero-shot (K6) | Train head trên lớp seen, thêm node unseen vào KG lúc test. ZSL top1 trên unseen; GZSL: seen / unseen / harmonic mean. Kiểm tra lại giao thức chia lớp của các bài skeleton ZSL trước khi báo cáo |
| Nhầm lẫn | Top-25 cặp nhầm (`analyze_confusion.py --compare` với K0): A1 sửa được cặp nào, hỏng cặp nào |

## 6. Ablation

Bắt buộc (từ tài liệu cha) + riêng của A1:

| ID | Cấu hình | Trả lời |
|---|---|---|
| K0 | Backbone + FC gốc (logit cache) | Mốc |
| K1 | Chỉ `s_c` (text embedding), không KG | KG có hơn text thuần? |
| **K2** | KG xáo trộn giữ bậc ×3 seed | **Tri thức thật có nghĩa?** |
| K3 | Bỏ VERIFY (`λ = 0`) | Suy luận có đóng góp? |
| K4 | Bỏ `f_c` (symbolic → neural) | Cần chiều ngược lại? |
| K4' | Chỉ `s + λv` **và** không ép GT, `λ` cố định | Biến thể "symbolic thuần" để trình bày giải thích |
| A1-d | `d_attr ≡ 1` | Độ phân biệt trong subgraph có cần? |
| A1-k | k ∈ {5, 10, 20, 120} | k = 120 ⇒ không còn "retrieval" |
| A1-enc | CLIP vs sentence-transformer | Độ nhạy theo text encoder |
| A1-kg | KG chỉ LLM vs đã duyệt tay | Công duyệt tay có đáng? |

Thứ tự chạy: K0, K1 → mô hình đầy đủ → **K2** → (quyết định, §7) → K3, K4, A1-* → K5, K6.

## 7. Lộ trình

| Pha | Việc | Ước tính | Tiêu chí qua |
|---|---|---|---|
| P0 | `cache_features.py`, cache X-Sub (5 view train) trên Kaggle | ~1–2 giờ GPU | Top1 của `fc_logits` cache khớp checkpoint gốc (± 0.1%) |
| P1 | KG: prompt LLM sinh thuộc tính + cạnh cho 120 lớp, gộp trùng, `validate_kg.py`; duyệt tay ≥ 40 lớp (ưu tiên các lớp trong top-25 cặp nhầm) | 2–3 ngày (công người) | Validator sạch; ≤ 10 cặp lớp trùng vector thuộc tính ngoài các cặp đảo thời gian |
| P2 | Embedding offline, K1 (chỉ retrieval) | 0.5 ngày | Recall@10 ≥ 98% (nếu thấp: tăng k hoặc thêm `CE_all`) |
| P3 | `KGRetrievalHead` + test + full model + K2 | 2 ngày | Xem cổng quyết định |
| P4 | K3, K4, A1-*; phân rã đóng góp; giải thích mẫu | 1–2 ngày | — |
| P5 | K5 can thiệp, K6 zero-shot | 2 ngày | — |
| P6 | (Tùy chọn) gắn head vào `main.py`, fine-tune l9 | ~3 giờ GPU | Chỉ nếu P3–P5 tích cực |

**Cổng quyết định sau P3** (so với KG xáo trộn, trung bình 3 seed):

- Full − K2 ≥ 0.5% Top1 **hoặc** K5 vượt rõ ràng KG xáo trộn → tri thức có ích: tiếp P4–P5, rồi A5.
- Full ≈ K2 về Top1 nhưng K6 (zero-shot) KG thật ≫ xáo trộn → trình bày A1 như mô hình zero-shot / giải
  thích, không tuyên bố cải thiện độ chính xác.
- Full ≈ K2 ở cả Top1, K5, K6 → lặp lại kết quả P1; dừng A1, xem lại chất lượng KG (A1-kg) trước khi
  đầu tư A5.
- Full < K0 quá 1% → kiểm tra Recall@k và tỷ lệ `|f|` trước khi kết luận.

## 8. Rủi ro

| Rủi ro | Dấu hiệu | Xử lý |
|---|---|---|
| `f_c` lấn át `v_c` → VERIFY chỉ là trang trí | `|f| ≫ |λv|`, K3 ≈ full | Báo cáo tỷ lệ đóng góp; chạy K4'; bật `γ` |
| Đầu thuộc tính chỉ học lại nhãn lớp (không grounding độc lập) | AUC thuộc tính cao nhưng K5 ≈ ngẫu nhiên | Chính là giới hạn đã nêu ở tài liệu cha — chỉ tuyên bố dựa trên K5/K6 |
| Retrieval bỏ sót GT | Recall@k < Top1 của K0 | Tăng k; `α` lớn hơn; báo cáo trần |
| Thuộc tính không phân biệt các cặp hay nhầm | Validator báo nhiều cặp trùng vector | Thêm thuộc tính có mục tiêu cho cặp nhầm (từ `analyze_confusion.py`) |
| Cache mất augmentation làm head overfit | Train ≫ test sau vài epoch | Thêm view, dropout trên `z`, giảm epoch |
| Lệch đường dẫn checkpoint Kaggle | Load lỗi / thiếu khóa | Checkpoint backbone ở `/kaggle/working/`, không theo `./pretrained/`; truyền `--weights` tường minh |

## 9. Sản phẩm cuối của A1 (tái sử dụng cho A2–A5)

- `annotations/action_kg.json` + validator + 3 KG xáo trộn.
- `annotations/kg_emb_*.npz`.
- Cache `z` X-Sub (và X-Set nếu kịp).
- `model/kg.py`, đầu thuộc tính đã train, `train_kg_head.py` / `evaluate_kg.py` với K0–K6.
- Bảng kết quả + 5–10 lời giải thích mẫu (đúng và sai) cho luận văn.

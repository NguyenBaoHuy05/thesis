# Hướng 3 — Knowledge Graph Retrieval: 5 kiến trúc đề xuất

> Trạng thái: **đề xuất, chưa triển khai**.
> Liên quan: [KG_PRIMITIVE_PLAN.md](KG_PRIMITIVE_PLAN.md) (KG + primitive),
> [SEMANTIC_HYPERGRAPH_PLAN.md](SEMANTIC_HYPERGRAPH_PLAN.md) (kết quả P1: nhóm ngẫu nhiên ≈ nhóm ngữ nghĩa).
>
> Ràng buộc: **không dùng predicate hình học**. Tri thức đến từ ngôn ngữ (LLM sinh, người kiểm tra),
> bằng chứng đến từ backbone Hyper-GCN.

## 0. Định nghĩa "retrieval" dùng trong tài liệu này

NTU120 chỉ có 120 lớp. Nếu retrieval chỉ trả về lớp gần nhất thì đó vẫn là phân loại bằng độ tương
đồng với embedding tên lớp (kiểu CLIP/GAP), không phải neuro-symbolic. Vì vậy mọi kiến trúc dưới
đây đều phải thỏa:

| # | Điều kiện | Nếu thiếu |
|---|---|---|
| R1 | Retrieval trả về **subgraph** (ứng viên + thuộc tính + quan hệ), không chỉ một nhãn | Chỉ là text-embedding classifier |
| R2 | Có **bước suy luận trên subgraph** với cạnh có kiểu **cố định** (không học tự do) | KG thành một GNN nữa, mất nghĩa ký hiệu |
| R3 | Tương tác **hai chiều**, mỗi chiều ablate được | "Tương tác" chỉ là lời khẳng định |

Lưu ý trung thực: không có hình học thì thuộc tính **không có grounding độc lập với nhãn** (nhãn
thuộc tính suy ra từ nhãn lớp qua KG). Do đó độ trung thực phải chứng minh bằng **can thiệp**
(sửa thuộc tính → dự đoán đổi đúng hướng) và **zero-shot**, không thể bằng độ chính xác thuộc tính.

## 1. Thành phần dùng chung

```
annotations/action_kg.json
  nodes:  action (120) · attribute (~50) · [exemplar, rule — tùy kiến trúc]
  edges:  action ─has_attr / lacks_attr─► attribute          (trọng số 1 | 2 như class_synergy.json)
          action ─similar / opposite / part_of─► action
          attribute ─implies / excludes─► attribute
```

- **Thuộc tính** (ví dụ): `touch_face`, `hold_object`, `repetitive`, `tilt_head_back`, `posture_change`,
  `two_person`, `contact_partner`, `lower_body_motion`, `bimanual`, `reverse_of_dressing`, ...
- **Text embedding** của mọi node: mã hóa offline một lần (`tools/build_kg_embeddings.py`, CLIP text
  ViT-B/32 hoặc sentence-transformer) → `.npy`. Train trên Kaggle không cần text encoder.
- **Backbone**: Hyper-GCN large, đóng băng; `z ∈ R^512` = pooling đặc trưng l9 (max qua actor).
- **Đầu thuộc tính**: `a = sigmoid(W_a z) ∈ [0,1]^A`, giám sát yếu bằng nhãn thuộc tính suy từ KG.
- **Retrieval mềm**: `s_c = cos(P z, e_c) / τ`; lấy top-k (k = 5–10) bằng softmax có mặt nạ
  (gradient chỉ qua ứng viên được chọn).

---

## A1. Retrieve → Verify → Fuse (kiến trúc gốc)

```
z ─► q ─► RETRIEVE top-k action ─► G_q = ứng viên ∪ thuộc tính (1 hop) ∪ láng giềng similar/opposite
                                     │
      a = đầu thuộc tính(z) ───────► VERIFY: v_c = Σ_attr  w(c,attr) · d_attr · (2a_attr − 1)
                                     d_attr = độ phân biệt của thuộc tính TRONG G_q
                                             (thuộc tính mọi ứng viên đều có → d ≈ 0)
                                     │
                                     ▼
      FUSE: h = CrossAttn(z, node(G_q))  →  logit_c = s_c + λ·v_c + fc(h)_c   (chỉ trên ứng viên)
```

- **Neural → symbolic**: q chọn subgraph. **Symbolic → neural**: h đưa tri thức subgraph về đặc trưng.
- **Giải thích**: "ứng viên {drink, brush_teeth, eat}; chọn drink vì `tilt_head_back`=0.9,
  `repetitive`=0.1" — chính là các số hạng của `v_c`.
- **Zero-shot**: thêm node action + cạnh, không train lại.
- **Chi phí**: rất nhỏ (head trên backbone đóng băng). **Rủi ro**: `fc(h)` có thể lấn át `v_c` →
  báo cáo tỷ lệ đóng góp và chạy biến thể không có `fc(h)`.

## A2. Retrieval nhiều bước lặp (neural ⇄ symbolic loop)

```
G_0 = top-k từ q_0 = P z
for t = 1..T (T = 2–3):
    m_t   = message passing trên G_{t-1} (cạnh có kiểu cố định, trọng số theo a)
    q_t   = GRU(q_{t-1}, readout(m_t))         ← neural cập nhật truy vấn từ tri thức
    G_t   = mở rộng G_{t-1} theo cạnh similar/opposite/implies, chọn node theo q_t
    dừng sớm nếu entropy ứng viên < ε
logit = readout(G_T)
```

- Tương tác **lặp lại nhiều vòng**, gần với Think-on-Graph / GraphRAG.
- **Giải thích**: **đường đi** qua KG (ví dụ: `hold_object` → {reading, writing} → `opposite`
  không áp dụng → `fine_hand_motion` cao → writing).
- **Zero-shot**: có. **Rủi ro**: chọn node rời rạc khó train (dùng Gumbel-softmax / straight-through),
  dễ bất ổn; T lớn không chắc giúp. Ablation riêng: T = 1 so với T = 3.

## A3. KG điều kiện hóa hyperedge của Hyper-GCN (hai lượt)

```
Lượt 1: l1..l9 (đóng băng) → z → RETRIEVE G_q → thuộc tính được kích hoạt
        mỗi thuộc tính có mặt nạ bộ phận cơ thể (tri thức, không phải đo hình học):
            touch_face → {tay, đầu}, lower_body_motion → {chân, hông}, ...
Lượt 2: l7..l9 chạy lại với  A = A_phys + α·G_dyn + β·G_KG(q)     (β khởi tạo 0)
        G_KG(q) = Σ_attr a_attr · hyperedge(mặt nạ của attr)
        → fc → logit
```

- **Symbolic đi vào bên trong neural** (Kautz kiểu 2/6), gắn chặt nhất với đóng góp Hyper-GCN.
- **Giải thích**: hyperedge nào do KG thêm vào và β·trọng số của nó.
- **Rủi ro lớn nhất**: chính là thất bại P1 — đặc trưng l7–l9 không cục bộ, nên hyperedge KG có thể
  không khác hyperedge ngẫu nhiên. Bắt buộc ablation mặt nạ ngẫu nhiên cùng kích thước.
  Chi phí: chạy lại 3 tầng cuối (~1.3× forward), cần fine-tune l7–l9.

## A4. Retrieval exemplar + KG (case-based)

```
KG mở rộng: node exemplar = prototype đặc trưng của mẫu train (K / lớp, k-means trên z)
            exemplar ─instance_of─► action ─has_attr─► attribute
RETRIEVE: kNN của z trong bank exemplar → lan truyền theo instance_of, has_attr
REASON:   điểm lớp = phiếu exemplar ⊕ độ khớp thuộc tính (như VERIFY ở A1)
```

- **Giải thích**: "giống các clip train #a, #b (lớp X), cùng chia sẻ thuộc tính Y".
- **Mạnh ở**: few-shot (thêm vài exemplar cho lớp mới), so sánh dễ với prototype network.
- **Zero-shot**: chỉ qua nhánh thuộc tính (không có exemplar). **Rủi ro**: phần exemplar là kNN thuần
  neural — phải chứng minh nhánh KG đóng góp thật (ablation bỏ cạnh has_attr).

## A5. Retrieval luật mờ có thứ tự thời gian (graph chứa rule)

```
KG lưu luật cho mỗi lớp (LLM đề xuất, người duyệt):
   put_on_jacket  :=  hold_object@đầu  ∧  arm_raise@giữa  ∧  ¬arm_raise@cuối
   take_off_jacket:=  arm_raise@đầu    ∧  hold_object@cuối
Đầu thuộc tính chạy THEO SEGMENT:  a[s] = sigmoid(W_a z_s),  s = 1..4
RETRIEVE: top-k lớp → lấy luật của chúng
EVALUATE: logic mờ khả vi (t-norm tích, ¬x = 1−x, "trước" = max_{s<s'} a[s]·b[s'])
logit_c = log(rule_c) + γ·s_c        (γ nhỏ, có cổng; báo cáo γ = 0)
```

- Trả lời câu hỏi ban đầu "rule hay graph": **rule sống trong graph, được retrieval chọn ra**.
  Kautz kiểu 6 — tuyên bố neuro-symbolic mạnh nhất trong 5 kiến trúc.
- **Mạnh ở** các cặp đảo thứ tự thời gian (put_on/take_off, sit_down/stand_up) mà A1–A4 khó tách.
- **Giải thích**: luật được thỏa và mức thỏa từng mệnh đề.
- **Rủi ro**: thuộc tính theo segment không có nhãn theo segment (giám sát rất yếu); viết/duyệt luật
  cho 120 lớp tốn công; trần độ chính xác có thể thấp như teacher hình học cũ (~46%) nếu γ = 0.

---

## 2. So sánh

| | A1 | A2 | A3 | A4 | A5 |
|---|---|---|---|---|---|
| Mức tích hợp (Kautz) | 3 + fuse | 2 lặp | 2/6 | 3 | 6 |
| Tương tác hai chiều | có | **nhiều vòng** | có | yếu | có |
| Giải thích | thuộc tính phân biệt | đường đi KG | hyperedge KG | exemplar + thuộc tính | **luật** |
| Zero-shot | có | có | một phần | chỉ thuộc tính | có (nếu có luật) |
| Công sức tri thức | thấp | thấp | trung bình (mặt nạ) | thấp | **cao** (luật) |
| Chi phí train | rất thấp | thấp | trung bình | thấp | thấp |
| Rủi ro kỹ thuật | thấp | cao | cao (P1) | thấp | trung bình |

## 3. Đề xuất thứ tự

1. **A1 trước** — dựng toàn bộ thành phần dùng chung (KG, embedding, đầu thuộc tính, retrieval) và
   bộ ablation. Mọi kiến trúc sau đều tái sử dụng.
2. **A5 làm đóng góp chính** nếu A1 cho thấy thuộc tính có ích (KG thật > KG xáo trộn).
   Bắt đầu với luật cho các cặp hay nhầm (lấy từ `analyze_confusion.py`), không cần đủ 120 lớp.
3. A2 / A3 / A4 chỉ làm nếu còn thời gian; A3 chỉ khi có bằng chứng đặc trưng l7–l9 đã cục bộ hơn.

## 4. Ablation bắt buộc (mọi kiến trúc)

| ID | Biến thể | Trả lời |
|---|---|---|
| K0 | Backbone + FC | Mốc độ chính xác |
| K1 | Chỉ embedding tên lớp, không KG | KG có hơn text thuần? |
| K2 | KG xáo trộn cạnh, giữ bậc | Tri thức thật có nghĩa? (điểm P1 đã thất bại) |
| K3 | Bỏ VERIFY / bỏ luật | Bước suy luận có đóng góp? |
| K4 | Bỏ chiều symbolic → neural | Tương tác hai chiều có cần? |
| K5 | Can thiệp thuộc tính | Giải thích có trung thực? |
| K6 | Zero-shot 110/10, 96/24 | KG cho khả năng mà neural không có? |

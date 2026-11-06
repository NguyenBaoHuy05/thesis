# Đề Xuất Nâng Cấp: CLIR-SkateFormer v2 (GA-CLIR)

> **Tên đầy đủ:** **G**rounded **A**daptive **C**oncept **L**ogic **I**terative **R**efinement
> **Quan hệ với `09`:** Đây là bản **nâng cấp trực tiếp trên chính CLIR-SkateFormer** (`d:\Thesis\CLIR-SkateFormer\`), không đổi codebase/backbone gốc — khác với `10_TRACE-SAR_Proposal.md` (hợp nhất với toàn bộ hạ tầng GRAFT-SAR). Giữ nguyên toàn bộ những gì CLIR đã làm tốt (part-grouped L1, AND2‖OR1 song song, vòng lặp logic-guided, rule extractor + Fidelity/Coverage/Precision) và vá đúng 4 khoảng trống còn lại đã xác định được **bằng cách đọc kỹ code hiện tại**, không suy đoán.
> **Backbone:** Giữ SkateFormer làm mặc định (đúng code hiện có); HyperGCN có thể thay thế dễ dàng cho Gap 1 vì HyperGCN vốn sinh sẵn đặc trưng theo từng bộ phận cơ thể (`hyper_feats` trong REASON) — hợp với thiết kế mới hơn cả SkateFormer.
> **Mã nguồn:** Sửa/thêm trực tiếp trong `d:\Thesis\CLIR-SkateFormer\model\` — không tạo repo mới.

---

## 0. 4 khoảng trống thật đã xác nhận trong code (không phải suy đoán)

| # | Khoảng trống | Bằng chứng trong code |
|---|---|---|
| **G1** | **Concept heads không thực sự "theo bộ phận cơ thể"** dù logic layer (L1) có ràng buộc part-grouped | `spatial_head`/`temporal_head` trong [clir_skateformer.py:401-412](../CLIR-SkateFormer/model/clir_skateformer.py#L401-L412) đều là MLP nhận **cùng một `spatial_feat`/`temporal_feat` toàn cục** (attention-pool trên *toàn bộ* V' khớp) rồi suy ra **cả 52/20 concept cùng lúc**. Tức là concept `head_tilt` và concept `leg_kick` được dự đoán từ **cùng một vector**, không có ràng buộc kiến trúc nào buộc `head_tilt` phải bắt nguồn từ vùng khớp đầu. Ràng buộc "part-grouped" chỉ tồn tại ở tầng logic (`conjunction_layer.py::get_part_groups`), **không tồn tại ở tầng nhận thức** — hai tầng bất nhất về mặt ngữ nghĩa "bộ phận cơ thể". |
| **G2** | **Không có ngôn ngữ/ngữ nghĩa cho concept** | Toàn bộ `clir_skateformer.py` không có CLIP/text encoder nào; concept chỉ là index số khớp với tên cột CSV, không dùng để giám sát/giải thích bằng ngôn ngữ tự nhiên. |
| **G3** | **Chuyển pha đột ngột ở epoch cố định** | `set_phase()` ([clir_skateformer.py:734-777](../CLIR-SkateFormer/model/clir_skateformer.py#L734-L777)) đổi `literal_mode`, `beta_max` (0.1→0.5), `lambda_reg` (0→0.005) **tức thời** tại đúng epoch `phase1_epochs`, và `CLIRTrainer.check_phase_switch()` trong notebook còn **tạo lại toàn bộ optimizer** (`_init_optimizer()`) ngay tại epoch đó — 2 cú sốc cộng dồn cùng lúc. |
| **G4** | **K cố định cho mọi mẫu, không thích ứng độ khó** | `IterativeLogicReasoner.forward()` ([clir_skateformer.py:219-273](../CLIR-SkateFormer/model/clir_skateformer.py#L219-L273)) chạy đúng `self.num_loops` vòng cho **mọi mẫu trong batch như nhau** — mẫu dễ và mẫu khó tốn compute như nhau, và `L_stable` (đo độ dao động vòng lặp) hiện chỉ dùng làm loss phụ, **không điều khiển** hành vi gì. |

---

## 1. G1 — `BodyPartGroundedConceptHeads` (vá gốc rễ, ưu tiên cao nhất)

**Xác nhận tính khả thi:** Đã kiểm tra `SkateFormer.py` — `PatchMergingTconv` ([SkateFormer.py:227-239](../CLIR-SkateFormer/model/SkateFormer.py#L227-L239)) dùng `Conv2d(kernel_size=(k,1), stride=(stride,1))`, **chỉ downsample theo chiều T**, chiều V (khớp × người) **không bao giờ bị gộp** qua toàn bộ backbone. Vậy `V'` trong `feature_map ∈ R^(B×D×T'×V')` vẫn ánh xạ 1:1 với chỉ số khớp gốc (`num_points × num_people`) — có thể cắt lát theo nhóm bộ phận **an toàn, không cần bảng ánh xạ lại theo scale**.

### Thiết kế
Thay vì 1 `spatial_pool_score` + 1 `spatial_head` dùng chung cho toàn bộ 52 concept không gian, tách theo đúng 6 nhóm bộ phận đã có sẵn trong `get_part_groups()` (`conjunction_layer.py`), nhưng áp dụng **từ tầng nhận thức**, không chỉ tầng logic:

```python
# model/clir_skateformer.py — thay thế self.spatial_pool_score / self.spatial_head

NTU25_BODY_PART_JOINTS = {  # chỉ số khớp NTU 25-joint chuẩn (0-indexed), nhân thêm cho num_people nếu M>1
    "head":  [2, 3, 20],
    "hand":  [7, 8, 11, 12, 21, 22, 23, 24],
    "arm":   [4, 5, 6, 9, 10],
    "hip":   [0, 1, 12, 16],
    "leg":   [13, 14, 17, 18],
    "foot":  [15, 19],
}  # NOTE: cần đối chiếu với concepts/ntu_rgbd_60_120.csv để khớp đúng thứ tự 6 nhóm/số concept mỗi nhóm

class BodyPartGroundedConceptHeads(nn.Module):
    """Mỗi nhóm bộ phận pool RIÊNG trên đúng tập khớp của mình rồi mới suy ra
    concept của nhóm đó — khớp 1-1 với part_groups ở tầng logic (conjunction_layer.py)."""
    def __init__(self, feature_dim, part_groups: dict, concepts_per_group: dict, dropout=0.1):
        super().__init__()
        self.part_groups = part_groups
        self.pool_scores = nn.ModuleDict({
            g: nn.Conv1d(feature_dim, 1, kernel_size=1) for g in part_groups
        })
        self.heads = nn.ModuleDict({
            g: nn.Sequential(
                nn.Linear(feature_dim, 128), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(128, concepts_per_group[g]),
            ) for g in part_groups
        })

    def forward(self, feature_map):  # [B, D, T', V']
        outs = []
        for g, joint_idx in self.part_groups.items():
            sub = feature_map[:, :, :, joint_idx].mean(dim=2)          # [B, D, |joints_g|] — pool theo T'
            attn = torch.softmax(self.pool_scores[g](sub), dim=-1)     # [B, 1, |joints_g|]
            pooled = (sub * attn).sum(dim=-1)                          # [B, D] — CHỈ từ khớp thuộc nhóm g
            outs.append(self.heads[g](pooled))                        # [B, concepts_g]
        return torch.cat(outs, dim=-1)                                 # [B, 52] — thứ tự phải khớp CSV
```

Nhóm `temporal` (20 concept) giữ nguyên cơ chế `temporal_pool_score`/`temporal_head` hiện có (pool theo V' toàn cục, hợp lý vì concept thời gian vốn không thuộc riêng bộ phận nào).

**Hệ quả tích cực kéo theo:** rule DNF trích ra sau này ("head_tilt ∧ hand_lift → …") giờ **có căn cứ nhận thức thật** — literal `head_tilt` được đảm bảo bắt nguồn từ đúng vùng khớp đầu về mặt kiến trúc, không chỉ vì nhãn CSV gọi tên nó vậy. Đây là khoảng trống quan trọng nhất vì nó ảnh hưởng trực tiếp đến độ tin cậy của toàn bộ tuyên bố "interpretable" của CLIR.

---

## 2. G2 — Semantic grounding nhẹ (không kéo theo LoRA CLIP đầy đủ như REASON/GRAFT-SAR)

Chỉ cần **CLIP text encoder đóng băng** (không LoRA fine-tune, không train text tower) — giảm tối đa chi phí/độ phức tạp so với REASON:

```python
class LightConceptGrounding(nn.Module):
    """CLIP đóng băng — chỉ học 1 projection nhỏ từ concept feature sang không gian CLIP."""
    def __init__(self, feature_dim, clip_dim=512, num_concepts=72):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(feature_dim, clip_dim), nn.LayerNorm(clip_dim))
        # text_anchors: [num_concepts, clip_dim] — encode 1 LẦN DUY NHẤT lúc khởi tạo
        # từ mô tả text của mỗi concept (cột "description" nếu concepts CSV có, hoặc tự sinh
        # template "a person is {concept_name}"), rồi register_buffer, KHÔNG forward CLIP lại
        # trong training loop.
        self.register_buffer("text_anchors", torch.zeros(num_concepts, clip_dim))

    def forward(self, part_features: torch.Tensor):  # [B, num_concepts_group, feature_dim] hoặc pooled
        skel_embed = F.normalize(self.proj(part_features), dim=-1)
        text_embed = F.normalize(self.text_anchors, dim=-1)
        return skel_embed, text_embed  # dùng cho contrastive loss + drift signal (xem G4 bên dưới)
```

Loss thêm (nhỏ, có trọng số riêng, không đổi cấu trúc `L_sup`/`L_reg` hiện có):
$$\mathcal{L}_{ground} = \text{MultiPositiveInfoNCE}(\text{skel\_embed}, \text{text\_embed}, \text{concept\_targets})$$

**Chi phí thực tế:** chỉ 1 lần forward CLIP text encoder cho toàn bộ concept CSV lúc khởi tạo model (vài trăm ms, không lặp lại mỗi batch) + 1 Linear+LayerNorm nhỏ — không tốn kém như LoRA fine-tuning của REASON.

---

## 3. G3 — Annealing liên tục thay set_phase() rời rạc

Sửa trực tiếp `set_phase(phase)` thành `set_progress(progress: float)` nội suy tuyến tính mọi tham số theo `progress = epoch / num_epoch ∈ [0,1]`, **không tạo lại optimizer**:

```python
def set_progress(self, progress: float):
    progress = float(max(0.0, min(1.0, progress)))
    # literal_mode: chuyển mượt qua nhiệt độ thay vì đổi hẳn 'soft'→'ste'
    self.reasoner.literal_mode = "ste"  # STE + nhiệt độ thấp ≈ soft ở giai đoạn đầu
    self.reasoner.temperature = 1.0 + progress * (10.0 - 1.0)      # 1.0 → 10.0 (thay vì switch cứng)
    self.reasoner.beta_max = 0.1 + progress * (0.5 - 0.1)          # 0.1 → 0.5
    self.lambda_reg = 0.0 if progress < 0.3 else 0.005 * (progress - 0.3) / 0.7
```

Lưu ý kỹ thuật: khi `temperature=1.0`, `BinarizeSTE` (`literal_layer.py`) đã xấp xỉ hành vi mượt gần giống `soft` (vì `sigmoid(1.0 * z)` không quá dốc) — nên có thể **bỏ hẳn mode `soft` riêng**, chỉ dùng `ste` với nhiệt độ khởi động thấp, hợp nhất 2 nhánh code hiện có thành 1 con đường liên tục. Cần kiểm tra thực nghiệm xem có tương đương đủ tốt với `soft` thật hay không trước khi xoá nhánh `soft`.

Optimizer trong `CLIRTrainer`/`main.py` cũng nên chuyển sang per-group LR **cố định ngay từ epoch 0** (giống cấu trúc phase-2 hiện tại) thay vì đổi param-group giữa chừng — loại bỏ hoàn toàn việc tạo lại `AdamW` tại một epoch cụ thể.

---

## 4. G4 — Adaptive Computation: vòng lặp dừng theo độ khó từng mẫu

Thêm 1 "halting head" nhỏ đọc trực tiếp từ Logic Evidence $r^t$ đã có sẵn (không cần thêm input mới), theo tinh thần Adaptive Computation Time (Graves, 2016) nhưng áp lên đúng vòng lặp K của CLIR:

```python
class HaltingUnit(nn.Module):
    def __init__(self, logic_dim=512):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(logic_dim, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, r_t):  # [B, 512]
        return torch.sigmoid(self.gate(r_t)).squeeze(-1)  # p_t ∈ (0,1) — xác suất "đủ tự tin, dừng ở vòng t"
```

Trong `IterativeLogicReasoner.forward()`, tích luỹ `p_t` theo đúng công thức ACT chuẩn (cumulative probability + remainder ở bước cuối cùng), dùng nó làm **trọng số trung bình** cho `y_all` thay vì chỉ lấy `y^K` cuối cùng làm dự đoán:

$$y_{\text{final}} = \sum_{t=0}^{N(b)} p_t^{(b)} \cdot y^t_{(b)}, \quad \text{N(b) = bước dừng của riêng mẫu } b$$

Thêm ponder cost nhỏ vào `L_reg`: $\mathcal{L}_{ponder} = \frac{1}{B}\sum_b (N(b) + R_{N(b)})$ để khuyến khích dừng sớm khi đủ tự tin — mẫu dễ dùng ít vòng, mẫu khó (concept mơ hồ) tự động dùng nhiều vòng hơn tới `K_max` (đặt `K_max=4`, cao hơn K=2 cố định hiện tại, vì giờ chi phí trung bình sẽ thấp hơn K_max cố định nhờ dừng sớm).

**Lợi ích giải thích được thêm:** số vòng lặp trung bình theo từng action class trở thành 1 chỉ số mới — action nào cần nhiều vòng tinh chỉnh hơn thường là action dễ nhầm lẫn (concept chồng lấn nhiều class), một tín hiệu chẩn đoán độc lập với Fidelity/Coverage/Precision đã có.

---

## 5. Các cải tiến nhỏ đi kèm (không bắt buộc, chi phí thấp)

- **Rule sparsity entropy-sharpened:** thêm entropy penalty $-p\log p - (1-p)\log(1-p)$ lên `sigmoid(weight_logit)` của `ConjunctionLayer`/`ConjunctionLayer2`/`DisjunctionLayer` (đẩy trọng số về gần {0,1}) cộng thêm vào `rule_sparsity_loss()` hiện tại — luật trích ra sẽ sắc nét hơn ở Phase 2.
- **LLM sanity-check cho rule đã trích (offline, không ảnh hưởng training):** mở rộng `scripts/extract_rules.py` — sau khi có `dnf_rules.txt` + text anchor từ G2, ghép câu luật thành ngôn ngữ tự nhiên rồi (tuỳ chọn) gọi 1 LLM để chấm điểm "có hợp lý về mặt vận động con người không" — thuần hậu xử lý, không đụng vào forward/train.

---

## 6. Hệ Loss tổng hợp (mở rộng từ CLIR gốc, không phá vỡ tương thích)

$$
\mathcal{L}_{\text{total}} = \underbrace{\mathcal{L}_{act}^{K} + \lambda_{sup}\mathcal{L}_{sup} + \lambda_{reg}(\mathcal{L}_{sparse} + \mathcal{L}_{stable} + \mathcal{L}_{ponder})}_{\text{CLIR gốc + ponder cost (G4)}} + \lambda_{ground}\mathcal{L}_{ground}\ \text{(G2)}
$$

`λ_ground` mặc định nhỏ (gợi ý 0.1, cùng bậc với `λ_sup` nhưng phụ) — không thêm cơ chế staged riêng, bật cùng lúc với `λ_reg` theo `progress` ở G3.

---

## 7. Ablation Studies

| ID | Cấu hình | Mục đích |
|---|---|---|
| **V0** | CLIR-SkateFormer gốc (baseline đã có, `09`) | Mốc so sánh |
| **V1** | + G1 (BodyPartGroundedConceptHeads) only | Đo riêng tác động của việc gắn concept vào đúng vùng khớp |
| **V2** | + G1 + G2 (grounding ngôn ngữ) | Đo tác động grounding lên Fidelity/Coverage khi rule đã có literal đáng tin từ nhận thức |
| **V3** | + G1 + G2 + G3 (annealing liên tục, bỏ set_phase) | Kiểm chứng loại bỏ "sốc" có cải thiện đường cong loss quanh epoch chuyển pha cũ không |
| **V4** | **Đầy đủ V1+V2+V3+G4 (Adaptive halting)** | **Mô hình chính đề xuất (GA-CLIR)** |
| **V5** | V4 nhưng halting ngẫu nhiên (random N(b)) | Kiểm chứng HaltingUnit học được tín hiệu thật, không phải nhiễu — cùng tinh thần A5 gốc |
| **V6** | V4 nhưng dùng HyperGCN thay SkateFormer | Kiểm tra backbone-agnostic; HyperGCN vốn có `hyper_feats` theo bộ phận nên có thể bỏ qua bước pool-theo-joint-index của G1, cắm thẳng |

---

## 8. Kế hoạch triển khai (2 tuần, vá trực tiếp trên CLIR-SkateFormer)

```
TUẦN 1: G1 + G3 (thay đổi kiến trúc + lịch huấn luyện, rủi ro thấp, không thêm dependency)
  [1] Đối chiếu concepts/ntu_rgbd_60_120.csv để lập đúng bảng {group: [concept_indices]} khớp NTU25_BODY_PART_JOINTS
  [2] model/clir_skateformer.py: thay spatial_pool_score/spatial_head bằng BodyPartGroundedConceptHeads
  [3] model/clir_skateformer.py: set_phase() → set_progress(); bỏ tạo lại optimizer giữa chừng trong main.py/notebook
  [4] Chạy V0 vs V1 vs V3 trên 1 fold nhỏ (vd NTU60 CS) để xác nhận không tụt accuracy trước khi làm tiếp

TUẦN 2: G2 + G4 (thêm dependency CLIP, thêm cơ chế mới)
  [5] model/crl/light_grounding.py: LightConceptGrounding (CLIP đóng băng + text anchor 1 lần)
  [6] model/crl/halting_unit.py: HaltingUnit + tích hợp ACT vào IterativeLogicReasoner.forward()
  [7] Cập nhật compute_sup_loss/compute_reg_loss thêm L_ground, L_ponder theo Mục 6
  [8] Chạy V4 đầy đủ, so Fidelity/Coverage/Precision + accuracy với V0, viết báo cáo so sánh vào 06_Comparison_and_Analysis.md
```

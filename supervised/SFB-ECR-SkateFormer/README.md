# 🚀 SFB-ECR-SkateFormer: Concept-Guided Soft Feedback Attention for Skeleton Action Recognition

> **Kiến trúc đề xuất dựa trên nghiên cứu:** [`RESEARCH/_NOTES.md`](file:///d:/Thesis/RESEARCH/_NOTES.md)  
> **Tên đầy đủ:** *ECR-SkateFormer with Concept-Guided Soft Feedback Attention and Symbolic Logical Reasoner*

---

## 📖 1. Giới thiệu & Động lực nghiên cứu

Trong các mô hình nhận dạng hành động skeleton hai giai đoạn (như REASON hay ECR-SkateFormer), dòng thông tin truyền đi theo hướng một chiều:
$$\text{Perception Backbone (SkateFormer)} \longrightarrow \text{Concept Decoders} \longrightarrow \text{Symbolic KG Reasoner} \longrightarrow \text{Action}$$

Cơ chế này có nhược điểm: **Perception backbone hoạt động hoàn toàn "mù" trước các khái niệm ngữ nghĩa (semantic concepts)** mà nó đang cố gắng dự đoán. Khi gặp các trường hợp khớp bị che khuất (occlusion), góc quay khó hoặc nhiễu động học, backbone không có cơ chế định hướng lại vùng chú ý không-thời gian.

### 💡 Giả thuyết cốt lõi (Core Hypothesis)
> *Các semantic concept (không gian và thời gian) được trích xuất sớm ở các tầng giữa của SkateFormer có thể đóng vai trò như tín hiệu dẫn đường (concept prior). Khi đưa tín hiệu này phản hồi mềm (Soft Feedback Attention) quay trở lại các khối xử lý tiếp theo, backbone sẽ tập trung sự chú ý vào các vùng khớp và khung hình liên quan mật thiết đến hành động, từ đó tăng độ chính xác, nâng cao độ bền vững (robustness) dưới che khuất và cải thiện chất lượng biểu diễn concept.*

---

## 🏗️ 2. Kiến trúc Tổng thể

```
Skeleton Input X [B, C_in, T, V, M]
        │
        ▼
SkateFormer Stem + Early Stages (Stage 0 & 1)
        │
        ├──► F_early [B, C_e, T_e, N_e]
        │        │
        │        ├──► Spatial Pooling Score ──────► c_spa^early [B, 52]
        │        ├──► Temporal Pooling Score ─────► c_tem^early [B, 20]
        │        │                                        │
        │        │                                        ▼
        │        │                              c_early = [c_spa, c_tem]
        │        │                                        │
        │        │             ┌──────────────────────────┴──────────────────────────┐
        │        │             ▼                                                     ▼
        │        │     Channel Modulation: g [B, C_e, 1, 1]              Token Soft Bias: B [B, 1, T_e, N_e]
        │        │             │                                                     │
        │        │             │                                      Confidence Gate: G [B, 1, 1, 1]
        │        │             │                                                     │
        │        └─────────────┴──────────────────┬──────────────────────────────────┘
        │                                         ▼
        ▼
F_modulated = F_early * (1 + β * g) + (λ_fb * G) * B * F_early
        │
        ▼
SkateFormer Late Stages (Stage 2 & 3)
        │
        ▼
F_late [B, C_l, T_l, N_l]
        │
        ├─────────────────────────────────────────┐
        ▼                                         ▼
Final Spatial Concept Head               Final Temporal Concept Head
c_spa^late [B, 52]                       c_tem^late [B, 20]
        │                                         │
        └───────────────────┬─────────────────────┘
                            ▼
            Concept Literals [c, 1 - c]
                            │
                            ▼
            KG-Primed Differentiable Reasoner
                            │
                            ▼
                Action Logits [B, Num_Classes]
```

---

## 🔬 3. Công thức Toán học & Module `SoftConceptFeedback`

### 3.1. Trích xuất Early Concept
Từ feature map trung gian $F_{\text{early}} \in \mathbb{R}^{B \times C_e \times T_e \times N_e}$ (với $N_e = V \times M$):
1. **Spatial Pooling:**
   $$s_{\text{spa}} = \frac{1}{T_e} \sum_{t=1}^{T_e} F_{\text{early}}[:, :, t, :] \in \mathbb{R}^{B \times C_e \times N_e}$$
   $$a_{\text{spa}} = \text{Softmax}(\text{Conv1D}_{1 \times 1}(s_{\text{spa}})) \in \mathbb{R}^{B \times 1 \times N_e}$$
   $$f_{\text{spa}} = \sum_{n=1}^{N_e} s_{\text{spa}}[:, :, n] \odot a_{\text{spa}}[:, 0, n] \in \mathbb{R}^{B \times C_e}$$

2. **Temporal Pooling:**
   $$s_{\text{tem}} = \frac{1}{N_e} \sum_{n=1}^{N_e} F_{\text{early}}[:, :, :, n] \in \mathbb{R}^{B \times C_e \times T_e}$$
   $$a_{\text{tem}} = \text{Softmax}(\text{Conv1D}_{1 \times 1}(s_{\text{tem}})) \in \mathbb{R}^{B \times 1 \times T_e}$$
   $$f_{\text{tem}} = \sum_{t=1}^{T_e} s_{\text{tem}}[:, :, t] \odot a_{\text{tem}}[:, 0, t] \in \mathbb{R}^{B \times C_e}$$

3. **Early Concept Prediction:**
   $$\hat{c}_{\text{early}} = \sigma\left( [\text{Linear}_{\text{spa}}(f_{\text{spa}}), \, \text{Linear}_{\text{tem}}(f_{\text{tem}})] \right) \in [0, 1]^{B \times 72}$$

### 3.2. Điều biến Kênh & Bias Mềm (Modulation & Soft Token Bias)
- **Channel Modulation:** $g = \sigma(\text{MLP}_{\text{chan}}(\hat{c}_{\text{early}})) \in \mathbb{R}^{B \times C_e \times 1 \times 1}$
- **Token Soft Bias:** $B_{\text{bias}} = \tanh\left(\text{Conv2D}_{1 \times 1}(F_{\text{early}}) + \text{MLP}_{\text{concept}}(\hat{c}_{\text{early}})\right) \in \mathbb{R}^{B \times 1 \times T_e \times N_e}$
- **Confidence Gate:** $G = \sigma(\text{Linear}_{\text{conf}}(\hat{c}_{\text{early}}.\text{detach}())) \in [0, 1]^{B \times 1 \times 1 \times 1}$
- **Feature Map sau Feedback:**
  $$F_{\text{mod}} = F_{\text{early}} \odot (1 + \beta \cdot g) + (\lambda_{\text{fb}} \cdot G) \odot B_{\text{bias}} \odot F_{\text{early}}$$

### 3.3. Multi-task Loss Function
$$\mathcal{L} = \mathcal{L}_{\text{action}} + \lambda_{\text{early}} \mathcal{L}_{\text{concept}}^{\text{early}} + \lambda_{\text{late}} \mathcal{L}_{\text{concept}}^{\text{late}} + \lambda_{\text{cons}} \mathcal{L}_{\text{cons}} + \lambda_{\text{KG}} \mathcal{L}_{\text{KG}} + \lambda_{\text{prior}} \mathcal{L}_{\text{prior}} + \lambda_{\text{sparse}} \mathcal{L}_{\text{sparse}} + \lambda_{\text{part}} \mathcal{L}_{\text{part}}$$

Trong đó:
- $\mathcal{L}_{\text{concept}}$: Asymmetric Loss giảm thiểu tác động của sự mất cân bằng concept.
- $\mathcal{L}_{\text{cons}} = \|\hat{c}_{\text{early}} - \hat{c}_{\text{late}}.\text{detach}()\|_2^2$: Khuyến khích tính nhất quán đa tầng.

---

## 🧪 4. Ma trận Nghiên cứu Đánh đổi (Ablation Matrix)

Hệ thống đã được thiết kế sẵn các file config YAML và flag tương ứng:

| ID | Tên cấu hình | Cơ chế Feedback | Gating | KG Reasoner | File Config |
|:---|:---|:---|:---:|:---:|:---|
| **A1** | ECR Baseline | Tắt (`disabled`) | ❌ | Có (Prototype) | `config/train/ntu_cs/ablation_A1_no_feedback.yaml` |
| **A2** | Early Head Only | Chỉ tính aux loss | ❌ | Có (Prototype) | `config/train/ntu_cs/ablation_A2_early_head_only.yaml` |
| **A3** | Channel Mod Only | Channel Scaling ($\beta=0.2$) | ❌ | Có (Prototype) | `config/train/ntu_cs/ablation_A3_channel_mod.yaml` |
| **A4** | **Proposed Full** | **Channel Mod + Token Bias** | **Learnable** | **Có (Prototype)** | **`config/train/ntu_cs/SFB_ECR_SkateFormer_j.yaml`** |
| **A5** | No Confidence Gate | Channel Mod + Token Bias | Cố định ($G=1.0$) | Có (Prototype) | `config/train/ntu_cs/ablation_A5_no_gate.yaml` |
| **A6** | Random Concept Proj | Nhiễu ngẫu nhiên $\sim \mathcal{N}(0, 1)$ | Learnable | Có (Prototype) | `config/train/ntu_cs/ablation_A6_random_proj.yaml` |
| **A7** | No KG Reasoner | Channel Mod + Token Bias | Learnable | Direct MLP | `config/train/ntu_cs/ablation_A7_no_kg.yaml` |

---

## 📁 5. Cấu trúc Thư mục

```text
d:\Thesis\SFB-ECR-SkateFormer/
├── SFB_ECR_SkateFormer_Colab.ipynb    # All-in-One Google Colab & Local Jupyter Notebook
├── README.md                          # Tài liệu hướng dẫn & công thức toán học
├── main.py                            # CLI Runner cho Train, Test, Occlusion, Counterfactual
├── verify_model.py                    # Bộ kiểm tra chẩn đoán toàn diện (6 Test Suites)
├── requirements.yaml                  # File môi trường Conda / Pip
├── pyproject.toml                     # Metadata đóng gói dự án
│
├── model/
│   ├── __init__.py
│   ├── SkateFormer.py                 # Backbone SkateFormer hỗ trợ Intermediate Extraction
│   ├── soft_concept_feedback.py       # Module SoftConceptFeedback (Early Pooling + Bias + Gate)
│   └── reason_skateformer.py          # SFB_ECR_SkateFormer tích hợp trọn vẹn
│
├── feeders/
│   ├── __init__.py
│   ├── feeder_ntu.py                  # PyTorch Dataset cho NTU-RGB+D (Joint/Bone/Motion)
│   ├── tools.py                       # Các phép biến đổi hình học và Augmentation
│   └── occlusion.py                   # Generator tạo che khuất khớp & temporal masking
│
├── concepts/
│   └── ntu_rgbd_60_120.csv            # Bảng Ontology 72 concept & Prototype ma trận
│
├── config/
│   ├── train/
│   │   ├── ntu_cs/                    # Cấu hình Joint/Bone/Ablations cho NTU Cross-Subject
│   │   └── ntu_cv/                    # Cấu hình cho NTU Cross-View
│   └── test/                          # Cấu hình đánh giá
│
└── torchlight/                        # Thư viện quản lý IO và Logging
```

---

## 🚀 6. Hướng dẫn Sử dụng

### 6.1. Chạy Kiểm tra Sanity Test (Khuyên dùng đầu tiên)
Kiểm tra tính toàn vẹn của gradient flow, forward pass và tất cả các chế độ ablation chỉ trong 10 giây:
```bash
python verify_model.py
```

### 6.2. Huấn luyện Mô hình Đề xuất (A4 - Joint Stream trên NTU-60 CS)
```bash
python main.py --config config/train/ntu_cs/SFB_ECR_SkateFormer_j.yaml
```

### 6.3. Huấn luyện các biến thể Ablation
```bash
# A1: Baseline ECR không feedback
python main.py --config config/train/ntu_cs/ablation_A1_no_feedback.yaml

# A3: Chỉ Channel Modulation
python main.py --config config/train/ntu_cs/ablation_A3_channel_mod.yaml

# A5: Không dùng Confidence Gate
python main.py --config config/train/ntu_cs/ablation_A5_no_gate.yaml
```

### 6.4. Đánh giá Tính Bền vững dưới Che khuất (Occlusion Robustness)
Đánh giá độ sụt giảm chính xác khi che khuất từng nhóm khớp (Arms, Legs, Trunk, Head) hoặc khung hình:
```bash
python main.py --config config/train/ntu_cs/SFB_ECR_SkateFormer_j.yaml --phase test_occlusion --weights work_dir/ntu/cs/SFB_ECR_SkateFormer_j/checkpoint_best.pt
```

### 6.5. Đánh giá Phản thực tế (Counterfactual Concept Intervention)
Đo lường độ nhạy của action logits khi can thiệp xóa bỏ concept kích hoạt:
```bash
python main.py --config config/train/ntu_cs/SFB_ECR_SkateFormer_j.yaml --phase test_counterfactual --weights work_dir/ntu/cs/SFB_ECR_SkateFormer_j/checkpoint_best.pt
```

### 6.6. Chạy trên Google Colab
Chỉ cần mở file [`SFB_ECR_SkateFormer_Colab.ipynb`](file:///d:/Thesis/SFB-ECR-SkateFormer/SFB_ECR_SkateFormer_Colab.ipynb) trên Google Colab. Toàn bộ mã nguồn, cấu hình checkpoint tự động resume và dummy test đã được đóng gói độc lập theo từng Block tuần tự.

# Nhật ký xây KG bằng LLM

> Mỗi lần chạy (toàn bộ hoặc một phần quy trình S1–S8) thêm một mục **Lần chạy N** ở cuối file.
> Số liệu thống kê: `python tools/kg_build.py report` → `annotations/kg_work/report.md`, dán vào mục
> "Số liệu". Phần còn lại (cấu hình, chi phí, sai khác, nhận xét) ghi tay.
> Quy trình và prompt: [KG_CONSTRUCTION_LLM_PLAN.md](KG_CONSTRUCTION_LLM_PLAN.md).
> Prompt dùng thật: `annotations/kg_prompts/*.txt`; câu trả lời thô: `annotations/kg_raw/`.

---

## Lần chạy 1 — 2026-10-16 — KG `llm_only` v1.0

### Cấu hình

| Mục | Giá trị |
|---|---|
| LLM | `claude-opus-5-5`, chạy dưới dạng subagent của Claude Code (mỗi agent một ngữ cảnh độc lập) |
| Nhiệt độ | Không kiểm soát được (mặc định của subagent) |
| Prompt | `system` v1, `s1`–`s6` v1 (`annotations/kg_prompts/`) |
| Đầu vào | 120 lớp NTU120 (`annotations/kg_inputs/actions.json`; A50–A60, A106–A120 là hai người) |
| Độc lập giữa các run | Mỗi agent bị cấm đọc `kg_raw/` và `kg_work/` của agent khác |
| Kết quả | `annotations/action_kg.json`, hash `sha256:fb8c63668bb7…` |

### Các bước và chi phí

| Bước | Việc | Agent | Run | Token (agent) | Thời gian (song song) |
|---|---|---|---|---|---|
| S1 | Định nghĩa 120 node hành động (12 lô × 10) | 4 | 1 | 261 006 | ~4.4 phút |
| S2 | Đề xuất thuộc tính, 8 nhóm | 3 | 3 (run3 ẩn tên lớp) | 267 974 | ~6.2 phút |
| S3 | Gộp, lọc từ vựng | 1 | 1 | 184 591 | ~10.4 phút |
| S4 | Gán cạnh 120 × 60 | 9 (3 run × 3 phần 40 lớp) | 3 | 773 050 | ~4.4 phút |
| S5 | similar/opposite, implies/excludes | 1 | 1 | 117 308 | ~4.3 phút (song song với S4) |
| S6 | Sửa cặp trùng hồ sơ | 1 | 1 | 89 513 | ~1 phút |
| **Tổng** | | **19** | | **1 693 442** | **~27 phút** |

Script: `tools/kg_inputs.py`, `tools/kg_build.py` (s1, s2-input, s2-collect, s4-input, s4-vote, s5,
build, report), `tools/validate_kg.py`, `tools/kg_review.py`.

### Số liệu

#### S1 — node hành động

- 120 hành động, 12 lô; kiểm tra tự động (tên, id, bộ phận cơ thể, 3 câu diễn đạt lại, tên
  `confusable_with`): **0 lỗi**
- `text`: 10–15 từ (trung bình 13.1)
- Có `not_observable` khác rỗng: 106/120

#### S2 — ứng viên thuộc tính

| Run | active_part | contact | interaction | location | motion_type | posture | temporal | whole_body | Tổng |
|---|---|---|---|---|---|---|---|---|---|
| run1 | 12 | 13 | 13 | 12 | 14 | 14 | 13 | 11 | 102 |
| run2 | 11 | 10 | 13 | 11 | 14 | 13 | 11 | 11 | 94 |
| run3_anon | 12 | 12 | 12 | 12 | 15 | 14 | 11 | 12 | 100 |

#### S3 — từ vựng

- 60 thuộc tính: active_part 6, contact 6, interaction 11, location 9, motion_type 7, posture 7,
  temporal 7, whole_body 7
- Số run đề xuất: cả 3 run: 53, 2 run: 2, 1 run: 5
- `inferred`: 1 (`hand_upright`)
- 77 cụm; 57/296 ứng viên bị loại

#### S4 — cạnh has/lacks

| Run | Nhãn khác `unknown` | / hành động |
|---|---|---|
| run1 | 1226 | 10.2 |
| run2 | 1255 | 10.5 |
| run3 | 1180 | 9.8 |

- Fleiss' κ (5 nhãn, 7200 ô): **0.805**. *Lưu ý khi báo cáo*: κ bị đẩy lên vì phần lớn ô là
  `unknown` ở cả 3 run. Con số chặt hơn là dòng dưới.
- Ô có ít nhất một nhãn khác `unknown`: 1427; cả 3 run trùng nhãn: **803 (56%)**
- Sau bỏ phiếu: 1175 cạnh (agree 1.0: 803, agree 0.67: 372); 248 ô không đạt đa số → `unknown`;
  4 mâu thuẫn has/lacks → `unknown`

#### S5 — quan hệ

- `similar`: 271 cặp (≥ 2 nguồn trong S5 hai chiều + `confusable_with` của S1); 77 cặp một nguồn chờ duyệt
- `opposite`: 9 cặp — sit_down/stand_up, put_on/take_off × 6 (jacket, shoe, glasses, hat_cap,
  headphone, bag), put_object_into_bag/take_object_out_of_bag, walking_towards/walking_apart
- `implies`: 2, `excludes`: 16

#### S6 — sửa độ phân biệt

- Validator (Hamming < 2 trên hồ sơ có dấu) tìm được 1 cặp: `run_on_the_spot` / `butt_kicks`
- 0 sửa cạnh; thêm thuộc tính `knee_lifts_forward` (whole_body, direct; has ở 6 lớp khác)

#### KG cuối

- 120 hành động, 61 thuộc tính
- `has`/hành động: 2–10 (trung bình 6.2); `lacks`/hành động: 0–8 (trung bình 3.7)
- Bậc thuộc tính: 2–60; cao nhất: `single_occurrence` 60, `single_arm_active` 49, `raise_hold_lower` 30
- Cạnh bị bỏ do `excludes`: 2 (`neck_pain`: `head_carries_motion`; `support_somebody`: `distance_decreases`)
- Validator: **0 lỗi, 0 cặp trùng hồ sơ**, 5 cảnh báo (nod_head_bow, shake_head, wipe_face chỉ 2 `has`;
  nausea_vomiting, support_somebody không có `lacks`)
- Đã duyệt tay: 0/120 hành động

### Sai khác so với plan

| Plan | Thực tế | Ảnh hưởng |
|---|---|---|
| S1 chạy 3 lần | 1 lần (3 câu diễn đạt lại lấy trong cùng lần chạy) | `text` chưa được chọn giữa nhiều bản |
| S3 gộp bằng embedding (cos > 0.85) rồi LLM | Một LLM gộp toàn bộ 296 ứng viên | Không có ngưỡng embedding để báo cáo; quyết định ghi trong `kg_raw/s3/decisions.json` |
| S3 duyệt tay từ vựng trước S4 | Chưa duyệt; S4 chạy trên từ vựng LLM | Sửa từ vựng sau này phải chạy lại S4 cho thuộc tính đổi |
| S6 dùng cặp nhầm trên validation | Chưa có (cần chạy backbone); chỉ dùng cặp trùng hồ sơ | Có thể còn cặp nhầm thực tế chưa có thuộc tính phân biệt |
| S6 chạy 3 lần | 1 lần | Cạnh của `knee_lifts_forward` có `agree = null`, `source = "s6"` |
| — | Vi phạm `excludes` xử lý tự động: bỏ cạnh yếu hơn | 2 cạnh, ghi trong `dropped_edges` |

### Nhận xét của các agent (định tính, dùng cho phần thảo luận)

Các lớp mà agent S1 cho là khó định nghĩa chỉ từ skeleton:

- Khác nhau chủ yếu ở **vật thể**: drop; giving_object / exchange_things; wield_knife /
  hit_with_object / punch_slap; put_object_into_bag / take_object_out_of_bag; nhóm thao tác hai tay
  trước người (counting_money, cutting_nails, play_magic_cube, open_bottle, apply_cream_on_hand).
- Khác nhau chủ yếu ở **hình dạng ngón tay**: make_OK_sign, make_victory_sign, thumb_up,
  thumb_down, snap_fingers, rock-paper-scissors.
- Khác nhau ở **điểm tiếp xúc trên người kia**: touch_pocket / pat_on_back (S3 tách bằng
  `touches_partner_upper_body` / `touches_partner_lower_body`).
- **Không quan sát được**: yawn (miệng), reading / play_with_phone_tablet (tư thế gần như giống nhau).

Quyết định S3 đáng chú ý: bỏ `wrist_rotation` (cả 3 run đề xuất) vì khớp thumb/hand-tip nhiễu →
thumb_up/thumb_down chỉ còn dựa vào `hand_upright` (`inferred`); bỏ các thuộc tính chung chung
(`lower_body_still`, `torso_carries_motion`) và thuộc tính chứa ngưỡng số (`few_repeats`).

### Việc còn mở sau lần chạy này

- [ ] S7 duyệt tay: `annotations/kg_review/edges.csv` (1185 dòng; ưu tiên 1: 2 cạnh bị bỏ; ưu tiên 2: cạnh agree < 1)
- [ ] Duyệt 4 mâu thuẫn S4 và 77 cặp `similar` một nguồn (`annotations/kg_work/s4_edges.json`, `s5_relations.json`)
- [ ] Xem lại `single_occurrence` (bậc 60, ở biên trên)
- [ ] S6 theo cặp nhầm trên validation (sau khi có kết quả backbone)
- [ ] S8: `tools/shuffle_kg.py` (3 KG xáo trộn cho K2), text embedding (CLIP / sentence-transformer, chạy trên Colab/Kaggle)

---

<!-- Mẫu cho lần chạy tiếp theo: chép từ đây

## Lần chạy N — YYYY-MM-DD — <mô tả: bước nào, phiên bản KG>

### Cấu hình
| Mục | Giá trị |
|---|---|
| LLM | |
| Prompt | (phiên bản, thay đổi so với lần trước) |
| Phạm vi | (bước nào chạy lại, thuộc tính / lớp nào) |
| Kết quả | (file, hash) |

### Các bước và chi phí
| Bước | Việc | Agent | Run | Token | Thời gian |
|---|---|---|---|---|---|

### Số liệu
(dán annotations/kg_work/report.md)

### Thay đổi so với lần chạy trước
### Sai khác so với plan
### Nhận xét
### Việc còn mở
-->

### S1 — node hành động

- 120 hành động, 12 lô
- `text`: 10–15 từ (trung bình 13.1)
- Có `not_observable` khác rỗng: 106/120

### S2 — ứng viên thuộc tính

| Run | active_part | contact | interaction | location | motion_type | posture | temporal | whole_body | Tổng |
|---|---|---|---|---|---|---|---|---|---|
| run1 | 12 | 13 | 13 | 12 | 14 | 14 | 13 | 11 | 102 |
| run2 | 11 | 10 | 13 | 11 | 14 | 13 | 11 | 11 | 94 |
| run3_anon | 12 | 12 | 12 | 12 | 15 | 14 | 11 | 12 | 100 |

### S3 — từ vựng

- 60 thuộc tính: active_part 6, contact 6, interaction 11, location 9, motion_type 7, posture 7, temporal 7, whole_body 7
- Số run đề xuất: 3 run: 53, 2 run: 2, 1 run: 5
- `inferred`: 1
- 77 cụm; 57 ứng viên bị loại

### S4 — cạnh has/lacks

| Run | Nhãn khác `unknown` | / hành động |
|---|---|---|
| run1 | 1226 | 10.2 |
| run2 | 1255 | 10.5 |
| run3 | 1180 | 9.8 |

- Fleiss' κ (5 nhãn, 7200 ô): **0.805**
- Ô có ít nhất một nhãn khác `unknown`: 1427; cả 3 run trùng nhãn: 803 (56%)
- Sau bỏ phiếu: 1175 cạnh (agree 1.0: 803, agree 0.67: 372); 4 mâu thuẫn has/lacks → `unknown`

### S5 — quan hệ

- `similar`: 271 cặp (≥ 2 nguồn); 77 cặp một nguồn chờ duyệt
- `opposite`: 9 cặp
- `implies`: 2, `excludes`: 16

### S6 — sửa độ phân biệt

- 1 cặp xử lý; 0 sửa cạnh; thuộc tính mới: knee_lifts_forward

### KG cuối

- 120 hành động, 61 thuộc tính, hash `sha256:fb8c63668bb7…`
- `has`/hành động: 2–10 (trung bình 6.2); `lacks`/hành động: 0–8 (trung bình 3.7)
- Bậc thuộc tính: 2–60; cao nhất: `single_occurrence` 60, `single_arm_active` 49, `raise_hold_lower` 30
- Cạnh bị bỏ do `excludes`: 2
- Đã duyệt tay: 0/120 hành động

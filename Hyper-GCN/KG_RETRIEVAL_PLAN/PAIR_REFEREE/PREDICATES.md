# Pair Referee: danh sách vị từ hình học

> Thiết kế tổng thể (sơ đồ): https://claude.ai/artifact/9rwB1w7Ug92hR8avmHG9WY
> Tiến độ: [CONCEPT_GRAPH_PROGRESS.md](../CONCEPT_GRAPH/CONCEPT_GRAPH_PROGRESS.md)
> Trạng thái 2026-10-27: **đặc tả, chưa có code**. Theo RULE.md, code chỉ viết sau khi người dùng duyệt file này.

## 0. Nguyên tắc

- Vị từ là **hàm cố định của toạ độ khớp**. Không đọc nhãn, không đọc feature của Hyper-GCN, không học.
- Vị từ ở đây là **chuỗi theo frame** `e_k(t)`, không phải một số cho cả clip như A1-G. Nhờ vậy dùng được `p@s` và
  `THEN`. Có 3 vị từ mức clip (§2.H) cho những thứ không có nghĩa theo frame.
- Vị từ chung cho mọi cặp. Cặp nào dùng vị từ nào là việc của bước học luật (AUC + L1), không viết tay.
- **Bất biến tay thuận:** dùng tay *active* `a` (tay có quãng đường lớn nhất so với spine shoulder trong clip) và
  tay *other* `o`, không dùng trái / phải. NTU có người thuận tay trái.
- Mọi khoảng cách và độ cao chia cho `L` (chiều dài thân trung bình), nên không phụ thuộc chiều cao người hay
  khoảng cách tới camera.

## 1. Quy ước đầu vào (giữ từ A1-G)

Tái dùng hạ tầng của `model/predicates.py` cũ (đã xoá; lấy lại bằng `git show 8cd3c4c^:./model/predicates.py`):
`absolute`, `valid_actors`, `Frame`.

- Đầu vào: output của feeder ở joint mode, `x [N, 3, T, 25, M=2]`. Mọi khớp tương đối với khớp 20 (spine shoulder),
  riêng khớp 20 giữ quỹ đạo tuyệt đối. `absolute(x)` → `[N, M, T, V, 3]`.
- Hệ trục theo thân, tính cho từng mẫu và từng người (bất biến với `random_rot` ±0.3 rad):
  `up` = hướng spine_base → spine_shoulder trung bình 20% frame đầu; `right` = vai phải − vai trái, trực giao hoá
  với `up`; `forward` = `up × right`; `L` = chiều dài thân trung bình.
  Ký hiệu: `h(v)` = chiếu lên `up` / L, `lat(v)` = chiếu lên `right` / L, `fwd(v)` = chiếu lên `forward` / L,
  `d(a, b)` = ‖a − b‖ / L.
- **Một view tất định cho mỗi mẫu** (transform của test), cho cả train và test.

Khớp NTU (0-based):

| Tên | Chỉ số | Tên | Chỉ số |
|---|---|---|---|
| spine_base `SB` | 0 | spine_mid `SM` | 1 |
| neck `NK` | 2 | head `HD` | 3 |
| shoulder L / R | 4 / 8 | elbow L / R | 5 / 9 |
| wrist L / R | 6 / 10 | hand L / R | 7 / 11 |
| hip L / R | 12 / 16 | knee L / R | 13 / 17 |
| ankle L / R | 14 / 18 | foot L / R | 15 / 19 |
| spine_shoulder `SS` | 20 | hand_tip L / R | 21 / 23 |
| thumb L / R | 22 / 24 | | |

Điểm dẫn xuất:
- `mouth = 0.6·HD + 0.4·NK` (skeleton không có miệng; ước lượng giữa đầu và cổ)
- `chest = (SM + SS) / 2`
- `ear± = HD ± 0.25·L·right` (hai bên đầu)
- `feet_floor(t) = min(h(foot_L), h(foot_R))`

Tay `a` / `o`: `H_a, W_a, T_a, U_a, E_a, S_a` lần lượt là hand, wrist, hand_tip, thumb, elbow, shoulder của tay active.
Tương tự với `o`.

## 2. Vị từ theo frame (K = 39)

Cột "Cặp mục tiêu" chỉ để kiểm tra độ phủ; không ràng buộc luật.
Dấu quy ước: **giá trị lớn = thuộc tính có mặt nhiều**.

### A. Vị trí tay (11)

| # | Tên | Công thức `e(t)` | Cặp mục tiêu |
|---|---|---|---|
| A1 | `near_mouth_a` | `−d(H_a, mouth)` | drink / yawn, hush / blow_nose, eat |
| A2 | `near_mouth_o` | `−d(H_o, mouth)` | yawn (che miệng), blow_nose |
| A3 | `both_at_face` | `−max(d(H_a, mouth), d(H_o, mouth))` | blow_nose / hush, wipe_face |
| A4 | `near_ear_a` | `−min(d(H_a, ear+), d(H_a, ear−))` | phone_call, put_on / take_off headphone |
| A5 | `above_head_a` | `h(H_a − HD)` | put_on / take_off hat, headphone |
| A6 | `near_chest_a` | `−d(H_a, chest)` | put_on / take_off jacket, into / out bag |
| A7 | `hand_height_a` | `h(H_a − SB)` | into / out bag (dùng với THEN, F2) |
| A8 | `hand_forward_a` | `fwd(H_a − SS)` | giving / exchange, point, reach |
| A9 | `hands_close` | `−d(H_a, H_o)` | clap, rub_hands, reading (cầm sách hai tay) |
| A10 | `hand_near_foot` | `−min_{h∈{a,o}, f∈{L,R}} d(H_h, foot_f)` | put_on / take_off shoe |
| A11 | `hand_behind` | `−fwd(H_a − SM)` | jacket (tay ra sau), scratch_back |

### B. Hình bàn tay (4)

Khớp tip / thumb của Kinect nhiễu. Đây là nguồn duy nhất cho các cặp cử chỉ ngón tay, nên vẫn giữ và báo cáo
riêng độ tin cậy (AUC train).

| # | Tên | Công thức | Cặp mục tiêu |
|---|---|---|---|
| B1 | `thumb_up_a` | `h(U_a − H_a)` | **thumb_up / thumb_down** (đổi dấu) |
| B2 | `tip_extended_a` | `d(T_a, W_a)` | make_victory / OK_sign, fist vs mở tay |
| B3 | `thumb_tip_gap_a` | `d(U_a, T_a)` | OK_sign (khép), victory, snap_fingers |
| B4 | `tip_up_a` | `h(T_a − H_a)` | hush (ngón dựng), point |

### C. Cánh tay (3)

| # | Tên | Công thức | Cặp mục tiêu |
|---|---|---|---|
| C1 | `elbow_straight_a` | `cos∠(S_a, E_a, W_a)` đảo dấu: `−cos` (thẳng = lớn) | reach / point vs gần thân |
| C2 | `elbow_raised_a` | `h(E_a − S_a)` | hat, headphone, comb_hair |
| C3 | `elbow_raised_o` | `h(E_o − S_o)` | jacket (hai tay), bag đeo vai |

### D. Đầu và thân (5)

| # | Tên | Công thức | Cặp mục tiêu |
|---|---|---|---|
| D1 | `head_back` | `−fwd(HD − NK)` | **drink / yawn** (ngửa đầu khi uống) |
| D2 | `head_down` | `fwd(HD − NK)` | reading / writing, play_phone |
| D3 | `head_side` | `abs(lat(HD − NK))` | headphone, phone_call |
| D4 | `torso_bend` | `fwd(SS − SB)` | shoe, pick_up, into / out bag |
| D5 | `torso_twist` | `abs(góc giữa (vai R − vai L) và (hông R − hông L) trên mặt phẳng ⊥ up) / π` | jacket, throw |

### E. Toàn thân (3)

| # | Tên | Công thức | Cặp mục tiêu |
|---|---|---|---|
| E1 | `pelvis_low` | `−h(SB) + feet_floor` | sit_down / stand_up, shoe (ngồi) |
| E2 | `knee_bent` | trung bình hai gối của `cos∠(hip, knee, ankle) + 1` / 2 (gập = lớn) | shoe, squat |
| E3 | `foot_raised` | `abs(h(foot_L − foot_R))` | **put_on / take_off shoe** (nhấc chân), kick |

### F. Động học (7)

Vận tốc lấy sai phân trung tâm trên chuỗi đã làm mượt (trung bình trượt 5 frame), đơn vị L / frame.

| # | Tên | Công thức | Cặp mục tiêu |
|---|---|---|---|
| F1 | `speed_a` | `‖v(H_a − SS)‖` | throw, punch vs chậm |
| F2 | `rising_a` | `h(v(H_a − SB))` (có dấu: > 0 đi lên) | **into / out bag**, put_on / take_off |
| F3 | `toward_body_a` | `−d/dt d(H_a, chest)` (> 0 tiến về thân) | into / out bag, pocket |
| F4 | `speed_o` | `‖v(H_o − SS)‖` | một tay vs hai tay |
| F5 | `head_speed` | `‖v(HD − SS)‖` | blow_nose (đầu giật), nod, yawn |
| F6 | `hands_closing` | `−d/dt d(H_a, H_o)` | clap, rub_hands |
| F7 | `fine_motion_a` | `‖v(T_a − W_a)‖ / (‖v(W_a − SS)‖ + 0.01)` (ngón động hơn cổ tay) | **reading / writing**, snap_fingers, typing |

### G. Hai người (6)

Chỉ hợp lệ khi có 2 người thật (`valid_actors`). Không hợp lệ → giá trị 0.5 sau chuẩn hoá (không có bằng chứng) và
cờ `valid = False`. Người 0 / 1 trong NTU là tuỳ ý, nên mọi vị từ đều **đối xứng** (min / max theo hai người).

| # | Tên | Công thức | Cặp mục tiêu |
|---|---|---|---|
| G1 | `body_dist` | `−d(SB⁰, SB¹)` (L trung bình hai người) | push / hug, approach / leave |
| G2 | `hand_to_other_torso` | `−min_{p, h} d(H_h^p, chest^{1−p})` | punch / push / pat_back |
| G3 | `hands_meet` | `−min d(H^0_h, H^1_h')` trên 4 cặp tay | **giving / exchange**, handshake |
| G4 | `approach` | `−d/dt d(SB⁰, SB¹)` | walk_toward / walk_apart |
| G5 | `both_reach` | `min_p fwd^p(H_a^p − SS^p)` (cả hai cùng đưa tay ra) | **giving / exchange** (exchange: cả hai) |
| G6 | `active_balance` | `min_p speed_a^p / max_p speed_a^p` | giving (một người chủ động) / exchange |

## 3. Vị từ mức clip (3) và số người

| Tên | Công thức | Dùng làm |
|---|---|---|
| `repetitions_a` | số chu kỳ qua lại của `principal_signal(H_a − SS)` (hàm `repetitions` cũ) | literal không có `@s` (writing, clap, wave) |
| `long_hold_a` | tỉ lệ frame `speed_a < 0.25·max` | literal không có `@s` (reading, phone) |
| `legs_vs_arms` | năng lượng chân / (chân + tay) (hàm `_energy` cũ) | literal không có `@s` |
| `n_person` | `valid_actors(x).sum(1)` | **mặt nạ cứng**, không phải literal |

## 4. Từ chuỗi frame tới literal

1. **Chuẩn hoá theo vị từ, không theo mẫu:** với mỗi `k`, lấy phân vị 5% và 95% của `e_k(t)` trên toàn bộ train
   (mọi frame, người thật), rồi `ê = clip((e − q5) / (q95 − q5), 0, 1)`. Vị từ có dấu (F2, F3, F6, G4) dùng
   `q = max(|q5|, |q95|)` và `ê = clip(0.5 + e / (2q), 0, 1)`, để 0.5 = đứng yên.
   (A1-G dùng min-max theo mẫu cho sự kiện THEN, làm mất mức tuyệt đối; ở đây chỉ dùng min-max theo mẫu bên trong
   toán tử THEN, §4.3.)
2. **Gộp theo đoạn:** chia `T` frame hợp lệ thành `S = 8` đoạn bằng nhau, `P[k, s] = mean_{t∈s} ê_k(t)`.
   Một người: lấy theo người có `n_person`; hai người cho vị từ A–F: max theo người thật (giống A1-G).
3. **Sinh literal ứng viên:**
   - `p@s`, `¬p@s = 1 − p@s`: 39 × 8 × 2 = 624
   - `DUR(p) = mean_s P[k, s]`: 39
   - `THEN(p, q)` = công thức A1-G trên chuỗi frame đã min-max theo mẫu, chỉ cho 10 vị từ sự kiện:
     `near_mouth_a, near_ear_a, above_head_a, near_chest_a, hand_height_a (thấp = 1 − ê), hands_close,
     hand_near_foot, hand_behind, torso_bend, pelvis_low` → 10 × 9 = 90
   - mức clip: 3 × 2 = 6
   - **Tổng ≈ 759 literal**, cache một lần dạng `float16 [N, 759]` (~77 MB cho train NTU120 X-Sub).

## 5. Độ phủ các cặp đã biết

Cặp lấy từ ghi chép A1 / CONCEPT_GRAPH (top-25 cặp nhầm của K0 chiếm 1 944 / 6 299 lỗi). Danh sách chính thức sẽ
lấy lại trên val tách subject (Bước 1).

| Cặp | Vị từ chính | Skeleton thấy được? |
|---|---|---|
| put_object_into_bag / take_object_out_of_bag | F2, F3, A7, THEN(near_chest, hand_low) | Có thể; lần trước ngược dấu, để dữ liệu chọn dấu |
| put_on / take_off a_shoe | E3, A10, D4, E1 | Một phần (thứ tự kéo / đẩy khó thấy) |
| put_on / take_off headphone, hat | A4, A5, C2, THEN | Có (THEN headphone AUC 0.89 ở A1-G) |
| thumb_up / thumb_down | B1 | Có nếu khớp thumb đủ sạch |
| make_OK_sign / make_victory_sign | B2, B3 | Yếu (khớp ngón nhiễu) |
| make_victory_sign / snap_fingers | B3, F7 | Yếu |
| drink_water / yawn | D1, A1, A2, F5 | Có |
| yawn / blow_nose, hush / blow_nose | A3, F5, B4 | Có một phần |
| reading / writing, writing / play_with_phone | F7, D2, `repetitions_a`, `long_hold_a` | Yếu (khác nhau ở vật) |
| reading / staple_book | A9, F1 (dập) | Yếu |
| giving_object / exchange_things | G3, G5, G6 | Có |

Các cặp "Yếu" là rủi ro chính đã nêu trong thiết kế. Bước 0 đo trần trên chúng trước.

## 6. Kiểm tra trước khi dùng (chỉ trên train)

| Kiểm tra | Ngưỡng |
|---|---|
| Hữu hạn, không NaN, mọi mẫu | 100% |
| Bất biến xoay: cùng mẫu, xoay ±0.3 rad quanh trục dọc → `|ΔP|` trung bình | < 0.02 |
| Test tổng hợp (khung xương giả: tay chạm miệng, ngón cái lên / xuống, ngồi, hai người lại gần) | đúng dấu |
| AUC theo cặp tốt nhất trên các cặp §5 | báo cáo, không lọc toàn cục (A1-G lọc AUC 0.75 toàn cục đã bỏ nhầm vị từ cho cặp khó) |

## 7. File sẽ viết (sau khi duyệt)

| File | Nội dung |
|---|---|
| `model/pair_predicates.py` | numpy: lấy lại `absolute`, `valid_actors`, `Frame` từ bản cũ + 39 vị từ theo frame, 3 vị từ clip, chuẩn hoá, gộp đoạn, sinh literal |
| `tools/cache_pair_predicates.py` | chạy qua feeder (view test), lưu `stats.npz` (q5 / q95 từ train), `literals_{train,test}.npy`, `n_person`, `valid` |
| `tests/test_pair_predicates.py` | test tổng hợp §6 |
| `KG_RETRIEVAL_PLAN/PAIR_REFEREE/RUN.md` | các bước chạy trên Colab từ đầu |

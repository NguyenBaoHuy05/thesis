# Concept–Action Logic Graph: tiến độ

> Plan: [CONCEPT_GRAPH_PLAN.md](CONCEPT_GRAPH_PLAN.md). Sơ đồ: [drawio/architecture_concept_graph.drawio](../../drawio/architecture_concept_graph.drawio).
> Số liệu chép từ log chạy thật; ô chưa có số ghi *chờ*. Dev: **1 seed (0)**.
> Cập nhật lần cuối: 2026-10-19

## Quyết định đã chốt (2026-10-19)

- Nhóm ∨ ở v1: chỉ posture (hip / leg static stand ∨ sit).
- Ngưỡng đạt của P1 như plan §7.
- Cache S = 4 (khoảng 13 GB) được phép, dùng ở E2-S4.
- Baseline REASON: E1b = port lớp logic của REASON lên decoder và cache của mình. REASON đầy đủ để ở P4.
- E1b dùng cùng lr với decoder (1e-3); REASON gốc dùng 1e-4 cho logic trong lịch 110 epoch end-to-end.
- P1: trọng số cạnh α cố định bằng 1 (`learn_alpha: False`), để E1c chỉ khác CL v2 đúng ở ¬(j|i).
  α học được để từ P2.

## P0: code

| File | Trạng thái |
|---|---|
| `model/concept_graph.py`, `tools/build_concept_graph.py` | xong; `tests/test_concept_graph.py` 7/7 qua (local, không cần torch) |
| `model/fuzzy_executor.py`, `model/concept_graph_head.py` | xong; test trong `tests/test_fuzzy_executor.py` (cần torch, *chờ* Colab) |
| `model/reason_logic.py` (port REASON, MIT) | xong; test *chờ* Colab |
| `model/pair_metrics.py` (PCR, per-pair, K5-pair) | xong; test *chờ* Colab |
| `train_concept_logic.py`: `head: cl \| graph \| reason`, chỉ số cặp cho mọi head và cho K0 | xong |
| `config/concept_graph/e1{b,c,d,e}.yaml` | xong (E1a = `config/concept/cl_v2.yaml`) |
| R-GCN, nhánh action (`model/rgcn.py`, `return_tokens`) | để P3 |

Đồ thị v1 trên v2.4 (`python tools/build_concept_graph.py`):

| | Số lượng |
|---|---|
| Motif ∧ | 25 (114 / 120 lớp có ít nhất 1 motif; 476 ô của M được phủ bằng motif) |
| Cổng ∨ | 2 (posture hip, leg) |
| Cặp dễ nhầm | 58; node ¬(j\|i): 113 (3 hướng có M_j∖M_i rỗng bị bỏ); 66 lớp có ít nhất 1 ¬ |
| Cạnh | r1 56, r2 4, r3 563, r4 226, r5 186, r6 / r7 113, r8 216 |

Ví dụ: ¬(take_off_a_shoe | put_on_a_shoe) = 1 − ∧(motion_upward, seq_hands_first, lower_then_raise).

## Lệnh Colab

Cache S = 1 đã có trên Drive (`kg_cache/ntu120_xsub_joint_*`). Chép về ổ local trước (memmap trên Drive chậm).

```
%cd /content/Hyper-GCN
!git pull
!pip -q install pytest
!python -m pytest tests -q
!python tools/build_concept_graph.py --out /content/graph_v1.json | head -40

!mkdir -p /content/jcache && cp /content/drive/MyDrive/BaoHuy/kg_cache/ntu120_xsub_joint_* /content/jcache/
%env J=/content/jcache
%env W=/content/drive/MyDrive/BaoHuy/work_dir/concept_graph
```

E1b lần 2 (sau khi sửa công thức train; E1a, c, d, e đã có số):

```
!python train_concept_logic.py --config config/concept_graph/e1b.yaml --cache-dir $J --work-dir $W/e1b
```

P1 (seed 0; mỗi lệnh vài phút, chạy lần lượt):

```
!python train_concept_logic.py --config config/concept/cl_v2.yaml        --cache-dir $J --work-dir $W/e1a
!python train_concept_logic.py --config config/concept_graph/e1b.yaml    --cache-dir $J --work-dir $W/e1b
!python train_concept_logic.py --config config/concept_graph/e1c.yaml    --cache-dir $J --work-dir $W/e1c
!python train_concept_logic.py --config config/concept_graph/e1d.yaml    --cache-dir $J --work-dir $W/e1d
!python train_concept_logic.py --config config/concept_graph/e1e.yaml    --cache-dir $J --work-dir $W/e1e
```

`head.pt` được lưu ngay sau khi train. Nếu bước đánh giá lỗi, chạy lại đúng lệnh đó với `--eval-only` (không train lại).
2026-10-19: E1b lần 1 train bị sụp (train top1 4.45%: dùng một lr 1e-3, không clip). Đã sửa theo công thức
REASON: lr logic 1e-4, clip norm gradient 1.0 cho decoder / 0.5 cho logic.
2026-10-19: E1b bị OOM ở K5 (logits REASON trên cả tập test một lần, khoảng 12.4 GiB). Đã sửa: tính theo batch.

Mỗi lần chạy in ra Top1 / Top5, PCR (head và K0), Top1 trong / ngoài cặp, 8 cặp nhầm nhiều nhất, K5, K5-pair.
Đầy đủ ở `$W/<e>/kg_metrics.json` (`pairs.per_pair` có 58 dòng).

## P1: kết quả

Quy tắc đạt (chốt trước): `PCR(tốt nhất của E1c, E1d) ≤ 0.8 × min(PCR E1a, PCR E1b)` **và** Top1 ≥ Top1(E1a) − 0.3
**và** PCR < PCR(E1e). Chênh trong 10% quanh ngưỡng thì chạy thêm seed 1, 2.

Dự đoán ghi trước: E1c giảm PCR 15–30% so với E1a, Top1 ±0.5. E1b có Top1 cao hơn E1a nhưng PCR tương đương.

| | Top1 / Top5 | PCR | Top1 trong / ngoài cặp | sửa / phá vs K0 | K5 hỗ trợ / ngẫu nhiên | K5-pair | train top1 cuối |
|---|---|---|---|---|---|---|---|
| K0 | 87.63 / 97.56 | **3.58** | 85.54 / 90.23 | | | | |
| E1a (CL v2) | 85.60 / **94.98** | 4.18 | 83.59 / 88.12 | 804 / 1836 | **14.49** / 0.00 | **67.30** | 99.80 |
| E1b (logic REASON), lần 1 | sụp | | | | | | 4.45 |
| E1b, lần 2 (công thức REASON + init mặc định) | 83.80 / 91.20 | 6.55 | 81.09 / 87.19 | 600 / 2548 | 20.41 / 0.02 | 43.96 | 99.00 |
| E1c (+¬, ω = 1) | **86.05** / 93.79 | **4.17** | **84.43** / 88.09 | 740 / 1542 | 7.69 / 0.00 | 38.20 | 99.94 |
| E1d (+¬, ω = 0) | 85.83 / 93.59 | **4.17** | 84.34 / 87.69 | 693 / 1610 | 10.19 / 0.00 | 43.19 | 99.91 |
| E1e (¬ ngẫu nhiên) | 86.01 / 94.06 | 4.19 | 84.15 / **88.33** | 699 / 1525 | 8.85 / 0.00 | 59.76 | 99.94 |

PCR chỉ tính trên 28 277 mẫu test có nhãn thuộc ít nhất một cặp.
Cặp nhầm nhiều nhất, giống nhau ở mọi head: make_OK_sign / make_victory_sign khoảng 27%,
put_on_a_shoe / take_off_a_shoe 11–13%, thumb_up / make_victory_sign khoảng 11%,
writing / play_with_phone_tablet khoảng 6%, giving_object / exchange_things khoảng 5–6%.

### Kết luận P1 (2026-10-19, seed 0): **C1 không đạt**

- Ngưỡng: PCR ≤ 0.8 × min(E1a, E1b) ≤ 0.8 × 4.18 = **3.34**. Tốt nhất của E1c / E1d là **4.17**: chỉ giảm 0.01 điểm.
  Ngưỡng không phụ thuộc E1b, vì E1b chỉ có thể làm ngưỡng thấp hơn.
- Cách ngưỡng rất xa (25%, không phải trong vùng 10%), nên **không chạy thêm seed**.
- Đối chứng E1e (¬ trên cặp ngẫu nhiên) có PCR 4.19, gần như bằng E1c. Top1 cũng tăng như E1c (+0.41 so với +0.45).
  Phần Top1 tăng thêm vì vậy không đến từ cặp đúng.
- **Mọi head trên concept đều nhầm cặp nhiều hơn K0** (4.17–4.19 so với 3.58). Top1 trong cặp cũng thấp hơn K0
  khoảng 1–2 điểm. Nút thắt concept làm mất bớt thông tin phân biệt cặp mà backbone có.
- Giải thích: AUC concept là 0.99 ở mọi head. Concept vẫn chỉ là mã của lớp (ECOC), như A1 / A1-G.
  ¬(j|i) là một hàm của cùng vector c, nên khi decoder sai lớp thì các concept phân biệt cũng sai theo (lỗi tương quan).
  Đổi trọng số các literal (ω, ν, cặp đúng hay ngẫu nhiên) không thêm thông tin.
- Ngoài lề: CL v2 hơn CL v1 1.5 điểm Top1 (85.60 so với 84.09). K5 tăng từ 6.6% lên 14.5% nhưng chưa tới 20% như dự đoán.
  K5-pair của E1a là 67%: bật các concept phân biệt thì dự đoán chuyển sang đối tác. Logic phản ứng đúng với concept,
  nhưng concept thì không đúng ở các mẫu khó.

- E1b (logic REASON trên cùng decoder, cùng cache): **thấp hơn CL v2 1.8 điểm Top1 và nhầm cặp nhiều hơn**
  (PCR 6.55 so với 4.18). Cặp drop / thumb_down nhầm tới 57.8%: concept bị nhị phân hoá trước khi vào logic, nên mất
  mức độ của concept, và các cặp chỉ khác nhau 1–2 concept dễ bị gộp. AUC concept 0.974, thấp hơn CL (0.987–0.989).
  K5 20.4% là cao nhất (logic rời rạc nhạy với từng concept). τ dừng ở 0.64.
  Kết luận phụ: AND mềm theo ma trận (CL v2) hơn logic học được kiểu REASON trên cùng concept, cả Top1 lẫn PCR.

**Theo quy tắc của plan: dừng, không làm P2–P3** cho tới khi có hướng đem lại thông tin mới ở các cặp.
E1b vẫn chạy lại để có baseline REASON đầy đủ.

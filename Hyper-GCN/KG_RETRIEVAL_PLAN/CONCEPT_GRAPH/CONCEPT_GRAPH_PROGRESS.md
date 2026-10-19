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

P1 (seed 0; mỗi lệnh vài phút, chạy lần lượt):

```
!python train_concept_logic.py --config config/concept/cl_v2.yaml        --cache-dir $J --work-dir $W/e1a
!python train_concept_logic.py --config config/concept_graph/e1b.yaml    --cache-dir $J --work-dir $W/e1b
!python train_concept_logic.py --config config/concept_graph/e1c.yaml    --cache-dir $J --work-dir $W/e1c
!python train_concept_logic.py --config config/concept_graph/e1d.yaml    --cache-dir $J --work-dir $W/e1d
!python train_concept_logic.py --config config/concept_graph/e1e.yaml    --cache-dir $J --work-dir $W/e1e
```

Mỗi lần chạy in ra Top1 / Top5, PCR (head và K0), Top1 trong / ngoài cặp, 8 cặp nhầm nhiều nhất, K5, K5-pair.
Đầy đủ ở `$W/<e>/kg_metrics.json` (`pairs.per_pair` có 58 dòng).

## P1: kết quả

Quy tắc đạt (chốt trước): `PCR(tốt nhất của E1c, E1d) ≤ 0.8 × min(PCR E1a, PCR E1b)` **và** Top1 ≥ Top1(E1a) − 0.3
**và** PCR < PCR(E1e). Chênh trong 10% quanh ngưỡng thì chạy thêm seed 1, 2.

Dự đoán ghi trước: E1c giảm PCR 15–30% so với E1a, Top1 ±0.5. E1b có Top1 cao hơn E1a nhưng PCR tương đương.

| | Top1 / Top5 | PCR | Top1 trong / ngoài cặp | sửa / phá vs K0 | K5 hỗ trợ / ngẫu nhiên | K5-pair | train top1 cuối |
|---|---|---|---|---|---|---|---|
| K0 | 87.63 / 97.56 | *chờ* | *chờ* | | | | |
| E1a (CL v2) | *chờ* | *chờ* | *chờ* | *chờ* | *chờ* | *chờ* | *chờ* |
| E1b (logic REASON) | *chờ* | *chờ* | *chờ* | *chờ* | *chờ* | *chờ* | *chờ* |
| E1c (+¬, ω = 1) | *chờ* | *chờ* | *chờ* | *chờ* | *chờ* | *chờ* | *chờ* |
| E1d (+¬, ω = 0) | *chờ* | *chờ* | *chờ* | *chờ* | *chờ* | *chờ* | *chờ* |
| E1e (¬ ngẫu nhiên) | *chờ* | *chờ* | *chờ* | *chờ* | *chờ* | *chờ* | *chờ* |

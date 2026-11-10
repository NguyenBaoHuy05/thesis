# Ghi chú 2026-11-10: chuyển NeSy sang baseline Flora

Phiên làm việc bắt đầu tối 2026-11-09 và kết thúc 2026-11-10. Commit: `886cacb` (đã push lên `origin/main`).

## 1. Quyết định

- **Đóng góp của luận văn là NeSy, không phải TTA.** (Ghi chú cũ trong `paper_notes.md` nghiêng về Skeleton-Cache TTA, nay không phải hướng chính.)
- **Baseline đổi sang Flora** (CVPR 2026 Findings, `zeroshot/Flora/`), không dùng Neuron nữa cho phần NeSy. Lý do của bạn: ground primitive trên cube của Neuron không ổn định.
- **"Symbolic" không bắt buộc là luật.** Dùng từ vựng primitive và cấu trúc KG theo kiểu mềm, không khớp luật `require`/`forbid` cứng.
- Hướng chọn: **kiểu 1 (compose)** và **kiểu 2 (neighbors)**, cả hai chỉ đụng phía ngữ nghĩa của Flora.

## 2. Vì sao Flora, và giới hạn của nó

ZSL (%) theo bài báo (nguồn: bảng trong `paper_notes.md`, Flora lấy từ `flora_repro.ipynb`):

| | NTU60 55/5 | 48/12 | NTU120 110/10 | 96/24 |
|---|---|---|---|---|
| Neuron | 86.9 | 62.7 | 71.5 | 57.1 |
| Flora | 85.8 | 61.5 | 80.7 | 64.1 |

- Flora hơn rõ ở NTU120, kém Neuron nhẹ ở NTU60. Repo Flora phủ nhiều benchmark: SynSE, STAR, SA-DAVE (3 split ngẫu nhiên), STAR-SMIE, PURLS (40/20, 30/30, 80/40, 60/60), PKU-51, low-shot.
- Repo trên máy **không có file trọng số**. "Nhiều file có sẵn" nằm trên Google Drive theo link README: (a) trọng số Shift-GCN tiền huấn luyện, (b) feature `.npz` trích sẵn cho nhiều benchmark. Không có checkpoint của đầu Flora, nên đầu Flora vẫn phải train (nhẹ vì feature có sẵn).
- Cách Flora quyết định ([models/fm.py:290-317](Flora/models/fm.py#L290-L317)): với mỗi lớp, tính MSE giữa vận tốc dự đoán và vận tốc đích của flow, rồi `argmin`.
- **Flora gộp encoder thành một vector 256 chiều**, không có trục thời gian, khớp, hay người thứ hai. Vì vậy không ground primitive (kiểu `hands.near`, `inter.mutual_reach`) được trên feature của Flora. Đây là lý do NeSy trên Flora đặt ở phía ngữ nghĩa.

## 3. Thiết kế NeSy-KG trên Flora

File: [Flora/kg_semantics.py](Flora/kg_semantics.py). Sửa trong `trainer.load_prototypes` và `options.py`.

- Lớp i của Flora (0-based) ứng với luật id `i+1` (A-number) trong `kg_rules_v2.json`.
- Ma trận lớp × primitive `W`: +1 nếu `require`, −0.5 nếu `forbid` (`--kg-forbid-weight`), 0 nếu không nhắc.
- **`neighbors`**: độ giống chọn láng giềng = `(1-α)·CLIP + α·KG`. KG = cosine giữa hai hàng của `W`. Mặc định α = 0.5.
- **`compose`**: embedding lớp = tổng có trọng số của CLIP embedding các mô tả primitive. Chỉ dùng primitive `require` (mô tả điều bị cấm không mô tả hành động), chuẩn hoá tổng = 1. Cộng `β · sem_primitive` vào `attuned_semantics`, mặc định β = 0.3. Embedding được lặp lại để đủ 1536 chiều (Flora ghép hai loại text, primitive chỉ có một).
- **`both`**: dùng cả hai. **`none`**: Flora gốc, giữ nguyên hành vi upstream.
- Cờ: `--kg-mode none|neighbors|compose|both`, `--kg-alpha`, `--kg-beta`, `--kg-forbid-weight`, `--kg-prims`, `--kg-rules`.
- Ghép hai chuỗi token CLIP khác nhau theo vị trí là xấp xỉ thô, nhưng cùng kiểu với cách Flora trộn láng giềng ([trainer.py:60-73](Flora/trainer.py#L60-L73)).
- KG chỉ được viết từ tên lớp và từ vựng primitive, không dùng dữ liệu skeleton, nên dùng cho lớp unseen cùng loại tri thức với văn bản tên lớp mà Flora đã dùng.

### Kiểm tra offline trên NTU60 (`python kg_semantics.py 60`)

- 82 primitive, trung bình 2.9 primitive `require` và 2.7 `forbid` mỗi lớp. KG khá thưa.
- Độ giống KG ngoài đường chéo: mean 0.07, p90 0.33.
- Một số láng giềng hợp lý: punching, pushing và point finger gần nhau; drink water gần eat meal.
- **4 cặp lớp KG không phân biệt được** (hàng giống hệt, độ giống = 1): wear/take off jacket, shoe, glasses, hat. Ở chế độ `neighbors` chúng thành láng giềng mạnh của nhau và có thể làm nhoè hai lớp. Cần theo dõi khi ablate; cách sửa là thêm primitive chiều (mặc lên, cởi ra).

## 4. Việc đã làm trong repo (commit `886cacb`)

- Thêm `Flora/kg_semantics.py`, `Flora/kg/` (khôi phục từ lịch sử git commit `51002aa`), sửa `trainer.py` và `options.py`.
- Thêm `Flora/semantics/ntu60_llm.txt` và `ntu120_llm.txt`: **upstream Flora thiếu hai file này**, nhưng `MappingNet.extract_text_embeddings` đọc chúng ([fm.py:164](Flora/models/fm.py#L164)). Bạn tải về (60 và 120 dòng, khớp số dòng csv). `.gitignore` gốc có `*.txt` nên phải `git add -f`.
- Thêm notebook [flora_nesy.ipynb](flora_nesy.ipynb) (xem mục 5).
- Xoá `neuron_extract.ipynb`, `neuron_plus.ipynb`, và gitlink mồ côi `skeletoncontext` (bạn xác nhận chủ ý xoá). Lưu ý: `neuron_plus.ipynb` lúc đầu có sửa chưa commit, phần đó coi như mất.
- Nhánh KG/NeSy của Neuron (`main_nesy.py`, `primitive_labels.py`, `kg_tools/`, các file `primitive*.json`, `kg_rules*.json`) đã bị gỡ khỏi working tree ở commit `e825143` nhưng còn trong lịch sử (ví dụ `51002aa`).

## 5. Cách chạy: `flora_nesy.ipynb`

Notebook clone `NguyenBaoHuy05/thesis` (repo private thì lưu token vào Colab Secrets tên `GH_TOKEN`) và dùng thẳng `zeroshot/Flora`.

1. Mount Drive, clone hoặc pull, `assert` có đủ file KG và `*_llm.txt`, có `--kg-mode`.
2. Tải và giải nén feature `star` (dùng chung file cache với `flora_repro.ipynb`).
3. In thống kê KG offline.
4. Chạy thử `both` một seed trên 48/12, `grep` log để xác nhận dòng "KG neighbors on" và "KG compose on".
5. Bốn chế độ × 3 seed trên 48/12, feature `star`.
6. Tổng hợp: `last`, `best`, trung bình ± độ lệch chuẩn qua seed, và `delta` so với `none` cùng seed.

Log: `Drive/.../flora_nesy/<kg>/<task>_ZSL_seed<k>/run.log`. Nếu `none` khớp bài báo (trong 1-2 điểm ở `best`) thì baseline sạch.

## 6. Chưa làm và cần nhớ

- **Chưa có kết quả huấn luyện nào.** Máy này không có GPU, CLIP, hay feature; chỉ kiểm tra được cú pháp, đường dẫn config, và phần ma trận KG offline. Chưa biết NeSy-KG có cộng được điểm không.
- **Số chính để so là `last`.** `best` chọn theo unseen test nên lạc quan, chỉ để đối chiếu bài báo.
- **α và β để mặc định (0.5, 0.3).** Dò chúng trên unseen test là chọn trên test. Muốn chọn sạch thì phải tách validation từ seen (Flora không có sẵn).
- GZSL chưa chạy (`--kg-mode` dùng được cho GZSL nhưng mỗi lần đánh giá tốn gấp 25-145 lần ZSL).
- Kiểu 3 (vector primitive từ detector hình học làm đầu vào thêm) và kiểu 4 (ground primitive trên vector đã gộp) chưa làm, để sau nếu kiểu 1 và 2 chưa đủ.
- **`paper_notes.md` dòng 4 đã cũ**: vẫn ghi "bỏ nhánh KG / NeSy". Cần sửa cho khớp hướng hiện tại. Đề cương docx cũng đã được reframe quanh Neuron và zero-shot (ghi chú trong bộ nhớ), có thể cần chỉnh theo baseline mới.
- `zeroshot/Neuron/tsne_feats.py` vẫn chưa commit (có sẵn từ trước).
- Notebook `flora_repro.ipynb` giữ nguyên là bản tái hiện Flora gốc.

## 7. Bước tiếp theo gợi ý

1. Chạy `flora_nesy.ipynb` ô thử, rồi ablate bốn chế độ trên 48/12.
2. Nếu `neighbors` kém `none`: kiểm tra 4 cặp wear/take off, giảm α, hoặc thêm primitive chiều vào KG.
3. Nếu có cải thiện: mở rộng sang 96/24 và các split NTU120, rồi 3 seed đủ để báo cáo ± độ lệch chuẩn.
4. Cập nhật `paper_notes.md`, đề cương, và so sánh với Neuron đã có.

Neuro-symbolic AI (AI thần kinh-biểu tượng) là hướng tiếp cận kết hợp hai trường phái lớn của trí tuệ nhân tạo:

Neural (kết nối/học sâu): mạng nơ-ron học từ dữ liệu, mạnh ở nhận dạng mẫu, xử lý hình ảnh, ngôn ngữ, dữ liệu nhiễu. Hạn chế: cần nhiều dữ liệu, khó giải thích, dễ “ảo giác”, yếu ở suy luận logic nhiều bước.
Symbolic (biểu tượng): biểu diễn tri thức bằng ký hiệu, luật, logic, đồ thị tri thức, chương trình. Mạnh ở suy luận chặt chẽ, giải thích được, tận dụng tri thức có sẵn. Hạn chế: “giòn” (brittle), khó xử lý dữ liệu thô và mơ hồ, tốn công xây dựng thủ công.

Ý tưởng cốt lõi: để phần neural lo việc tri giác và học, phần symbolic lo việc suy luận, ràng buộc và giải thích, nhằm bù điểm yếu cho nhau.

Một số kiểu kết hợp thường gặp:

Neural trích xuất khái niệm từ dữ liệu thô, rồi symbolic suy luận trên đó (ví dụ: nhận diện đối tượng trong ảnh rồi trả lời câu hỏi bằng chương trình logic, như NS-VQA).
LLM sinh ra mã hoặc biểu thức hình thức, rồi bộ giải/trình thông dịch thực thi và kiểm chứng (ví dụ: AlphaGeometry, các hệ thống chứng minh định lý có công cụ kiểm tra).
Nhúng luật logic hoặc ràng buộc vào quá trình huấn luyện (hàm mất mát có ràng buộc logic, Logic Tensor Networks, DeepProbLog).
Dùng đồ thị tri thức để “neo” đầu ra của mô hình ngôn ngữ (retrieval có cấu trúc).

Một số điểm cần nhìn phê phán, vì đây là lĩnh vực hay bị thổi phồng:

Không có định nghĩa thống nhất. Ranh giới khá mờ: một LLM gọi công cụ tính toán hay trình thông dịch Python có được coi là neuro-symbolic không? Tùy tác giả mà câu trả lời khác nhau.
Lời hứa “dễ giải thích” không tự động đúng. Phần neural vẫn là hộp đen, nên toàn hệ thống chỉ giải thích được một phần.
Điểm nghẽn thật sự là giao diện giữa hai phần: chuyển đổi liên tục (vector) sang rời rạc (ký hiệu) thường không khả vi, khiến việc huấn luyện đầu-cuối rất khó.
Bằng chứng thực nghiệm còn hạn chế ở quy mô lớn; nhiều kết quả ấn tượng nằm ở bài toán có cấu trúc rõ (hình học, toán, câu hỏi hình ảnh tổng hợp). Cần thận trọng khi suy rộng sang bài toán đời thực.

Nếu bạn cho biết bối cảnh dùng (nghiên cứu, bài thuyết trình, chọn kiến trúc cho dự án), tôi có thể đi sâu vào ví dụ hoặc so sánh phù hợp hơn. Lưu ý rằng câu hỏi của bạn khá ngắn nên tôi chưa biết bạn cần mức độ khái niệm hay kỹ thuật.

Không có một phân loại chuẩn duy nhất. Phân loại được trích dẫn nhiều nhất là của Henry Kautz (bài giảng AAAI 2020), gồm 6 kiểu. Ngoài ra còn cách phân loại theo chức năng.

Phân loại 6 kiểu của Kautz
#	Kiểu	Ý tưởng	Ví dụ thường dẫn
1	Symbolic → Neuro → Symbolic	Đầu vào và đầu ra là ký hiệu (từ, token), bên trong là mạng nơ-ron	Mô hình ngôn ngữ, dịch máy
2	Symbolic[Neuro]	Bộ khung suy luận/tìm kiếm ký hiệu gọi mạng nơ-ron như một hàm con	AlphaGo (tìm kiếm cây + mạng đánh giá), AlphaGeometry
3	Neuro; Symbolic	Mạng nơ-ron biến dữ liệu thô thành khái niệm, rồi bộ suy luận ký hiệu xử lý tiếp	NS-VQA, Neuro-Symbolic Concept Learner
4	Neuro:Symbolic → Neuro	Tri thức ký hiệu được “biên dịch” vào quá trình huấn luyện (sinh dữ liệu, hướng dẫn học)	Dùng luật để sinh dữ liệu huấn luyện, chưng cất tri thức
5	Neuro_Symbolic	Luật logic trở thành ràng buộc mềm trong hàm mất mát	Logic Tensor Networks, Semantic Loss
6	Neuro[Symbolic]	Cơ chế suy luận ký hiệu được nhúng khả vi bên trong mạng	Neural Theorem Provers, DeepProbLog

Kautz coi kiểu 6 là hướng “hoàn chỉnh” nhất, nhưng đây cũng là kiểu khó nhất và có ít kết quả ở quy mô lớn nhất.

Cách phân loại theo chức năng (hay gặp trong các bài khảo sát)
Learning for reasoning: neural hỗ trợ suy luận (ví dụ dùng mạng để chọn bước tiếp theo cho bộ chứng minh).
Reasoning for learning: tri thức ký hiệu hỗ trợ học (ràng buộc, tiên nghiệm, giải thích).
Learning-reasoning: hai phần tương tác qua lại hai chiều.
Những điểm cần nhìn phê phán
Ranh giới giữa các kiểu mờ. Cùng một hệ thống có thể bị xếp vào kiểu khác nhau tùy tác giả. Việc xếp LLM vào kiểu 1 là cách hiểu rất rộng (“token là ký hiệu”), nhiều người cho rằng làm vậy khiến khái niệm gần như mất ý nghĩa.
Đây là phân loại theo kiến trúc, không phải theo hiệu quả. Kiểu nằm ở “mức độ tích hợp” cao hơn không có nghĩa hệ thống tốt hơn. Trên thực tế, các kiểu ghép lỏng (2 và 3) đang cho kết quả thực dụng nhất.
Tôi đang trả lời từ trí nhớ về một bài giảng, nên tên kiểu và ví dụ có thể lệch so với bản gốc hoặc các bài khảo sát mới hơn. Nếu bạn định trích dẫn, hãy đối chiếu trực tiếp với slide của Kautz hoặc khảo sát của Garcez & Lamb.
Bạn chưa nói mục đích. Nếu bạn cần phân loại để viết bài học thuật, thì nên dùng một khung và nêu rõ nó chỉ là một trong nhiều cách. Nếu bạn cần chọn kiến trúc cho dự án, thì phân loại này ít giúp ích bằng việc hỏi: bài toán của tôi cần ràng buộc cứng hay mềm, và có tri thức ký hiệu sẵn không?
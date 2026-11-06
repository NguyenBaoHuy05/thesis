# Xây KG cho RVF-HGCN bằng LLM: quy trình và prompt

> Trạng thái: **kế hoạch, chưa triển khai**.
> Phục vụ: [A1_RETRIEVE_VERIFY_FUSE_PLAN.md](../A1_RETRIEVE_VERIFY_FUSE_PLAN/A1_RETRIEVE_VERIFY_FUSE_PLAN.md) (RVF-HGCN), §2 "Knowledge graph".
> Sản phẩm: `annotations/action_kg.json` (một KG duy nhất), kèm toàn bộ prompt và câu trả lời thô
> của LLM để tái lập.

## 0. Nguyên tắc

| # | Nguyên tắc | Lý do |
|---|---|---|
| P1 | **Chỉ mô tả những gì skeleton quan sát được** (25 khớp × ≤ 2 người, không vật thể, không cử động ngón tay chi tiết) | Thuộc tính không quan sát được thì đầu thuộc tính chỉ còn học lại nhãn lớp |
| P2 | **Thuộc tính không được là tên lớp viết lại** (`reverse_of_dressing`, `drinking_motion` bị loại) | Nếu không, VERIFY chỉ là một classifier thứ hai |
| P3 | **Từ vựng thuộc tính cố định trước, gán cạnh sau** | Cho LLM đặt tên tự do khi gán cạnh sẽ sinh hàng trăm từ đồng nghĩa |
| P4 | **Mọi bước lưu prompt, model id, nhiệt độ và câu trả lời thô** | Tái lập được; báo cáo được trong luận văn |
| P5 | **Chạy lặp nhiều lần và bỏ phiếu**, ghi độ đồng thuận làm độ tin cậy | Một lần gọi LLM là nhiễu |
| P6 | **Không dùng kết quả trên tập test để sửa KG** | Tránh rò rỉ; vòng sửa (S6) dùng ma trận nhầm trên tập validation tách từ train |
| P7 | Mô tả bằng **tiếng Anh** | Text encoder (CLIP, sentence-transformer) mạnh nhất ở tiếng Anh |

Ràng buộc chung của RVF: không predicate hình học. LLM **không** được sinh ngưỡng số
(góc, khoảng cách); thuộc tính là khái niệm ngôn ngữ, bằng chứng đến từ backbone.

## 1. Tổng quan quy trình

```
S0 Chuẩn bị đầu vào ─► S1 Định nghĩa node HÀNH ĐỘNG (120)
                          │
                          ▼
                       S2 Đề xuất từ vựng THUỘC TÍNH (3 lần chạy) ─► S3 Gộp, lọc, duyệt tay  ──┐
                                                                                             │
          ┌──────────────────────────────────────────────────────────────────────────────────┘
          ▼
       S4 Gán cạnh has/lacks (3 lần chạy, bỏ phiếu) ─► S5 Quan hệ action–action, attr–attr
          │
          ▼
       S6 Vòng sửa độ phân biệt (validator ─► LLM đề xuất thuộc tính phân biệt ─► về S3)
          │
          ▼
       S7 Duyệt tay có ưu tiên ─► S8 Đóng băng phiên bản, sinh KG xáo trộn
```

## 2. Lược đồ đầu ra `annotations/action_kg.json`

```json
{
  "version": "1.0", "hash": "sha256:...", "built": "2026-10-20",
  "llm": {"model": "<model id>", "temperature": 0.3, "runs": 3},

  "attributes": {
    "hand_to_head": {
      "group": "contact",
      "text": "a hand moves to touch the head or face",
      "definition": "At some point one or both hands come into contact with, or very close to, the head or face.",
      "body_parts": ["hand_R", "hand_L", "head"],
      "positive_examples": ["drink_water", "brush_teeth", "phone_call"],
      "negative_examples": ["clapping", "kicking"],
      "observable": "direct",
      "source": "llm", "reviewed": true
    }
  },

  "actions": {
    "drink_water": {
      "id": 0, "persons": 1,
      "text": "a person drinks water from a cup",
      "paraphrases": ["someone lifts a cup to the mouth and drinks", "..."],
      "motion": "One hand rises from waist level to the mouth, stays there briefly while the head tilts back, then lowers.",
      "body_parts": ["hand_R", "arm_R", "head"],
      "phases": {"start": "hand rises toward mouth", "middle": "hand at mouth, head tilts back", "end": "hand lowers"},
      "not_observable": ["the cup", "swallowing"],
      "has":   {"hand_to_head": {"w": 2, "agree": 1.0}, "tilt_head_back": {"w": 2, "agree": 0.67}},
      "lacks": {"repetitive": {"w": 1, "agree": 1.0}, "legs_active": {"w": 2, "agree": 1.0}},
      "similar": ["eat_meal", "brush_teeth"], "opposite": [],
      "source": "llm", "reviewed": true
    }
  },

  "attribute_relations": {"implies": [["squat", "bend_knees"]], "excludes": [["sit", "jump"]]}
}
```

- `text` → embedding chính của node; `paraphrases` → lấy trung bình embedding (ổn định hơn một câu).
- `motion`, `phases` → **chỉ dùng để LLM suy luận ở S2/S4 và để người duyệt đọc**; `phases` để
  dành cho A5 (luật thời gian), RVF không dùng.
- `agree` = tỷ lệ số lần chạy đồng ý (S4); dùng để lọc cạnh yếu và ưu tiên duyệt tay.
- `observable`: `direct` (thấy trực tiếp trên khớp) | `inferred` (suy từ tư thế, ví dụ cầm vật nặng).
  Thuộc tính `inferred` được giữ nhưng báo cáo riêng, có ablation bỏ chúng.

## 3. Các bước chi tiết

### S0. Chuẩn bị đầu vào

| Đầu vào | Nguồn |
|---|---|
| 120 tên lớp, thứ tự nhãn | `NTU_ACTION_NAMES` trong `model/concept_schema.py` |
| Số người mỗi lớp | NTU120: A50–A60 và A106–A120 là tương tác hai người |
| Mô tả khả năng của skeleton | Viết tay, đưa vào mọi system prompt (§4.0) |
| Cặp hay nhầm | `analyze_confusion.py` trên **tập validation tách từ train** (P6), không phải tập test |
| Nhóm thuộc tính | Danh sách cố định ở §4.2 |

### S1. Định nghĩa node hành động

- Gọi theo lô 10 lớp/lần (đủ ngữ cảnh để LLM phân biệt các lớp gần nhau, không quá dài).
  Sắp các lớp gần nhau vào cùng lô (ví dụ put_on/take_off cùng lô) để mô tả có tính đối chiếu.
- 3 lần chạy. `text` chọn bản được người duyệt chấp nhận; `paraphrases` lấy từ các lần chạy.
- Kiểm tra tự động: `motion` không chứa vật thể như một bằng chứng; `body_parts` thuộc danh sách
  chuẩn; `persons` khớp với S0.

### S2. Đề xuất từ vựng thuộc tính

- Đầu vào: **toàn bộ 120 định nghĩa S1** (chỉ `motion` + `body_parts`, bỏ tên lớp ở một trong
  3 lần chạy để giảm xu hướng đặt tên theo lớp).
- Yêu cầu theo từng nhóm ở §4.2, mỗi nhóm 5–15 ứng viên, mỗi ứng viên có định nghĩa, ví dụ dương,
  ví dụ âm và ước lượng số lớp có thuộc tính.
- 3 lần chạy (có thể dùng 2 LLM khác nhau) → hợp các ứng viên, thường 120–200 ứng viên.

### S3. Gộp, lọc, duyệt từ vựng

1. **Gộp tự động**: embedding định nghĩa, cụm các ứng viên có cosine > 0.85.
2. **Gộp bằng LLM** (§4.3): với mỗi cụm, chọn một tên chuẩn và viết lại định nghĩa.
3. **Lọc theo luật**:
   - vi phạm P1 (không quan sát được) → loại hoặc đánh dấu `inferred`;
   - vi phạm P2 (ứng viên chỉ có đúng một lớp, hoặc tên chứa tên lớp) → loại;
   - tần suất ước lượng > 60% số lớp → loại (`d_attr` gần như luôn bằng 0).
4. **Duyệt tay** toàn bộ từ vựng (~50–80 mục còn lại): đây là chỗ công người đáng giá nhất.
5. Mục tiêu: **40–60 thuộc tính**, mỗi nhóm ≥ 3.

### S4. Gán cạnh has/lacks

- Mỗi lần gọi: **một** hành động + toàn bộ từ vựng đã chốt (§4.4). Một lớp/lần để LLM xét từng
  thuộc tính, không bỏ sót.
- Với mỗi thuộc tính, LLM trả một trong: `has:2` (thiết yếu), `has:1` (thường có), `unknown`,
  `lacks:1` (thường không), `lacks:2` (chắc chắn không). Kèm một câu bằng chứng ngắn cho mọi
  quyết định khác `unknown`.
- 3 lần chạy, nhiệt độ 0.3–0.7. **Bỏ phiếu**:
  - nhãn đa số (≥ 2/3) được giữ, `agree` = tỷ lệ đồng ý;
  - không có đa số → `unknown` (không giám sát thuộc tính cho cặp đó);
  - `has` và `lacks` cùng xuất hiện → `unknown` và đưa vào danh sách duyệt tay.
- `lacks` chỉ giữ khi có ý nghĩa phân biệt: ghi `lacks` cho thuộc tính mà **một lớp gần** có
  (từ `similar`, từ cặp nhầm). Không ghi hàng loạt `lacks` cho mọi thuộc tính không liên quan
  (để `unknown`), nếu không BCE bị ngập nhãn âm dễ.

### S5. Quan hệ giữa các node

- **`similar`**: LLM đề xuất 2–5 lớp gần nhất cho mỗi lớp (§4.5), hợp với cặp nhầm của S0;
  ép đối xứng; người duyệt các cặp chỉ một nguồn đề xuất.
- **`opposite`**: cặp đảo ngược thời gian (put_on/take_off, sit_down/stand_up, walking_towards/
  walking_apart...). Danh sách ngắn, viết tay rồi để LLM bổ sung, người duyệt hết.
- **`implies` / `excludes`** giữa thuộc tính: LLM đề xuất, người duyệt; validator kiểm tra không
  có lớp nào vi phạm (lớp `has` A mà A `excludes` B thì lớp đó không được `has` B).

### S6. Vòng sửa độ phân biệt

```
validator ──► danh sách cặp lớp (c1, c2) có
                 · vector thuộc tính (has/lacks) trùng nhau, hoặc
                 · khoảng cách Hamming < 2 VÀ là cặp nhầm trên validation
          ──► prompt §4.6: "thuộc tính quan sát được nào phân biệt c1 với c2?"
          ──► (a) thuộc tính đã có nhưng gán thiếu → sửa cạnh
              (b) thuộc tính mới → đưa vào S3 (cùng luật lọc), rồi S4 cho các lớp liên quan
```

- Tối đa 2 vòng. Các cặp đảo thời gian (`opposite`) được phép trùng vector: RVF không tách
  được chúng bằng thuộc tính tĩnh, đó là việc của A5. Ghi lại danh sách này.

### S7. Duyệt tay

| Ưu tiên | Đối tượng | Kiểm tra |
|---|---|---|
| 1 | Toàn bộ từ vựng thuộc tính | P1, P2, định nghĩa rõ, không trùng nghĩa |
| 2 | Lớp trong top-25 cặp nhầm (validation) | Cạnh has/lacks phân biệt được cặp nhầm |
| 3 | Cạnh có `agree` < 1 hoặc mâu thuẫn ở S4 | Quyết định cuối |
| 4 | Mẫu ngẫu nhiên 20 lớp còn lại | Ước lượng tỷ lệ lỗi của phần chưa duyệt |

- Bảng duyệt: xuất CSV `class, attribute, llm_label, agree, evidence, human_label`; người chỉ
  sửa cột cuối, script ghi ngược vào JSON với `source = "human"`.
- Báo cáo trong luận văn: **tỷ lệ cạnh LLM được người chấp nhận**, ước lượng trên mẫu ngẫu nhiên.
- Ước tính công: ~1 ngày cho từ vựng + ưu tiên 2, ~0.5 ngày cho ưu tiên 3–4.

### S8. Đóng băng

- `validate_kg.py` sạch → ghi `hash`, `version`; commit JSON + `annotations/kg_raw/` + prompt.
- Sinh 3 KG xáo trộn giữ bậc cho K2 (`tools/shuffle_kg.py`).
- Sinh `annotations/kg_emb_*.npz` (text embedding) từ đúng phiên bản này.
- Hai phiên bản để ablation A1-kg: `llm_only` (trước S7) và `reviewed` (sau S7).

## 4. Prompt

Quy ước: `{{...}}` là biến điền bằng script. Mọi prompt yêu cầu **chỉ trả JSON** theo schema;
script kiểm tra schema và gọi lại (tối đa 2 lần) nếu sai.

### 4.0 System prompt chung

```text
You are helping build a knowledge graph for skeleton-based human action recognition
on the NTU RGB+D 120 dataset.

What the sensor sees: 3D positions of 25 body joints per person (head, neck, spine,
shoulders, elbows, wrists, hands, hand tips, thumbs, hips, knees, ankles, feet), for at
most 2 people, at about 30 frames per second, for 1-10 seconds.

What the sensor does NOT see: objects (cups, phones, paper, clothes), fine finger
motion, facial expression, sound, gaze, clothing, the room.

Rules:
1. Describe only what could be visible in the joint positions and their motion.
   If something is essential to the action but not visible (e.g. the cup), list it
   under "not_observable" and do not rely on it.
2. Never give numeric thresholds (angles, distances, speeds).
3. Use plain, concrete English. No metaphors.
4. Answer with JSON only, matching the requested schema exactly.
```

### 4.1 S1 — Định nghĩa hành động (lô 10 lớp)

```text
Define each of the following actions as it would appear in skeleton data.

Actions (id, name, number of people):
{{batch: e.g. 13 put_on_jacket 1; 14 take_off_jacket 1; ...}}

For each action return:
- "text": one sentence, 8-15 words, "a person ..." (or "two people ..."), describing the
  action itself (this sentence is embedded by a text encoder).
- "paraphrases": 3 different sentences with the same meaning and the same length range.
- "motion": 2-4 sentences describing the visible body motion from start to end:
  which body parts move, in which direction, relative to which other body parts,
  whether it repeats, how the posture changes.
- "body_parts": the main moving parts, chosen from
  {{body_part_list: head, torso, hip, hand_R, hand_L, arm_R, arm_L, leg_R, leg_L, feet}}.
- "phases": {"start": ..., "middle": ..., "end": ...}, each a short phrase.
- "not_observable": things essential to the action that the sensor cannot see.
- "confusable_with": up to 3 action names from the full list below that would look
  most similar in skeleton data, with one phrase each on what differs visibly.

Full list of action names (for "confusable_with" only):
{{all_120_names}}

Schema:
{"actions": [{"id": int, "name": str, "text": str, "paraphrases": [str, str, str],
  "motion": str, "body_parts": [str], "phases": {"start": str, "middle": str, "end": str},
  "not_observable": [str], "confusable_with": [{"name": str, "visible_difference": str}]}]}
```

`confusable_with` cũng là một nguồn cho `similar` (S5) và cho danh sách cặp cần phân biệt (S6).

### 4.2 S2 — Đề xuất từ vựng thuộc tính

Các nhóm cố định:

| Nhóm | Câu hỏi nhóm trả lời | Ví dụ mong muốn |
|---|---|---|
| `contact` | Bộ phận nào chạm / đến gần bộ phận nào? | `hand_to_head`, `hand_to_hand`, `hand_to_foot`, `hand_to_partner` |
| `active_part` | Bộ phận nào chuyển động chính? | `one_arm_active`, `both_arms_active`, `legs_active`, `head_active` |
| `motion_type` | Chuyển động kiểu gì? | `raise_arm`, `lower_arm`, `swing`, `rotate_wrist`, `push_forward`, `pull_back` |
| `posture` | Tư thế nào xuất hiện hoặc thay đổi? | `bend_torso`, `squat`, `sit`, `tilt_head_back`, `arms_crossed` |
| `temporal` | Nhịp và thứ tự? | `repetitive`, `sudden`, `slow`, `posture_change`, `returns_to_start` |
| `location` | Tay hoạt động ở vùng nào của cơ thể? | `hands_above_head`, `hands_at_chest`, `hands_at_waist`, `hands_near_floor` |
| `interaction` | (hai người) quan hệ giữa hai người? | `approach`, `move_apart`, `mutual_contact`, `one_acts_other_reacts` |
| `whole_body` | Toàn thân dịch chuyển? | `locomotion`, `jump`, `fall`, `balance_loss` |

```text
Below are visible-motion definitions of the {{n}} actions of NTU RGB+D 120.
{{for each action: "<name>: <motion> Body parts: <body_parts>"}}
{{variant B: same list with names replaced by "action_<id>"}}

Propose ATTRIBUTES for the group "{{group}}": {{group_question}}

An attribute is a visible property of the motion that is shared by several actions.
Requirements:
- Visible in joint positions (see the system rules). Mark "observable": "inferred" if it
  can only be inferred indirectly (e.g. carrying something heavy from a bent posture).
- Shared by at least 2 actions and by at most about half of them.
- Not a renamed action: "drinking_motion" or "reverse_of_dressing" are NOT attributes.
- Name: snake_case, 1-4 words, describing motion or posture, not intention.
- Prefer attributes that tell apart actions that look alike.

Propose 5 to 15 attributes. For each give:
{"name": str, "group": "{{group}}", "text": "one short sentence for a text encoder",
 "definition": "1-2 sentences an annotator could apply consistently",
 "body_parts": [str], "observable": "direct" | "inferred",
 "positive_examples": [3-6 action names], "negative_examples": [3-6 action names],
 "estimated_count": int,
 "distinguishes": [["action_a", "action_b"], ...]}

Schema: {"attributes": [ ...as above... ]}
```

Chạy 8 nhóm × 3 lần (lần 3 dùng biến thể B ẩn tên lớp). `interaction` chỉ đưa các lớp hai người.

### 4.3 S3 — Gộp cụm thuộc tính

```text
These candidate attributes were proposed separately and may mean the same thing:
{{cluster: name, definition, positive_examples for each}}

Decide:
- "merge": they describe the same visible property -> give one canonical name, text and
  definition covering all of them.
- "keep_separate": they differ in a way that is visible in skeleton data -> explain the
  difference in one sentence and return each with a sharpened definition.
- "drop": any member that violates the requirements (not visible, renamed action,
  too generic). Give the reason.

Schema: {"decision": "merge" | "keep_separate", "attributes": [{"name", "text", "definition",
 "group", "body_parts", "observable"}], "dropped": [{"name": str, "reason": str}]}
```

### 4.4 S4 — Gán cạnh cho một hành động

```text
Action: {{name}} ({{persons}} person(s))
Visible motion: {{motion}}
Phases: {{phases}}
Not observable: {{not_observable}}
Actions that look similar: {{similar_and_confusable with their visible differences}}

Attribute vocabulary (name: definition):
{{all ~50 attributes}}

For EVERY attribute, decide how it relates to this action as seen in skeleton data:
- "has2": essential, present in almost every performance of the action
- "has1": usually present, but some people perform the action without it
- "unknown": unrelated, or it depends on the person
- "lacks1": usually absent
- "lacks2": clearly absent, and its absence helps tell this action apart from a similar one

Use "lacks1"/"lacks2" only when the attribute is present in one of the similar actions
listed above or when its absence is otherwise informative; otherwise use "unknown".
Give a short visible evidence phrase for every label other than "unknown".

Schema: {"action": "{{name}}", "labels": {"<attribute>": {"label": "has2" | "has1" |
 "unknown" | "lacks1" | "lacks2", "evidence": str}}}
```

Ánh xạ sang JSON: `has2 → has w=2`, `has1 → has w=1`, `lacks1 → lacks w=1`, `lacks2 → lacks w=2`.

### 4.5 S5 — Quan hệ action–action

```text
Action: {{name}} — {{motion}}
Candidates: {{all other action names with their "text"}}

1. "similar": up to 5 actions whose skeleton motion is most alike (same body parts, same
   kind of movement), most similar first, each with the main visible difference.
2. "opposite": actions that are the same motion performed in reverse temporal order
   (e.g. put on vs take off). Usually zero or one.

Schema: {"similar": [{"name": str, "visible_difference": str}],
         "opposite": [{"name": str, "reason": str}]}
```

```text
Attribute vocabulary: {{name: definition}}
List pairs (A, B) where:
- "implies": every action with A necessarily has B in skeleton data;
- "excludes": no action can have A and B at the same time.
Only list pairs that hold by definition, not by coincidence in this dataset.
Schema: {"implies": [[str, str]], "excludes": [[str, str]]}
```

### 4.6 S6 — Thuộc tính phân biệt cho một cặp

```text
Two actions currently have the same attribute profile, but they must be told apart:
- {{c1}}: {{motion_c1}}  attributes: {{has/lacks c1}}
- {{c2}}: {{motion_c2}}  attributes: {{has/lacks c2}}

Attribute vocabulary: {{name: definition}}

1. Is there an attribute in the vocabulary that one action has and the other lacks,
   but which is labelled wrongly or missing? List corrections.
2. If none exists, propose at most 2 NEW attributes that are visible in skeleton data,
   shared by other actions too (list them), and not a renamed action.
3. If the two actions differ only in temporal order or in something not visible
   (objects), say so and propose nothing.

Schema: {"corrections": [{"action": str, "attribute": str, "label": str, "evidence": str}],
         "new_attributes": [{"name", "group", "text", "definition", "body_parts",
                             "observable", "positive_examples", "negative_examples"}],
         "not_separable": {"reason": "temporal_order" | "not_observable" | null}}
```

## 5. Hạ tầng

| File | Nội dung |
|---|---|
| `annotations/kg_prompts/*.txt` | Đúng các prompt §4, có số phiên bản ở dòng đầu |
| `annotations/kg_raw/<stage>/<run>/<key>.json` | Câu trả lời thô + model id, nhiệt độ, thời điểm, hash prompt |
| `tools/kg_llm.py` | Gọi LLM (provider tùy chọn), cache theo hash (prompt + biến) → chạy lại không tốn phí, kiểm tra schema, gọi lại khi sai |
| `tools/kg_build.py` | S1–S6: ghép kết quả, bỏ phiếu, gộp, ghi `action_kg.json` |
| `tools/kg_review.py` | Xuất/nhập CSV duyệt tay (S7) |
| `tools/validate_kg.py`, `tools/shuffle_kg.py` | Như plan A1 |

- Dùng một LLM mạnh cho S1, S2, S4 và ghi model id cụ thể. Có thể dùng LLM thứ hai cho một trong
  3 lần chạy S2/S4 để giảm thiên lệch của một mô hình.
- Chi phí ước lượng: S4 là lớn nhất (120 lớp × 3 lần, mỗi lần ~3k token vào) ≈ 1M token vào; tất
  cả các bước cộng lại cỡ 2M token.

## 6. Chỉ số chất lượng KG (báo cáo trong luận văn)

| Chỉ số | Mục tiêu |
|---|---|
| Số thuộc tính / nhóm | 40–60; mỗi nhóm ≥ 3 |
| Bậc thuộc tính (số lớp `has`) | Mọi thuộc tính trong [2, 60] |
| Số `has` / lớp | 3–10; `lacks` / lớp 1–5 |
| Đồng thuận giữa các lần chạy S4 | Tỷ lệ cạnh `agree = 1`; Fleiss' κ trên nhãn 5 mức |
| Tỷ lệ người chấp nhận (mẫu ngẫu nhiên) | Báo cáo con số, không đặt ngưỡng |
| Cặp lớp trùng vector | 0 ngoài danh sách `opposite` / không phân biệt được |
| Phủ cặp nhầm | % cặp nhầm top-25 (validation) có ≥ 2 thuộc tính phân biệt |
| Tỷ lệ thuộc tính `inferred` | Càng thấp càng tốt; có ablation bỏ chúng |

## 7. Lưu ý cho zero-shot (K6)

- Định nghĩa S1 và cạnh S4 của lớp unseen được sinh **bằng đúng quy trình** như lớp seen. Đây là
  thiết lập chuẩn trong zero-shot dựa trên thuộc tính: thuộc tính được biết cho mọi lớp, chỉ dữ liệu
  của lớp unseen là không có.
- **Vòng sửa S6 không được dùng cặp nhầm có lớp unseen** (sẽ cần dữ liệu của lớp unseen). Với các
  lần chạy K6, chỉ áp dụng S6 cho cặp seen–seen; ghi rõ trong báo cáo.
- Duyệt tay lớp unseen không được nhìn vào kết quả mô hình trên chúng.

## 8. Rủi ro

| Rủi ro | Xử lý |
|---|---|
| LLM mô tả theo vật thể và ý định thay vì chuyển động | System prompt §4.0, trường `not_observable`, luật lọc P1, biến thể ẩn tên lớp ở S2 |
| Từ vựng phình to, trùng nghĩa | Nhóm cố định, gộp bằng embedding + LLM, giới hạn 40–60 |
| Nhãn S4 không ổn định | 3 lần chạy, bỏ phiếu, `unknown` khi không đa số, `agree` làm độ tin cậy |
| Nhiều lớp không phân biệt được bằng thuộc tính tĩnh | Chấp nhận các cặp đảo thời gian (chuyển cho A5), ghi danh sách |
| Rò rỉ từ tập test | P6: ma trận nhầm từ validation tách từ train |
| Kết quả không tái lập | Cache câu trả lời thô, ghi model id và hash prompt |

## 9. Lộ trình

| Bước | Việc | Ước tính |
|---|---|---|
| S0 | Chia validation từ train, chạy `analyze_confusion.py`, viết system prompt | 0.5 ngày |
| S1 | Định nghĩa 120 lớp, kiểm tra tự động, đọc lướt | 0.5 ngày |
| S2–S3 | Đề xuất, gộp, duyệt từ vựng | 1–1.5 ngày |
| S4–S5 | Gán cạnh, bỏ phiếu, quan hệ | 0.5 ngày (chủ yếu chờ API) |
| S6 | 1–2 vòng sửa độ phân biệt | 0.5 ngày |
| S7 | Duyệt tay có ưu tiên | 1–1.5 ngày |
| S8 | Đóng băng, embedding, KG xáo trộn | 0.5 ngày |

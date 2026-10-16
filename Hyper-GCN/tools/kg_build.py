"""Assemble annotations/action_kg.json from the raw LLM answers (KG_CONSTRUCTION_LLM_PLAN.md).

  python tools/kg_build.py s1          check + merge S1 batches   -> kg_work/s1_actions.json
  python tools/kg_build.py s2-input    S2 inputs (named + anonymous definitions)
  python tools/kg_build.py s2-collect  all S2 candidates          -> kg_work/s2_candidates.json
  python tools/kg_build.py s4-input    S4 inputs (actions + vocabulary)
  python tools/kg_build.py s4-vote     majority vote of S4 runs   -> kg_work/s4_edges.json
  python tools/kg_build.py s5          symmetric relations        -> kg_work/s5_relations.json
  python tools/kg_build.py build       final KG                   -> annotations/action_kg.json
  python tools/kg_build.py report      statistics of every stage  -> kg_work/report.md (for KG_BUILD_LOG.md)
"""

import argparse
import glob
import hashlib
import json
import os
from collections import Counter, defaultdict

ROOT = "annotations"
RAW = os.path.join(ROOT, "kg_raw")
WORK = os.path.join(ROOT, "kg_work")
BODY_PARTS = {"head", "torso", "hip", "hand_R", "hand_L", "arm_R", "arm_L", "leg_R", "leg_L", "feet"}
LABELS = {"has2": ("has", 2), "has1": ("has", 1), "lacks1": ("lacks", 1), "lacks2": ("lacks", 2)}


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=1, ensure_ascii=False)


def actions_index():
    return load(os.path.join(ROOT, "kg_inputs", "actions.json"))["actions"]


def s1():
    index = actions_index()
    names = {a["name"] for a in index}
    merged, issues = {}, []
    for path in sorted(glob.glob(os.path.join(RAW, "s1", "run1", "batch_*.json"))):
        for action in load(path)["actions"]:
            expected = index[action["id"]]
            if action["name"] != expected["name"]:
                issues.append(f"{path}: id {action['id']} is {expected['name']}, got {action['name']}")
            bad_parts = set(action["body_parts"]) - BODY_PARTS
            if bad_parts:
                issues.append(f"{action['name']}: body parts {sorted(bad_parts)}")
            for item in action.get("confusable_with", []):
                if item["name"] not in names:
                    issues.append(f"{action['name']}: unknown confusable {item['name']}")
            if len(action.get("paraphrases", [])) != 3:
                issues.append(f"{action['name']}: {len(action.get('paraphrases', []))} paraphrases")
            action["persons"] = expected["persons"]
            merged[expected["name"]] = action
    missing = [a["name"] for a in index if a["name"] not in merged]
    dump({"actions": merged}, os.path.join(WORK, "s1_actions.json"))
    print(f"{len(merged)} actions, missing {missing}")
    print("\n".join(issues) or "no issues")


def s2_input():
    actions = load(os.path.join(WORK, "s1_actions.json"))["actions"]
    for anonymous in (False, True):
        lines = []
        for name, a in sorted(actions.items(), key=lambda kv: kv[1]["id"]):
            label = f"action_{a['id']}" if anonymous else name
            lines.append(f"{label} ({a['persons']} person(s)): {a['motion']} Body parts: {', '.join(a['body_parts'])}")
        suffix = "_anon" if anonymous else ""
        path = os.path.join(ROOT, "kg_inputs", f"s2_definitions{suffix}.txt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        print(path)


def s2_collect():
    names = {f"action_{a['id']}": a["name"] for a in actions_index()}

    def rename(items):
        return [rename(x) if isinstance(x, list) else names.get(x, x) for x in items]

    candidates = []
    for path in sorted(glob.glob(os.path.join(RAW, "s2", "*", "*.json"))):
        run = os.path.basename(os.path.dirname(path))
        for attribute in load(path)["attributes"]:
            attribute["run"] = run
            for key in ("positive_examples", "negative_examples", "distinguishes"):
                attribute[key] = rename(attribute.get(key, []))
            candidates.append(attribute)
    by_group = Counter(c["group"] for c in candidates)
    dump({"candidates": candidates}, os.path.join(WORK, "s2_candidates.json"))
    print(f"{len(candidates)} candidates", dict(by_group))


def s4_input():
    actions = load(os.path.join(WORK, "s1_actions.json"))["actions"]
    vocabulary = load(os.path.join(WORK, "s3_vocabulary.json"))["attributes"]
    items = []
    for name, a in sorted(actions.items(), key=lambda kv: kv[1]["id"]):
        similar = "; ".join(f"{c['name']} (differs: {c['visible_difference']})" for c in a["confusable_with"])
        items.append({"id": a["id"], "name": name, "persons": a["persons"], "motion": a["motion"],
                      "phases": a["phases"], "not_observable": a["not_observable"], "similar": similar})
    dump({"actions": items}, os.path.join(ROOT, "kg_inputs", "s4_actions.json"))
    lines = [f"{n}: {v['definition']}" for n, v in sorted(vocabulary.items(), key=lambda kv: (kv[1]["group"], kv[0]))]
    with open(os.path.join(ROOT, "kg_inputs", "s4_vocabulary.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"{len(items)} actions, {len(lines)} attributes")


def s4_vote():
    vocabulary = load(os.path.join(WORK, "s3_vocabulary.json"))["attributes"]
    runs = sorted(glob.glob(os.path.join(RAW, "s4", "run*")))
    votes = defaultdict(lambda: defaultdict(list))  # action -> attribute -> [label]
    evidence = defaultdict(dict)
    for run in runs:
        for path in glob.glob(os.path.join(run, "*.json")):
            for answer in load(path)["answers"]:
                for attr, item in answer["labels"].items():
                    if attr in vocabulary:
                        votes[answer["action"]][attr].append(item["label"])
                        if item["label"] in LABELS:
                            evidence[(answer["action"], attr)].setdefault(item["label"], item.get("evidence", ""))
    edges, conflicts, stats = {}, [], Counter()
    for action, per_attr in votes.items():
        has, lacks = {}, {}
        for attr, labels in per_attr.items():
            label, count = Counter(labels).most_common(1)[0]
            polarity = {LABELS[l][0] for l in labels if l in LABELS}
            if len(polarity) > 1:
                conflicts.append([action, attr, labels])
                stats["conflict"] += 1
                continue
            if count * 2 <= len(runs) or label not in LABELS:
                stats["unknown"] += 1
                continue
            kind, weight = LABELS[label]
            agree = round(count / len(runs), 2)
            stats[f"agree_{agree}"] += 1
            (has if kind == "has" else lacks)[attr] = {"w": weight, "agree": agree,
                                                      "evidence": evidence[(action, attr)].get(label, "")}
        edges[action] = {"has": has, "lacks": lacks}
    dump({"runs": len(runs), "edges": edges, "conflicts": conflicts}, os.path.join(WORK, "s4_edges.json"))
    print(f"{len(edges)} actions from {len(runs)} runs; {dict(stats)}")


def s5():
    """Symmetric similar/opposite. A similar pair needs >= 2 sources among S5 (both directions)
    and S1 confusable_with (both directions); single-source pairs go to the review list."""
    actions = load(os.path.join(WORK, "s1_actions.json"))["actions"]
    answers = load(os.path.join(RAW, "s5", "part_a.json"))["answers"]
    sources = Counter()
    opposite = defaultdict(set)
    for answer in answers:
        name = answer["action"]
        for item in answer["similar"]:
            if item["name"] in actions and item["name"] != name:
                sources[frozenset((name, item["name"]))] += 1
        for item in answer["opposite"]:
            if item["name"] in actions:
                opposite[name].add(item["name"])
                opposite[item["name"]].add(name)
    for name, a in actions.items():
        for item in a["confusable_with"]:
            if item["name"] in actions and item["name"] != name:
                sources[frozenset((name, item["name"]))] += 1
    similar, single = defaultdict(set), []
    for pair, count in sources.items():
        x, y = sorted(pair)
        if count >= 2:
            similar[x].add(y)
            similar[y].add(x)
        else:
            single.append([x, y])
    # Reversed pairs are opposite, not similar.
    for name in opposite:
        similar[name] -= opposite[name]
    order = {n: a["id"] for n, a in actions.items()}
    relations = {
        "similar": {n: sorted(v, key=order.get) for n, v in similar.items()},
        "opposite": {n: sorted(v, key=order.get) for n, v in opposite.items()},
        "attribute_relations": load(os.path.join(RAW, "s5", "part_b.json")),
        "single_source_similar": sorted(single),
    }
    dump(relations, os.path.join(WORK, "s5_relations.json"))
    degrees = [len(v) for v in similar.values()]
    print(f"similar pairs {sum(degrees) // 2} (actions with none: {len(actions) - len(similar)}), "
          f"single-source {len(single)}, opposite pairs {sum(len(v) for v in opposite.values()) // 2}")


def apply_s6(vocabulary, edges):
    """S6 repairs (single run): corrections and new attributes, marked source="s6", agree=None."""
    def put(action, attr, label, evidence):
        entry = edges.setdefault(action, {"has": {}, "lacks": {}})
        entry["has"].pop(attr, None)
        entry["lacks"].pop(attr, None)
        if label in LABELS:
            kind, weight = LABELS[label]
            entry[kind][attr] = {"w": weight, "agree": None, "evidence": evidence, "source": "s6"}

    for path in sorted(glob.glob(os.path.join(RAW, "s6", "*", "*.json"))):
        answer = load(path)
        for item in answer.get("corrections", []):
            put(item["action"], item["attribute"], item["label"], item.get("evidence", ""))
        for attribute in answer.get("new_attributes", []):
            name = attribute.pop("name")
            vocabulary[name] = dict(attribute, source="s6", runs=[os.path.relpath(path, RAW)])
        for attr, labels in answer.get("new_attribute_labels", {}).items():
            for action, item in labels.items():
                put(action, attr, item["label"], item.get("evidence", ""))


def build():
    actions = load(os.path.join(WORK, "s1_actions.json"))["actions"]
    vocabulary = load(os.path.join(WORK, "s3_vocabulary.json"))["attributes"]
    edges = load(os.path.join(WORK, "s4_edges.json"))["edges"]
    relations = load(os.path.join(WORK, "s5_relations.json"))
    apply_s6(vocabulary, edges)
    kg = {"version": "1.0", "llm": {"model": "claude-opus-5-5 (Claude Code subagents)", "runs_s4": 3},
          "attributes": vocabulary, "actions": {}, "attribute_relations": relations["attribute_relations"]}
    for name, a in sorted(actions.items(), key=lambda kv: kv[1]["id"]):
        kg["actions"][name] = {
            "id": a["id"], "persons": a["persons"], "text": a["text"], "paraphrases": a["paraphrases"],
            "motion": a["motion"], "body_parts": a["body_parts"], "phases": a["phases"],
            "not_observable": a["not_observable"],
            "has": edges.get(name, {}).get("has", {}), "lacks": edges.get(name, {}).get("lacks", {}),
            "similar": relations["similar"].get(name, []), "opposite": relations["opposite"].get(name, []),
            "source": "llm", "reviewed": False,
        }
    # An excludes violation keeps the stronger edge (weight, then agreement) and logs the drop.
    kg["dropped_edges"] = []
    for x, y in kg["attribute_relations"].get("excludes", []):
        for name, a in kg["actions"].items():
            if x in a["has"] and y in a["has"]:
                key = lambda attr: (a["has"][attr]["w"], a["has"][attr]["agree"])
                weak = min((x, y), key=key)
                kg["dropped_edges"].append({"action": name, "attribute": weak, "edge": a["has"].pop(weak),
                                            "reason": f"{x} excludes {y}"})
    payload = json.dumps(kg, sort_keys=True, ensure_ascii=False).encode("utf-8")
    kg["hash"] = "sha256:" + hashlib.sha256(payload).hexdigest()
    dump(kg, os.path.join(ROOT, "action_kg.json"))
    print(f"{len(kg['actions'])} actions, {len(vocabulary)} attributes -> annotations/action_kg.json")


def fleiss_kappa(table):
    """table: rows = items, columns = category counts; every row sums to the number of raters."""
    n = sum(table[0])
    items = len(table)
    p_cat = [sum(row[j] for row in table) / (items * n) for j in range(len(table[0]))]
    p_item = [(sum(c * c for c in row) - n) / (n * (n - 1)) for row in table]
    p_bar, p_e = sum(p_item) / items, sum(p * p for p in p_cat)
    return (p_bar - p_e) / (1 - p_e)


def report():
    """Markdown statistics of every stage, for the build log (KG_BUILD_LOG.md)."""
    out = []
    s1 = load(os.path.join(WORK, "s1_actions.json"))["actions"]
    words = [len(a["text"].split()) for a in s1.values()]
    out += ["### S1 — node hành động", "",
            f"- {len(s1)} hành động, {len(glob.glob(os.path.join(RAW, 's1', '*', '*.json')))} lô",
            f"- `text`: {min(words)}–{max(words)} từ (trung bình {sum(words) / len(words):.1f})",
            f"- Có `not_observable` khác rỗng: {sum(bool(a['not_observable']) for a in s1.values())}/{len(s1)}", ""]

    groups = sorted({c["group"] for c in load(os.path.join(WORK, "s2_candidates.json"))["candidates"]})
    out += ["### S2 — ứng viên thuộc tính", "", "| Run | " + " | ".join(groups) + " | Tổng |",
            "|---|" + "---|" * (len(groups) + 1)]
    for run_dir in sorted(glob.glob(os.path.join(RAW, "s2", "*"))):
        counts = Counter(a["group"] for p in glob.glob(os.path.join(run_dir, "*.json")) for a in load(p)["attributes"])
        out.append(f"| {os.path.basename(run_dir)} | " + " | ".join(str(counts[g]) for g in groups)
                   + f" | {sum(counts.values())} |")
    out.append("")

    vocabulary = load(os.path.join(WORK, "s3_vocabulary.json"))["attributes"]
    decisions = load(os.path.join(RAW, "s3", "decisions.json"))
    per_group = Counter(v["group"] for v in vocabulary.values())
    support = Counter(len(v.get("runs", [])) for v in vocabulary.values())
    dropped = len(decisions["dropped"]) + sum(len(c["members"]) for c in decisions["clusters"] if c["decision"] == "drop")
    out += ["### S3 — từ vựng", "",
            f"- {len(vocabulary)} thuộc tính: " + ", ".join(f"{g} {per_group[g]}" for g in groups),
            f"- Số run đề xuất: " + ", ".join(f"{k} run: {v}" for k, v in sorted(support.items(), reverse=True)),
            f"- `inferred`: {sum(v['observable'] == 'inferred' for v in vocabulary.values())}",
            f"- {len(decisions['clusters'])} cụm; {dropped} ứng viên bị loại", ""]

    runs = sorted(glob.glob(os.path.join(RAW, "s4", "run*")))
    names, attrs = sorted(s1), sorted(vocabulary)
    cats = ["has2", "has1", "unknown", "lacks1", "lacks2"]
    labels = {}
    out += ["### S4 — cạnh has/lacks", "", "| Run | Nhãn khác `unknown` | / hành động |", "|---|---|---|"]
    for run in runs:
        table = {}
        for path in glob.glob(os.path.join(run, "*.json")):
            for answer in load(path)["answers"]:
                for attr, item in answer["labels"].items():
                    if attr in vocabulary and item["label"] != "unknown":
                        table[(answer["action"], attr)] = item["label"]
        labels[run] = table
        out.append(f"| {os.path.basename(run)} | {len(table)} | {len(table) / len(names):.1f} |")
    cells = [[sum(labels[r].get((a, t), "unknown") == c for r in runs) for c in cats] for a in names for t in attrs]
    touched = [row for row in cells if row[cats.index("unknown")] < len(runs)]
    unanimous = sum(max(row) == len(runs) for row in touched)
    edges = load(os.path.join(WORK, "s4_edges.json"))
    agree = Counter(e["agree"] for x in edges["edges"].values() for k in ("has", "lacks") for e in x[k].values())
    out += ["", f"- Fleiss' κ (5 nhãn, {len(cells)} ô): **{fleiss_kappa(cells):.3f}**",
            f"- Ô có ít nhất một nhãn khác `unknown`: {len(touched)}; cả {len(runs)} run trùng nhãn: "
            f"{unanimous} ({100 * unanimous / max(len(touched), 1):.0f}%)",
            f"- Sau bỏ phiếu: {sum(agree.values())} cạnh (" + ", ".join(f"agree {k}: {v}" for k, v in sorted(agree.items(), reverse=True))
            + f"); {len(edges['conflicts'])} mâu thuẫn has/lacks → `unknown`", ""]

    relations = load(os.path.join(WORK, "s5_relations.json"))
    out += ["### S5 — quan hệ", "",
            f"- `similar`: {sum(map(len, relations['similar'].values())) // 2} cặp (≥ 2 nguồn); "
            f"{len(relations['single_source_similar'])} cặp một nguồn chờ duyệt",
            f"- `opposite`: {sum(map(len, relations['opposite'].values())) // 2} cặp",
            f"- `implies`: {len(relations['attribute_relations']['implies'])}, "
            f"`excludes`: {len(relations['attribute_relations']['excludes'])}", ""]

    s6 = [load(p) for p in sorted(glob.glob(os.path.join(RAW, "s6", "*", "*.json")))]
    out += ["### S6 — sửa độ phân biệt", "",
            f"- {len(s6)} cặp xử lý; {sum(len(x.get('corrections', [])) for x in s6)} sửa cạnh; thuộc tính mới: "
            + (", ".join(a["name"] for x in s6 for a in x.get("new_attributes", [])) or "không"), ""]

    kg = load(os.path.join(ROOT, "action_kg.json"))
    has = [len(a["has"]) for a in kg["actions"].values()]
    lacks = [len(a["lacks"]) for a in kg["actions"].values()]
    degree = Counter(t for a in kg["actions"].values() for t in a["has"])
    out += ["### KG cuối", "",
            f"- {len(kg['actions'])} hành động, {len(kg['attributes'])} thuộc tính, hash `{kg['hash'][:19]}…`",
            f"- `has`/hành động: {min(has)}–{max(has)} (trung bình {sum(has) / len(has):.1f}); "
            f"`lacks`/hành động: {min(lacks)}–{max(lacks)} (trung bình {sum(lacks) / len(lacks):.1f})",
            f"- Bậc thuộc tính: {min(degree.values())}–{max(degree.values())}; cao nhất: "
            + ", ".join(f"`{t}` {d}" for t, d in degree.most_common(3)),
            f"- Cạnh bị bỏ do `excludes`: {len(kg.get('dropped_edges', []))}",
            f"- Đã duyệt tay: {sum(a['reviewed'] for a in kg['actions'].values())}/{len(kg['actions'])} hành động", ""]
    path = os.path.join(WORK, "report.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(out))
    print(path)


if __name__ == "__main__":
    steps = {"s1": s1, "s2-input": s2_input, "s2-collect": s2_collect, "s4-input": s4_input,
             "s4-vote": s4_vote, "s5": s5, "build": build, "report": report}
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=steps)
    steps[parser.parse_args().step]()

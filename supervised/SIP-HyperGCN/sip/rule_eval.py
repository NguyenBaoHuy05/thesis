"""Evaluate induced rules (rules.json, pipeline v3) on TEST.

Per rule k (literals [column, op, theta] applied to TEST ``meas``; y = true label,
yhat = network base prediction = argmax of test_feats logits; S = rows satisfying the rule):

* coverage       = |S ∩ {y=k}| / |{y=k}|
* precision      = |S ∩ {y=k}| / |S|
* fidelity       = |S ∩ {yhat=k}| / |S|
* discrimination = precision restricted to rows with y in {k} ∪ C_k (C_k = confusable
                   classes stored in rules.json, else top-m of the TEST confusion)
* length (and number of "<" / ">" literals)
* necessity on TEST (``causal_dir`` of the TEST split):
  - measure necessity: for every measure of the rule, excess / p_class of
    causal_measures.npz at the measure's layer (strong ``layer_of``, else evidence layer);
    confirmed iff excess > 0 and p_class < alpha for all of them.
  - region necessity: for every measure, the region / layer recorded in the VAL evidence,
    looked up in causal_regions.npz; confirmed iff excess > 0 and p_class < alpha.
  - causally_confirmed = measure confirmed (and region confirmed when ``require_region``
    and the regions file is given).
* rule-based classifier: each row -> satisfied rule with the highest train precision
  (abstain if none); accuracy and agreement with the network (global fidelity).
* human-knowledge check vs the concepts CSV (direction-aware, via
  ``sip.concept_map.MEASURE_TO_CONCEPTS`` / ``class_level_measure_prior``), per literal
  ``m op theta``:
  - agree      : class k has a concept in MEASURE_TO_CONCEPTS[m][op]
  - disagree   : not agree, but class k has a concept of the opposite direction
  - not-listed : direction op is mapped, class k has none of those (nor opposite) concepts
  - no-mapping : measure / direction has no concept mapping in the CSV (or class not in CSV)
  Disagreements are findings, not errors.
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np

from .rule_induction import (column_measure, confusable_classes, literal_to_string,
                             rule_to_string, satisfies)

__all__ = ["knowledge_base", "literal_kb", "evaluate_rules", "rule_classifier", "write_reports"]

_OPP = {"<": ">", ">": "<"}


def _nan(x):
    return float("nan") if x is None else float(x)


def _f(x):
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def _ratio(a: int, b: int) -> float:
    return a / b if b else float("nan")


def _str_list(a) -> list[str]:
    return [str(x) for x in np.asarray(a).reshape(-1)]


# ----------------------------------------------------------------------------- knowledge
def knowledge_base(csv_path: str, num_class: int) -> dict:
    """Direction-aware class-level knowledge from the concepts CSV.

    Returns {"prior": {measure: {"<": (K',) bool, ">": (K',) bool}},
             "mapped": {measure: {"<": [concepts in CSV], ">": [...]}},
             "class_concepts": [set of concept names per class row], "n_rows": K'}.
    K' = min(num_class, CSV rows); classes beyond K' -> no-mapping."""
    from .concept_map import MEASURE_TO_CONCEPTS, class_level_measure_prior, load_concept_csv
    _, cols, mat = load_concept_csv(csv_path)
    n_rows = min(int(num_class), int(mat.shape[0]))
    colset = set(cols)
    mapped = {str(m): {op: [c for c in (d.get(op, []) or []) if c in colset] for op in _OPP}
              for m, d in MEASURE_TO_CONCEPTS.items()}
    raw = class_level_measure_prior(csv_path, num_class=n_rows)
    prior = {}
    for m, d in raw.items():
        prior[str(m)] = {op: np.asarray(d.get(op, np.zeros(n_rows)), dtype=np.float64)
                         .reshape(-1)[:n_rows] > 0.5 for op in _OPP}
    class_concepts = [{cols[j] for j in np.flatnonzero(mat[k] > 0.5)} for k in range(n_rows)]
    return {"prior": prior, "mapped": mapped, "class_concepts": class_concepts,
            "n_rows": n_rows}


def literal_kb(measure: str, op: str, k: int, kb: dict | None) -> str:
    """'agree' | 'disagree' | 'not-listed' | 'no-mapping' (see module doc)."""
    if kb is None or k >= kb["n_rows"]:
        return "no-mapping"
    mp = kb["mapped"].get(measure)
    pr = kb["prior"].get(measure)
    if not mp or pr is None:
        return "no-mapping"
    opp = _OPP[op]

    def has(direction):
        a = pr.get(direction)
        return bool(mp.get(direction)) and a is not None and k < a.shape[0] and bool(a[k])

    if has(op):
        return "agree"
    if has(opp):
        return "disagree"
    if mp.get(op):
        return "not-listed"
    return "no-mapping"


# ----------------------------------------------------------------------------- classifier
def rule_classifier(S: np.ndarray, rule_cls: list[int], train_prec: list[float],
                    train_sup: list[float]) -> np.ndarray:
    """S: (N, R) bool satisfied. Returns (N,) assigned class, -1 = abstain.
    Ties on train precision are broken by train support, then rule order."""
    N, R = S.shape
    if R == 0:
        return np.full(N, -1, dtype=np.int64)
    score = np.asarray([_nan(p) for p in train_prec], dtype=np.float64)
    score = np.where(np.isfinite(score), score, -1.0)
    sup = np.asarray([_nan(s) for s in train_sup], dtype=np.float64)
    score = score + 1e-6 * np.where(np.isfinite(sup), sup, 0.0)
    M = np.where(S, score[None, :], -np.inf)
    best = M.argmax(axis=1)
    out = np.asarray(rule_cls, dtype=np.int64)[best]
    out[~S.any(axis=1)] = -1
    return out


def _clf_stats(assign: np.ndarray, y: np.ndarray, yhat: np.ndarray) -> dict:
    cov = assign >= 0
    n = len(y)
    return {
        "n": int(n),
        "covered_frac": _ratio(int(cov.sum()), n),
        "accuracy": _ratio(int((assign == y).sum()), n),
        "accuracy_on_covered": _ratio(int(((assign == y) & cov).sum()), int(cov.sum())),
        "global_fidelity": _ratio(int((assign == yhat).sum()), n),
        "fidelity_on_covered": _ratio(int(((assign == yhat) & cov).sum()), int(cov.sum())),
        "network_accuracy": _ratio(int((yhat == y).sum()), n),
    }


# ----------------------------------------------------------------------------- causal lookup
class _Causal:
    def __init__(self, d: dict | None):
        self.ok = d is not None
        if not self.ok:
            return
        self.col = {n: i for i, n in enumerate(_str_list(d["unit_names"]))}
        self.layers = [int(v) for v in np.asarray(d["layers"]).reshape(-1)]
        self.excess = np.asarray(d["excess"], dtype=np.float64)
        self.p = np.asarray(d["p_class"], dtype=np.float64)

    def get(self, unit: str, layer, k: int):
        """(excess, p) or None if unavailable."""
        if (not self.ok or unit not in self.col or layer is None or int(layer) not in self.layers
                or k >= self.excess.shape[2]):
            return None
        li = self.layers.index(int(layer))
        return float(self.excess[li, self.col[unit], k]), float(self.p[li, self.col[unit], k])


def _confirmed(v, alpha) -> bool:
    return v is not None and np.isfinite(v[0]) and np.isfinite(v[1]) and v[0] > 0 and v[1] < alpha


# ----------------------------------------------------------------------------- main eval
def evaluate_rules(rules_js: dict, meas: np.ndarray, meas_cols, labels: np.ndarray,
                   logits: np.ndarray, causal_m: dict | None = None,
                   causal_r: dict | None = None, strong: dict | None = None,
                   concept_csv: str | None = None, alpha: float = 0.05, top_m: int = 3,
                   require_region: bool = True) -> dict:
    cols = _str_list(meas_cols)
    col = {c: i for i, c in enumerate(cols)}
    X = np.asarray(meas, dtype=np.float64)
    y = np.asarray(labels).reshape(-1).astype(np.int64)
    yhat = np.asarray(logits).argmax(axis=1).astype(np.int64)
    rules = rules_js["rules"]
    K = max(len(rules), int(np.asarray(logits).shape[1]))
    warnings = []
    rc_cols = [str(c) for c in rules_js.get("columns", [])]
    if rc_cols and rc_cols != cols:
        warnings.append("rules.json columns differ from test_feats meas_cols "
                        "(literals are matched by column name)")

    CM, CR = _Causal(causal_m), _Causal(causal_r)
    induced_with_region = bool(rules_js.get("params", {}).get("region_check", True))
    use_region = bool(require_region and CR.ok and induced_with_region)
    if require_region and not CR.ok:
        warnings.append("no causal_regions: region necessity not required for confirmation")
    if require_region and CR.ok and not induced_with_region:
        warnings.append("rules were induced without the region check: region necessity "
                        "not required for confirmation")
    layer_of = dict((strong or {}).get("layer_of", {}))

    # confusable fallback from TEST confusion (or network confusion on test)
    if causal_m is not None and "confusion" in causal_m:
        conf = np.asarray(causal_m["confusion"], dtype=np.float64)
    else:
        conf = np.zeros((K, K))
        np.add.at(conf, (np.clip(y, 0, K - 1), np.clip(yhat, 0, K - 1)), 1.0)
    conf_fallback = confusable_classes(conf, top_m)

    kb = None
    if concept_csv:
        if os.path.isfile(concept_csv):
            try:
                kb = knowledge_base(concept_csv, K)
            except Exception as e:  # noqa: BLE001
                warnings.append(f"concept CSV / concept map unusable ({e}); knowledge check skipped")
        else:
            warnings.append(f"concept CSV not found: {concept_csv}; knowledge check skipped")

    rows, S_cols, rule_cls, tprec, tsup = [], [], [], [], []
    findings = []
    for r in rules:
        k = int(r["class_idx"])
        lits = [[str(c), str(op), float(th)] for c, op, th in r["literals"]]
        missing = [c for c, _, _ in lits if c not in col]
        lits_ok = not missing
        if missing:
            warnings.append(f"class {k}: columns {missing} not in test meas_cols; rule skipped")
        L = r.get("learnable", {}) or {}
        flags = {f: bool(L.get(f, False)) for f in ("data_supported", "encodable", "consistent")}
        learnable = bool(L.get("learnable", all(flags.values())))
        ev_all = r.get("evidence", {}) or {}
        row = {"class_idx": k, "class_name": r.get("class_name", str(k)),
               "rule": rule_to_string(lits), "length": len(lits),
               "n_lt": sum(1 for _, op, _ in lits if op == "<"),
               "n_gt": sum(1 for _, op, _ in lits if op == ">"),
               "train_support": _nan(r.get("train_support")),
               "train_precision": _nan(r.get("train_precision")),
               **flags, "learnable": learnable}
        in_k = y == k
        row["n_test_class"] = int(in_k.sum())
        if lits and lits_ok:
            sat = satisfies(X, lits, col)
            n_sat = int(sat.sum())
            Ck = [int(j) for j in r.get("confusable",
                                         conf_fallback[k] if k < len(conf_fallback) else [])]
            # no confusable class: the stump used all other classes as negatives, so
            # discrimination falls back to plain precision instead of a trivial 1.0
            dom = np.isin(y, [k] + Ck) if Ck else np.ones_like(in_k)
            row.update({
                "n_satisfied": n_sat,
                "coverage": _ratio(int((sat & in_k).sum()), int(in_k.sum())),
                "precision": _ratio(int((sat & in_k).sum()), n_sat),
                "fidelity": _ratio(int((sat & (yhat == k)).sum()), n_sat),
                "discrimination": _ratio(int((sat & in_k).sum()), int((sat & dom).sum())),
                "confusable": " ".join(str(j) for j in Ck),
            })
            S_cols.append(sat)
            rule_cls.append(k)
            tprec.append(r.get("train_precision"))
            tsup.append(r.get("train_support"))
        else:
            row.update({"n_satisfied": 0, "coverage": float("nan"), "precision": float("nan"),
                        "fidelity": float("nan"), "discrimination": float("nan"),
                        "confusable": ""})

        # necessity on TEST (one check per measure of the rule)
        measures = list(dict.fromkeys(column_measure(c) for c, _, _ in lits))
        nec, reg_nec = [], []
        m_conf = bool(lits) and lits_ok and CM.ok
        r_conf = bool(lits) and lits_ok and CR.ok
        for m in measures:
            ev = ev_all.get(m, {}) or {}
            lay = layer_of.get(m, ev.get("layer"))
            v = CM.get(m, lay, k)
            if v is None:
                nec.append(f"{m}@?:n/a")
                m_conf = False
            else:
                m_conf = m_conf and _confirmed(v, alpha)
                nec.append(f"{m}@l{int(lay)}:ex={v[0]:+.3f},p={v[1]:.3g}")
            reg, rl = ev.get("region"), ev.get("region_layer")
            rv = CR.get(str(reg), rl, k) if reg else None
            if rv is None:
                reg_nec.append(f"{m}->{reg or '?'}@?:n/a")
                r_conf = False
            else:
                r_conf = r_conf and _confirmed(rv, alpha)
                reg_nec.append(f"{m}->{reg}@l{int(rl)}:ex={rv[0]:+.3f},p={rv[1]:.3g}")
        row["necessity"] = "; ".join(nec)
        row["measure_confirmed"] = bool(m_conf)
        row["region_necessity"] = "; ".join(reg_nec)
        row["region_confirmed"] = bool(r_conf)
        row["causally_confirmed"] = bool(m_conf and (r_conf if use_region else True))
        row["step5"] = bool(learnable and row["causally_confirmed"])

        # human knowledge (direction-aware)
        kb_res = {"agree": [], "disagree": [], "not-listed": [], "no-mapping": []}
        for c, op, th in lits:
            m = column_measure(c)
            v = literal_kb(m, op, k, kb)
            lit_s = literal_to_string([c, op, th])
            kb_res[v].append(lit_s)
            if v == "disagree":
                opp = kb["mapped"][m][_OPP[op]]
                have = sorted(set(opp) & kb["class_concepts"][k])
                same = kb["mapped"][m][op]
                findings.append({
                    "class_idx": k, "class_name": row["class_name"], "literal": lit_s,
                    "note": (f"rule needs {m} {op} θ (concepts [{', '.join(same) or '-'}]) but "
                             f"the CSV lists the opposite-direction concept(s) "
                             f"[{', '.join(have)}] for this class"),
                    "causally_confirmed": row["causally_confirmed"]})
        row["kb_agree"] = " | ".join(kb_res["agree"])
        row["kb_disagree"] = " | ".join(kb_res["disagree"])
        row["kb_not_listed"] = " | ".join(kb_res["not-listed"])
        row["kb_no_mapping"] = " | ".join(kb_res["no-mapping"])
        row["n_kb_agree"] = len(kb_res["agree"])
        row["n_kb_disagree"] = len(kb_res["disagree"])
        row["n_kb_not_listed"] = len(kb_res["not-listed"])
        row["n_kb_no_mapping"] = len(kb_res["no-mapping"])
        rows.append(row)

    # rule-based classifiers
    S = np.stack(S_cols, axis=1) if S_cols else np.zeros((len(y), 0), dtype=bool)
    clf_all = _clf_stats(rule_classifier(S, rule_cls, tprec, tsup), y, yhat)
    step5_k = {rw["class_idx"] for rw in rows if rw["step5"]}
    sel = [i for i, kk in enumerate(rule_cls) if kk in step5_k]
    clf_step5 = _clf_stats(rule_classifier(S[:, sel], [rule_cls[i] for i in sel],
                                           [tprec[i] for i in sel], [tsup[i] for i in sel]),
                           y, yhat)

    nonempty = [rw for rw in rows if rw["length"] > 0]

    def mean(key, rs):
        v = np.asarray([rw[key] for rw in rs], dtype=np.float64)
        return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")

    summary = {
        "n_classes": len(rows),
        "n_rules": len(nonempty),
        "n_learnable": sum(1 for rw in nonempty if rw["learnable"]),
        "n_measure_confirmed": sum(1 for rw in nonempty if rw["measure_confirmed"]),
        "n_region_confirmed": sum(1 for rw in nonempty if rw["region_confirmed"]),
        "n_causally_confirmed": sum(1 for rw in nonempty if rw["causally_confirmed"]),
        "n_step5": sum(1 for rw in nonempty if rw["step5"]),
        "mean_coverage": mean("coverage", nonempty),
        "mean_precision": mean("precision", nonempty),
        "mean_fidelity": mean("fidelity", nonempty),
        "mean_discrimination": mean("discrimination", nonempty),
        "mean_length": mean("length", nonempty),
        "kb_literals": {"agree": sum(rw["n_kb_agree"] for rw in rows),
                        "disagree": sum(rw["n_kb_disagree"] for rw in rows),
                        "not_listed": sum(rw["n_kb_not_listed"] for rw in rows),
                        "no_mapping": sum(rw["n_kb_no_mapping"] for rw in rows)},
        "classifier_all_rules": clf_all,
        "classifier_step5_rules": clf_step5,
        "alpha": alpha,
        "require_region": use_region,
        "concept_csv": concept_csv if kb is not None else None,
    }
    return {"rows": rows, "summary": summary, "findings": findings, "warnings": warnings}


# ----------------------------------------------------------------------------- outputs
CSV_COLS = ["class_idx", "class_name", "rule", "length", "n_lt", "n_gt", "n_test_class",
            "n_satisfied", "coverage", "precision", "fidelity", "discrimination", "confusable",
            "train_support", "train_precision", "data_supported", "encodable", "consistent",
            "learnable", "necessity", "measure_confirmed", "region_necessity",
            "region_confirmed", "causally_confirmed", "step5", "kb_agree", "kb_disagree",
            "kb_not_listed", "kb_no_mapping"]


def _fmt(x):
    if isinstance(x, float):
        return "-" if not np.isfinite(x) else f"{x:.3f}"
    return str(x)


def write_reports(res: dict, rules_js: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    rows, sm = res["rows"], res["summary"]

    with open(os.path.join(out_dir, "rule_report.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for rw in rows:
            w.writerow({c: (f"{rw[c]:.6g}" if isinstance(rw.get(c), float) else rw.get(c, ""))
                        for c in CSV_COLS})

    by_k = {int(r["class_idx"]): r for r in rules_js["rules"]}
    step5 = []
    for rw in rows:
        if not rw["step5"]:
            continue
        r = by_k[rw["class_idx"]]
        step5.append({
            "class_idx": rw["class_idx"], "class_name": rw["class_name"],
            "literals": r["literals"], "rule": rw["rule"], "evidence_val": r.get("evidence", {}),
            "test": {c: _f(rw[c]) if isinstance(rw[c], float) else rw[c] for c in
                     ("coverage", "precision", "fidelity", "discrimination", "necessity",
                      "region_necessity", "n_satisfied", "n_test_class")},
            "train_support": r.get("train_support"), "train_precision": r.get("train_precision"),
            "kb": {"agree": rw["kb_agree"], "disagree": rw["kb_disagree"],
                   "not_listed": rw["kb_not_listed"], "no_mapping": rw["kb_no_mapping"]},
        })
    crit = ("learnable (data_supported & encodable & consistent, from rules.json) AND causally "
            f"confirmed on TEST (every measure of the rule: excess > 0 and p_class < {sm['alpha']}"
            " at its layer" + (", and its VAL evidence region likewise at the recorded layer"
                               if sm["require_region"] else "") + ")")
    with open(os.path.join(out_dir, "learnable_rules.json"), "w", encoding="utf-8") as fh:
        json.dump({"criteria": crit,
                   "measures": sorted({column_measure(l[0]) for s in step5 for l in s["literals"]}),
                   "columns": sorted({l[0] for s in step5 for l in s["literals"]}),
                   "rules": step5}, fh, indent=2, ensure_ascii=False)

    def clf_line(name, c):
        return (f"| {name} | {_fmt(c['covered_frac'])} | {_fmt(c['accuracy'])} | "
                f"{_fmt(c['accuracy_on_covered'])} | {_fmt(c['global_fidelity'])} | "
                f"{_fmt(c['fidelity_on_covered'])} |")

    kl = sm["kb_literals"]
    L = [
        "# Rule evaluation on TEST", "",
        f"- classes: {sm['n_classes']}; non-empty rules: {sm['n_rules']}; "
        f"learnable: {sm['n_learnable']}; measure-necessity confirmed: "
        f"{sm['n_measure_confirmed']}; region-necessity confirmed: {sm['n_region_confirmed']}; "
        f"causally confirmed on TEST: {sm['n_causally_confirmed']}; "
        f"**step-5 rules (learnable ∧ confirmed): {sm['n_step5']}**",
        f"- mean over non-empty rules: coverage {_fmt(sm['mean_coverage'])}, precision "
        f"{_fmt(sm['mean_precision'])}, fidelity {_fmt(sm['mean_fidelity'])}, discrimination "
        f"{_fmt(sm['mean_discrimination'])}, length {_fmt(sm['mean_length'])}",
        f"- knowledge check (literals, direction-aware): agree {kl['agree']}, disagree "
        f"{kl['disagree']}, not-listed {kl['not_listed']}, no-mapping {kl['no_mapping']}"
        + ("" if sm["concept_csv"] else " (no concept CSV used)"),
        f"- network accuracy on TEST: {_fmt(sm['classifier_all_rules']['network_accuracy'])}", "",
        "## Rule-based classifier (highest train precision among satisfied rules; abstain = wrong)", "",
        "| rule set | covered | accuracy | acc. on covered | global fidelity | fidelity on covered |",
        "|---|---|---|---|---|---|",
        clf_line("all rules", sm["classifier_all_rules"]),
        clf_line("step-5 rules", sm["classifier_step5_rules"]), "",
        "## Learnable rules for step 5 (learnable ∧ causally confirmed on TEST)", "",
    ]
    s5 = [rw for rw in rows if rw["step5"]]
    if s5:
        L += ["| k | class | rule | cov | prec | fid | disc | necessity (TEST) | region (TEST) |",
              "|---|---|---|---|---|---|---|---|---|"]
        L += [f"| {rw['class_idx']} | {rw['class_name']} | {rw['rule']} | {_fmt(rw['coverage'])} | "
              f"{_fmt(rw['precision'])} | {_fmt(rw['fidelity'])} | {_fmt(rw['discrimination'])} | "
              f"{rw['necessity']} | {rw['region_necessity']} |" for rw in s5]
    else:
        L += ["(none)"]
    L += ["", "## Findings (rule literal vs concepts CSV, opposite direction)", ""]
    if res["findings"]:
        L += [f"- [{f['class_idx']}] {f['class_name']}: `{f['literal']}` — {f['note']}"
              f" (causally confirmed rule: {'yes' if f['causally_confirmed'] else 'no'})"
              for f in res["findings"]]
    else:
        L += ["(none)"]
    L += ["", "## All rules", "",
          "| k | class | rule | len | cov | prec | fid | disc | learnable | meas. nec. | "
          "region nec. | confirmed | KB agree/dis/not-listed/none |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    yn = {True: "Y", False: "n"}
    for rw in rows:
        L.append(f"| {rw['class_idx']} | {rw['class_name']} | {rw['rule']} | {rw['length']} | "
                 f"{_fmt(rw['coverage'])} | {_fmt(rw['precision'])} | {_fmt(rw['fidelity'])} | "
                 f"{_fmt(rw['discrimination'])} | {yn[rw['learnable']]} | "
                 f"{yn[rw['measure_confirmed']]} | {yn[rw['region_confirmed']]} | "
                 f"{yn[rw['causally_confirmed']]} | "
                 f"{rw['n_kb_agree']}/{rw['n_kb_disagree']}/{rw['n_kb_not_listed']}/"
                 f"{rw['n_kb_no_mapping']} |")
    if res["warnings"]:
        L += ["", "## Warnings", ""] + [f"- {w}" for w in res["warnings"]]
    with open(os.path.join(out_dir, "rule_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")

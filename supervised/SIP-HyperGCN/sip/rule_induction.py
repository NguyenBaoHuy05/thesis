"""Induce one interpretable conjunctive rule per class (pipeline v3, docs/PIPELINE_V3.md).

Rules are an OUTPUT of the intervention study. Thresholds appear only here, learned on
TRAIN-FIT. Inputs:

* ``causal_dir/causal_measures.npz`` and ``causal_dir/causal_regions.npz`` measured on VAL
  (unit_names, class_names, layers, excess, p_class, confusion, val_index, ...)
* ``strong_measures.json`` (only ``selected`` measures, each at its ``layer_of`` layer)
* ``train_feats.npz`` (``meas`` (N, D), ``meas_cols`` (D,), ``labels``) restricted to
  TRAIN-FIT rows (train rows not in ``val_index``). TEST is never touched here.

Algorithm for class k:

1. Candidate measures m (selected, at layer l_m = layer_of[m]):
   ``excess[l_m, m, k] >= min_excess`` and BH-FDR ``q < alpha`` (over the selected x K grid)
   and the per-class region cross-check: some region r of m (``MEASURE_REGIONS[m]``) has
   ``region excess[l, r, k] >= region_min_excess`` and ``region p_class[l, r, k] < region_alpha``
   at some l in ``region_layers`` (default 0, 3). The best such (r, l) is the evidence.
2. Literal per candidate by a decision stump on TRAIN-FIT: class k vs the union of its top-m
   confusable classes (symmetrised row-normalised VAL confusion). Over the measure's columns,
   both directions, thresholds = midpoints of ``n_quantiles`` quantiles (rounded to
   ``theta_decimals``); maximise balanced accuracy subject to class-k coverage
   ``>= stump_min_coverage``. Literal = [column, "<" | ">", theta].
3. Greedy: candidates in decreasing excess; a literal is kept iff it strictly raises
   TRAIN-FIT precision (all classes) and keeps class-k coverage ``>= min_coverage``;
   at most ``max_len`` literals.
4. Learnability flags: data_supported (each literal covers >= min_support of class k on
   VAL, i.e. on rows the threshold was not fit on), encodable (strong tier ``encoded``;
   holds by construction since only selected measures are used), consistent (no other class has the identical
   (column, op) set; Jaccard of measure sets with confusable classes < consistency_jaccard).
"""
from __future__ import annotations

import json
import os

import numpy as np

__all__ = [
    "DEFAULT_PARAMS",
    "load_npz",
    "load_strong",
    "load_measure_regions",
    "bh_adjust",
    "train_fit_mask",
    "confusable_classes",
    "column_measure",
    "measure_columns",
    "literal_mask",
    "satisfies",
    "fit_stump",
    "format_theta",
    "literal_to_string",
    "rule_to_string",
    "induce_rules",
    "write_rules_md",
]

DEFAULT_PARAMS = {
    "min_excess": 0.02,
    "alpha": 0.05,
    "fdr": True,
    "region_check": True,
    "region_layers": [0, 3],
    "region_min_excess": 0.01,
    "region_alpha": 0.05,
    "top_m_confusable": 3,
    "n_quantiles": 64,
    "theta_decimals": 3,
    "stump_min_coverage": 0.5,
    "min_coverage": 0.3,
    "max_len": 4,
    "min_support": 0.5,
    "consistency_jaccard": 0.8,
}

_OPS = ("<", ">")


# ----------------------------------------------------------------------------- io
def load_npz(path: str) -> dict:
    """npz -> plain dict (arrays loaded eagerly; 0-d arrays kept as arrays)."""
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def _str_list(a) -> list[str]:
    return [str(x) for x in np.asarray(a).reshape(-1)]


def load_strong(path: str) -> dict:
    """Reads strong_measures.json. Returns dict with ``selected`` list[str], ``layer_of``
    {name: int}, ``tiers`` {name: dict}, ``metrics`` {name: dict}, ``regions`` {name: list}."""
    with open(path, "r", encoding="utf-8") as fh:
        js = json.load(fh)
    ms = js.get("measures", []) or []
    tiers = {d["name"]: dict(d.get("tiers", {}) or {}) for d in ms if "name" in d}
    metrics = {d["name"]: dict(d.get("metrics", {}) or {}) for d in ms if "name" in d}
    regions = {d["name"]: [str(r) for r in (d.get("regions") or [])] for d in ms if "name" in d}
    layer_of = {str(k): int(v) for k, v in (js.get("layer_of", {}) or {}).items() if v is not None}
    for d in ms:  # fallback: per-measure "layer"
        if "name" in d and d["name"] not in layer_of and d.get("layer") is not None:
            layer_of[d["name"]] = int(d["layer"])
    selected = js.get("selected")
    if selected is None:  # derive: all tiers hold
        selected = [n for n, t in tiers.items() if t and all(bool(v) for v in t.values())]
    return {"selected": [str(s) for s in selected], "layer_of": layer_of, "tiers": tiers,
            "metrics": metrics, "regions": regions, "raw": js}


def load_measure_regions(strong: dict | None = None) -> tuple[dict, str]:
    """measure -> region names. ``sip.measures.MEASURE_REGIONS`` first, else the regions
    listed in strong_measures.json. Returns (mapping, source)."""
    try:
        from .measures import MEASURE_REGIONS
        return {str(k): [str(r) for r in v] for k, v in MEASURE_REGIONS.items()}, \
            "sip.measures.MEASURE_REGIONS"
    except Exception:  # noqa: BLE001
        return dict((strong or {}).get("regions", {})), "strong_measures.json regions"


# ----------------------------------------------------------------------------- stats
def bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values), same shape. NaN -> 1."""
    p = np.asarray(p, dtype=np.float64)
    flat = np.where(np.isfinite(p), p, 1.0).reshape(-1)
    n = flat.size
    if n == 0:
        return p.copy()
    order = np.argsort(flat, kind="mergesort")
    ranked = flat[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n, dtype=np.float64)
    out[order] = np.clip(q, 0.0, 1.0)
    return out.reshape(p.shape)


def train_fit_mask(n_train: int, val_index) -> np.ndarray:
    """(n_train,) bool: True for TRAIN-FIT rows (not in VAL)."""
    m = np.ones(int(n_train), dtype=bool)
    vi = np.asarray(val_index, dtype=np.int64).reshape(-1)
    if vi.size:
        if vi.min() < 0 or vi.max() >= n_train:
            raise ValueError(f"val_index out of range for {n_train} train rows")
        m[vi] = False
    return m


def confusable_classes(confusion: np.ndarray, m: int) -> list[list[int]]:
    """Top-m most confused classes per class from a (K, K) count confusion matrix
    (rows = true, cols = predicted), symmetrised after row normalisation. Only classes
    with a non-zero confusion are returned."""
    C = np.asarray(confusion, dtype=np.float64)
    rs = C.sum(axis=1, keepdims=True)
    R = np.divide(C, rs, out=np.zeros_like(C), where=rs > 0)
    S = R + R.T
    np.fill_diagonal(S, 0.0)
    out = []
    for k in range(S.shape[0]):
        order = np.argsort(-S[k], kind="mergesort")
        out.append([int(j) for j in order[:m] if S[k, j] > 0])
    return out


# ----------------------------------------------------------------------------- literals
def column_measure(col: str) -> str:
    """'hand_head_dist:min' -> 'hand_head_dist'."""
    return str(col).split(":", 1)[0]


def measure_columns(cols) -> dict[str, list[int]]:
    """measure -> list of column indices into ``cols`` (``measure:stat`` names)."""
    out: dict[str, list[int]] = {}
    for i, c in enumerate(cols):
        out.setdefault(column_measure(c), []).append(i)
    return out


def literal_mask(X: np.ndarray, lit, col: dict) -> np.ndarray:
    """(N,) bool for one literal [column, op, theta]; non-finite values never satisfy it."""
    c, op, th = lit
    x = np.asarray(X[:, col[str(c)]], dtype=np.float64)
    with np.errstate(invalid="ignore"):
        if op == "<":
            return x < float(th)
        if op == ">":
            return x > float(th)
    raise ValueError(f"unknown literal operator {op!r}")


def satisfies(X: np.ndarray, literals, col: dict) -> np.ndarray:
    """(N,) bool: rows satisfying the conjunction. Empty rule -> all True."""
    sat = np.ones(X.shape[0], dtype=bool)
    for lit in literals:
        sat &= literal_mask(X, lit, col)
    return sat


def fit_stump(Xp: np.ndarray, Xn: np.ndarray, n_quantiles: int = 64, min_coverage: float = 0.5,
              decimals: int = 3) -> dict | None:
    """Decision stump separating positives ``Xp`` (n_p, c) from negatives ``Xn`` (n_n, c).

    Over columns j and ops "<" / ">", thresholds are the midpoints of consecutive unique
    quantiles (``n_quantiles`` + 1 levels) of the pooled finite values, rounded to
    ``decimals``. Maximises balanced accuracy = (TPR + TNR) / 2 subject to TPR (class-k
    coverage) >= ``min_coverage``; ties -> higher TPR, then first column / "<".
    Non-finite values never satisfy a literal. Returns None if no threshold qualifies, else
    {"j", "op", "theta", "bal_acc", "coverage", "neg_rate"} (neg_rate = FPR on negatives)."""
    Xp = np.asarray(Xp, dtype=np.float64)
    Xn = np.asarray(Xn, dtype=np.float64)
    n_pos, n_neg = Xp.shape[0], Xn.shape[0]
    if n_pos == 0 or n_neg == 0:
        return None
    levels = np.linspace(0.0, 1.0, int(n_quantiles) + 1)
    best, best_key = None, None
    for j in range(Xp.shape[1]):
        p = np.sort(Xp[:, j][np.isfinite(Xp[:, j])])
        n = np.sort(Xn[:, j][np.isfinite(Xn[:, j])])
        if p.size == 0 or n.size == 0:
            continue
        q = np.unique(np.quantile(np.concatenate([p, n]), levels))
        if q.size < 2:
            continue
        th = np.unique(np.round(0.5 * (q[:-1] + q[1:]), int(decimals)))
        tp = {"<": np.searchsorted(p, th, side="left"),
              ">": p.size - np.searchsorted(p, th, side="right")}
        fp = {"<": np.searchsorted(n, th, side="left"),
              ">": n.size - np.searchsorted(n, th, side="right")}
        for op in _OPS:
            tpr = tp[op] / n_pos
            fpr = fp[op] / n_neg
            bal = 0.5 * (tpr + 1.0 - fpr)
            ok = tpr >= min_coverage - 1e-12
            if not ok.any():
                continue
            idx = np.flatnonzero(ok)
            # lexicographic (bal, tpr) maximum among qualifying thresholds
            bb = bal[idx]
            top = idx[bb >= bb.max() - 1e-12]
            i = int(top[np.argmax(tpr[top])])
            key = (float(bal[i]), float(tpr[i]))
            if (best_key is None or key[0] > best_key[0] + 1e-12
                    or (abs(key[0] - best_key[0]) <= 1e-12 and key[1] > best_key[1] + 1e-12)):
                best_key = key
                best = {"j": int(j), "op": op, "theta": round(float(th[i]), int(decimals)),
                        "bal_acc": float(bal[i]), "coverage": float(tpr[i]),
                        "neg_rate": float(fpr[i])}
    return best


def format_theta(t) -> str:
    s = f"{float(t):.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def literal_to_string(lit) -> str:
    c, op, th = lit
    return f"{c} {op} {format_theta(th)}"


def rule_to_string(literals) -> str:
    if not literals:
        return "(no rule)"
    return " ∧ ".join(literal_to_string(l) for l in literals)


def _prec_cov(sat: np.ndarray, in_k: np.ndarray):
    n_sat = int(sat.sum())
    n_k = int(in_k.sum())
    tp = int((sat & in_k).sum())
    prec = tp / n_sat if n_sat else 0.0
    cov = tp / n_k if n_k else 0.0
    return prec, cov


def _f(x):
    """json-safe float (NaN -> None)."""
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


# ----------------------------------------------------------------------------- induction
def induce_rules(causal_m: dict, causal_r: dict | None, strong: dict, meas: np.ndarray,
                 meas_cols, labels: np.ndarray, fit_mask: np.ndarray, class_names=None,
                 measure_regions: dict | None = None, net_pred: np.ndarray | None = None,
                 params: dict | None = None) -> dict:
    """Returns the rules.json dict (spec format + params + diagnostics).

    causal_m / causal_r: dicts from causal_measures.npz / causal_regions.npz (VAL).
    strong: from ``load_strong``. meas / labels: full train_feats arrays; ``fit_mask``
    selects TRAIN-FIT rows. measure_regions: measure -> region names
    (default ``load_measure_regions(strong)``). net_pred: network predictions on train rows
    (only used if ``confusion`` is absent)."""
    P_ = dict(DEFAULT_PARAMS)
    if params:
        P_.update({k: v for k, v in params.items() if v is not None})
    warnings: list[str] = []

    # ---- measure causal arrays (VAL)
    m_names = _str_list(causal_m["unit_names"])
    m_col = {n: i for i, n in enumerate(m_names)}
    layers = [int(v) for v in np.asarray(causal_m["layers"]).reshape(-1)]
    excess = np.asarray(causal_m["excess"], dtype=np.float64)        # (L, U, K)
    p_class = np.asarray(causal_m["p_class"], dtype=np.float64)      # (L, U, K)
    K = excess.shape[2]
    if class_names is None:
        class_names = (_str_list(causal_m["class_names"]) if "class_names" in causal_m
                       else [f"class_{k}" for k in range(K)])

    # ---- region causal arrays (VAL)
    region_check = bool(P_["region_check"])
    r_col, r_lidx, r_layers, r_ex, r_p = {}, [], [], None, None
    if region_check:
        if causal_r is None:
            raise ValueError("region check requested but causal_regions is missing")
        r_col = {n: i for i, n in enumerate(_str_list(causal_r["unit_names"]))}
        r_layers = [int(v) for v in np.asarray(causal_r["layers"]).reshape(-1)]
        r_ex = np.asarray(causal_r["excess"], dtype=np.float64)
        r_p = np.asarray(causal_r["p_class"], dtype=np.float64)
        for l in P_["region_layers"]:
            if int(l) in r_layers:
                r_lidx.append((int(l), r_layers.index(int(l))))
            else:
                warnings.append(f"region layer {l} not among causal_regions layers {r_layers}")
        if not r_lidx:
            warnings.append("no usable region layer: every candidate fails the region check")
    if measure_regions is None:
        measure_regions, mr_src = load_measure_regions(strong)
    else:
        mr_src = "argument"

    # ---- TRAIN-FIT data
    cols = _str_list(meas_cols)
    c_idx = {c: i for i, c in enumerate(cols)}
    mcols = measure_columns(cols)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    X = np.asarray(meas, dtype=np.float64)[fit_mask]
    y = np.asarray(labels).reshape(-1)[fit_mask].astype(np.int64)
    # VAL rows: held out from the stump fit, used only for the data_supported flag
    Xv = np.asarray(meas, dtype=np.float64)[~fit_mask]
    yv = np.asarray(labels).reshape(-1)[~fit_mask].astype(np.int64)

    # ---- usable measures: selected, in causal + feats, layer among causal layers
    usable, layer_idx = [], {}
    for name in strong["selected"]:
        if name not in m_col:
            warnings.append(f"{name}: not in causal_measures unit_names, skipped")
            continue
        if name not in mcols:
            warnings.append(f"{name}: no column in train_feats meas_cols, skipped")
            continue
        lay = strong["layer_of"].get(name)
        if lay is None or int(lay) not in layers:
            warnings.append(f"{name}: layer_of={lay} not among causal layers {layers}, skipped")
            continue
        if region_check and not [r for r in measure_regions.get(name, []) if r in r_col]:
            warnings.append(f"{name}: no region (from {mr_src}) in causal_regions; "
                            "it cannot pass the region check")
        usable.append(name)
        layer_idx[name] = layers.index(int(lay))

    # ---- per (usable measure, class): excess, p, q at its layer
    U = len(usable)
    ex_u = np.zeros((U, K))
    p_u = np.ones((U, K))
    for i, name in enumerate(usable):
        ex_u[i] = excess[layer_idx[name], m_col[name]]
        p_u[i] = p_class[layer_idx[name], m_col[name]]
    ex_u = np.where(np.isfinite(ex_u), ex_u, -np.inf)
    q_u = bh_adjust(p_u) if (P_["fdr"] and U) else np.where(np.isfinite(p_u), p_u, 1.0)

    def region_evidence(name: str, k: int):
        """Best (region, layer, excess, p) passing the per-class region check, or None."""
        best = None
        for r in measure_regions.get(name, []):
            if r not in r_col:
                continue
            for l, li in r_lidx:
                ex = float(r_ex[li, r_col[r], k])
                pv = float(r_p[li, r_col[r], k])
                if (np.isfinite(ex) and np.isfinite(pv) and ex >= P_["region_min_excess"]
                        and pv < P_["region_alpha"]):
                    if best is None or ex > best[2]:
                        best = (r, l, ex, pv)
        return best

    # ---- confusable classes (VAL confusion; fallback TRAIN-FIT network confusion)
    if "confusion" in causal_m:
        confusion = np.asarray(causal_m["confusion"], dtype=np.float64)
        conf_src = "causal_measures.confusion (VAL)"
    elif net_pred is not None:
        npred = np.asarray(net_pred).reshape(-1)[fit_mask].astype(np.int64)
        confusion = np.zeros((K, K))
        np.add.at(confusion, (np.clip(y, 0, K - 1), np.clip(npred, 0, K - 1)), 1.0)
        conf_src = "TRAIN-FIT network predictions (fallback)"
        warnings.append("confusion missing in causal_measures; used TRAIN-FIT network confusion")
    else:
        confusion = np.zeros((K, K))
        conf_src = "none (stump negatives = all other classes)"
        warnings.append("no confusion available: stumps use all other classes as negatives")
    conf_cls = confusable_classes(confusion, int(P_["top_m_confusable"]))

    encoded = {n: bool(strong["tiers"].get(n, {}).get("encoded", False)) for n in usable}

    rules = []
    for k in range(K):
        in_k = y == k
        n_k = int(in_k.sum())
        Ck = conf_cls[k] if k < len(conf_cls) else []
        neg = np.isin(y, Ck) if Ck else np.zeros(len(y), dtype=bool)
        neg_src = "confusable"
        if not neg.any():
            neg = ~in_k
            neg_src = "all other classes"
        diag = {"n_excess_fdr": 0, "n_region_ok": 0, "n_stump_ok": 0, "rejected": {}}

        # 1. candidates
        cands = []
        for i, name in enumerate(usable):
            if not (ex_u[i, k] >= P_["min_excess"] and q_u[i, k] < P_["alpha"]):
                continue
            diag["n_excess_fdr"] += 1
            reg = region_evidence(name, k) if region_check else None
            if region_check and reg is None:
                diag["rejected"][name] = "region check"
                continue
            diag["n_region_ok"] += 1
            cands.append((ex_u[i, k], i, name, reg))
        cands.sort(key=lambda t: (-t[0], t[1]))

        # 2-3. stump literal + greedy
        lits, evidence, lit_cov = [], {}, []
        sat = np.ones(len(y), dtype=bool)
        cur_prec, _ = _prec_cov(sat, in_k)
        for ex, i, name, reg in cands:
            if len(lits) >= int(P_["max_len"]):
                diag["rejected"][name] = "max_len reached"
                continue
            jj = mcols[name]
            st = fit_stump(X[in_k][:, jj], X[neg][:, jj], n_quantiles=int(P_["n_quantiles"]),
                           min_coverage=float(P_["stump_min_coverage"]),
                           decimals=int(P_["theta_decimals"])) if n_k else None
            if st is None:
                diag["rejected"][name] = "no stump with class coverage >= " \
                                         f"{P_['stump_min_coverage']}"
                continue
            diag["n_stump_ok"] += 1
            lit =[cols[jj[st["j"]]], st["op"], st["theta"]]
            lm = literal_mask(X, lit, c_idx)
            new = sat & lm
            prec, cov = _prec_cov(new, in_k)
            if not (prec > cur_prec + 1e-12 and cov >= P_["min_coverage"]):
                diag["rejected"][name] = (f"greedy (precision {prec:.3f} vs {cur_prec:.3f}, "
                                          f"coverage {cov:.3f})")
                continue
            lits.append(lit)
            lit_cov.append(_prec_cov(lm, in_k)[1])
            sat, cur_prec = new, prec
            ev = {"layer": layers[layer_idx[name]], "excess": _f(ex), "q_class": _f(q_u[i, k]),
                  "p_class": _f(p_u[i, k])}
            if reg is not None:
                ev.update({"region": reg[0], "region_layer": int(reg[1]),
                           "region_excess": _f(reg[2]), "region_p": _f(reg[3])})
            else:
                ev.update({"region": None, "region_layer": None, "region_excess": None,
                           "region_p": None})
            ev.update({"column": lit[0], "stump_bal_acc": _f(st["bal_acc"]),
                       "stump_coverage": _f(st["coverage"]),
                       "stump_neg_rate": _f(st["neg_rate"]),
                       "literal_coverage": _f(lit_cov[-1])})
            evidence[name] = ev

        prec, cov = _prec_cov(sat, in_k)
        n_sat = int(sat.sum())
        # data support is measured on VAL: on TRAIN-FIT it would equal the stump's own
        # coverage constraint and could never fail
        in_kv = yv == k
        val_cov = [_prec_cov(literal_mask(Xv, l, c_idx), in_kv)[1] for l in lits] \
            if (lits and in_kv.any()) else []
        if lits:
            data_sup = bool(val_cov) and all(c >= P_["min_support"] - 1e-12 for c in val_cov)
            enc = all(encoded.get(column_measure(l[0]), False) for l in lits)
        else:
            data_sup = enc = False
        rules.append({
            "class_idx": int(k), "class_name": str(class_names[k]),
            "literals": lits,
            "evidence": evidence,
            "train_support": _f(cov) if (lits and n_k) else None,
            "train_precision": _f(prec) if (lits and n_sat) else None,
            "learnable": {"data_supported": bool(data_sup), "encodable": bool(enc),
                          "consistent": False, "learnable": False},
            "confusable": [int(j) for j in Ck],
            "stump_negatives": neg_src,
            "train_n_class": n_k, "train_n_satisfied": n_sat if lits else None,
            "train_class_prior": _f(n_k / len(y)) if len(y) else None,
            "val_literal_coverage": [_f(c) for c in val_cov],
            "diagnostics": diag,
        })

    # ---- consistency (needs all rules)
    lit_sets = [frozenset((str(c), str(op)) for c, op, _ in r["literals"]) for r in rules]
    m_sets = [frozenset(column_measure(c) for c, _, _ in r["literals"]) for r in rules]
    for k, r in enumerate(rules):
        if not lit_sets[k]:
            r["learnable"]["consistent"] = False
            r["consistency_note"] = "empty rule"
            continue
        dup = [j for j in range(K) if j != k and lit_sets[j] == lit_sets[k]]
        jac = {}
        for j in r["confusable"]:
            u = m_sets[k] | m_sets[j]
            jac[j] = len(m_sets[k] & m_sets[j]) / len(u) if u else 0.0
        bad_j = [j for j, v in jac.items() if v >= P_["consistency_jaccard"]]
        r["learnable"]["consistent"] = not dup and not bad_j
        r["max_jaccard_confusable"] = _f(max(jac.values())) if jac else 0.0
        if dup or bad_j:
            r["consistency_note"] = ((f"identical (column, op) set to {dup}" if dup else "") +
                                     (f" measure jaccard>={P_['consistency_jaccard']} with {bad_j}"
                                      if bad_j else "")).strip()
    for r in rules:
        r["learnable"]["learnable"] = bool(all(r["learnable"][f] for f in
                                               ("data_supported", "encodable", "consistent")))

    return {
        "columns": cols,
        "rules": rules,
        "params": {**P_, "region_layers": [int(l) for l in P_["region_layers"]],
                   "fdr_grid": f"{U} selected measures x {K} classes",
                   "confusion_source": conf_src, "measure_regions_source": mr_src,
                   "data_rows": "TRAIN-FIT (train minus val_index)",
                   "n_train_fit": int(len(y))},
        "measures": usable,
        "class_names": [str(c) for c in class_names],
        "layers": layers,
        "warnings": warnings,
    }


# ----------------------------------------------------------------------------- report
def write_rules_md(rules_js: dict, path: str) -> None:
    P_ = rules_js.get("params", {})
    rules = rules_js["rules"]
    n_nonempty = sum(1 for r in rules if r["literals"])
    n_learn = sum(1 for r in rules if r["learnable"].get("learnable"))
    lines = [
        "# Induced rules (VAL causal matrices + TRAIN-FIT thresholds)", "",
        f"- usable strong measures: {len(rules_js.get('measures', []))} "
        f"({', '.join(rules_js.get('measures', [])) or 'none'})",
        f"- classes with a rule: {n_nonempty}/{len(rules)}; learnable: {n_learn}",
        "- params: " + ", ".join(f"{k}={v}" for k, v in P_.items()), "",
    ]
    if rules_js.get("warnings"):
        lines += ["Warnings:", ""] + [f"- {w}" for w in rules_js["warnings"]] + [""]

    def fmt(x):
        return "-" if x is None else f"{x:.3f}"

    def ev_str(r):
        out = []
        for m, e in r.get("evidence", {}).items():
            s = f"{m}@l{e.get('layer')} ex={fmt(e.get('excess'))} q={fmt(e.get('q_class'))}"
            if e.get("region"):
                s += (f"; {e['region']}@l{e.get('region_layer')} ex={fmt(e.get('region_excess'))}"
                      f" p={fmt(e.get('region_p'))}")
            out.append(s)
        return "<br>".join(out) or "-"

    lines += ["| k | class | rule | len | support | precision | evidence (VAL) | data | enc | "
              "cons | learnable |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    yn = {True: "Y", False: "n"}
    for r in rules:
        L = r["learnable"]
        lines.append(
            f"| {r['class_idx']} | {r['class_name']} | {rule_to_string(r['literals'])} | "
            f"{len(r['literals'])} | {fmt(r['train_support'])} | {fmt(r['train_precision'])} | "
            f"{ev_str(r)} | {yn[L['data_supported']]} | {yn[L['encodable']]} | "
            f"{yn[L['consistent']]} | {yn[L['learnable']]} |")
    lines += ["", "support = fraction of class-k TRAIN-FIT samples satisfying the rule; "
              "precision = fraction of satisfying TRAIN-FIT samples that are class k. "
              "Thresholds are in measure units (distances / spine length L, angles in degrees)."]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

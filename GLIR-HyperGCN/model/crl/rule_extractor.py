"""RuleExtractor: Trích xuất và đánh giá luật DNF sau khi huấn luyện.

Module này thực hiện post-hoc extraction của các luật DNF từ mô hình CLIR đã huấn luyện:
1. Binarize các trọng số theo ngưỡng tau:
   - w_and1 = (sigmoid(theta_and1) >= tau_and)   [L1 AND rules]
   - w_and2 = (sigmoid(theta_and2) >= tau_and2)  [L2a AND2 rules]
   - w_or1  = (sigmoid(theta_or1)  >= tau_or)    [L2b OR1 rules]
   - W_action: lọc các trọng số dương > tau_act (excitatory clauses)

2. Executable DNF Engine:
   - Đánh giá luật dạng boolean/nhị phân trên từng mẫu dữ liệu
   - Kiểm tra L1 conjunction rules (tất cả literal thành phần = 1)
   - Kiểm tra L2a compound rules (tất cả L1 thành phần = 1)
   - Kiểm tra L2b alternative rules (ít nhất 1 L1 thành phần = 1)
   - Tính toán điểm kích hoạt lớp từ các mệnh đề DNF dương

3. Chỉ số đánh giá theo kế hoạch (Mục 10 — CLIR plan):
   - Rule Coverage: Tỷ lệ mẫu được bao phủ bởi ít nhất 1 luật DNF
   - Rule Precision: Tỷ lệ mẫu thỏa mãn luật thực sự thuộc về class đó
   - Model-Rule Fidelity: Tỷ lệ tương đồng giữa dự đoán của mạng nơ-ron và luật logic tượng trưng
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .conjunction_layer import _DEFAULT_N_CONCEPTS, get_part_groups


_DEFAULT_CONCEPT_NAMES = [f"concept_{i}" for i in range(_DEFAULT_N_CONCEPTS)]


class RuleExtractor:
    """Trình trích xuất và thực thi luật DNF từ CRLReasoner đã huấn luyện.

    Args:
        reasoner:       Instance của CRLReasoner (sau khi load checkpoint).
        concept_names:  Danh sách tên concepts.
        action_names:   Danh sách tên actions.
        tau_and:        Ngưỡng binarize cho L1 AND weights (mặc định 0.5).
        tau_and2:       Ngưỡng binarize cho L2a AND2 weights (mặc định 0.5).
        tau_or:         Ngưỡng binarize cho L2b OR1 weights (mặc định 0.5).
        tau_act:        Ngưỡng trọng số dương ActionHead (mặc định 0.05).
        top_k_l2:       Số L2 rules tối đa cho mỗi action (mặc định 5).
        top_k_l1:       Số L1 rules tối đa trong mỗi L2 rule (mặc định 5).
        top_k_literals: Số literals tối đa trong mỗi L1 rule (mặc định 6).
    """

    def __init__(
        self,
        reasoner,
        concept_names:  Optional[List[str]] = None,
        action_names:   Optional[List[str]] = None,
        tau_and:        float = 0.5,
        tau_and2:       float = 0.5,
        tau_or:         float = 0.5,
        tau_act:        float = 0.05,
        top_k_l2:       int = 5,
        top_k_l1:       int = 5,
        top_k_literals: int = 6,
    ):
        self.reasoner       = reasoner
        self.n_concepts     = reasoner.n_concepts
        self.n_actions      = reasoner.n_actions
        self.concept_names  = concept_names or _DEFAULT_CONCEPT_NAMES[:self.n_concepts]
        self.action_names   = action_names  or [f"action_{i}" for i in range(self.n_actions)]
        self.tau_and        = float(tau_and)
        self.tau_and2       = float(tau_and2)
        self.tau_or         = float(tau_or)
        self.tau_act        = float(tau_act)
        self.top_k_l2       = int(top_k_l2)
        self.top_k_l1       = int(top_k_l1)
        self.top_k_literals = int(top_k_literals)

    # ─── Lấy trọng số binarized (giữ nguyên device hiện tại) ───────────────────

    @torch.no_grad()
    def _get_w_and1(self) -> torch.Tensor:
        """W_and1 binarized [n_literals, r1]."""
        w = torch.sigmoid(self.reasoner.l1_and.weight_logit)
        if self.reasoner.l1_and.group_mask is not None:
            w = w * self.reasoner.l1_and.group_mask
        return (w >= self.tau_and).float()

    @torch.no_grad()
    def _get_w_and2(self) -> torch.Tensor:
        """W_and2 binarized [r1, r2a]."""
        w = torch.sigmoid(self.reasoner.l2a_and2.weight_logit)
        return (w >= self.tau_and2).float()

    @torch.no_grad()
    def _get_w_or1(self) -> torch.Tensor:
        """W_or1 binarized [r1, r2b]."""
        w = torch.sigmoid(self.reasoner.l2b_or1.weight_logit)
        return (w >= self.tau_or).float()

    @torch.no_grad()
    def _get_action_weights(self) -> torch.Tensor:
        """ActionHead weights [r2a + r2b, n_actions]."""
        return self.reasoner.action_head.weight.T

    # ─── Chuyển đổi literal index → tên concept ────────────────────────────

    def _literal_to_name(self, lit_idx: int) -> Tuple[str, bool]:
        n = self.n_concepts
        if lit_idx < n:
            cname = self.concept_names[lit_idx] if lit_idx < len(self.concept_names) else f"c_{lit_idx}"
            return cname, True
        else:
            base_idx = lit_idx - n
            cname = self.concept_names[base_idx] if base_idx < len(self.concept_names) else f"c_{base_idx}"
            return cname, False

    # ─── Trích xuất một L1 rule ─────────────────────────────────────────────

    def _extract_l1_rule(self, w_and1: torch.Tensor, l1_rule_idx: int) -> List[Tuple[str, bool]]:
        active_lits = w_and1[:, l1_rule_idx].nonzero(as_tuple=True)[0].tolist()
        w_col = torch.sigmoid(self.reasoner.l1_and.weight_logit[:, l1_rule_idx])
        if self.reasoner.l1_and.group_mask is not None:
            w_col = w_col * self.reasoner.l1_and.group_mask[:, l1_rule_idx]
        sorted_lits = sorted(active_lits, key=lambda i: -w_col[i].item())
        sorted_lits = sorted_lits[:self.top_k_literals]
        return [self._literal_to_name(i) for i in sorted_lits]

    # ─── Trích xuất rules cho một action ────────────────────────────────────

    def extract_rules_for_action(self, action_idx: int) -> Dict:
        """Trích xuất các mệnh đề DNF khẳng định (w > tau_act) cho action_idx."""
        w_and1 = self._get_w_and1()
        w_and2 = self._get_w_and2()
        w_or1  = self._get_w_or1()
        W_act  = self._get_action_weights()

        r2a = self.reasoner.r2a
        act_weights = W_act[:, action_idx]
        and2_weights = act_weights[:r2a]
        or1_weights  = act_weights[r2a:]

        clauses = []

        # ── AND2 (L2a) — chỉ lấy trọng số dương khẳng định
        top_and2_idx = and2_weights.argsort(descending=True)[:self.top_k_l2].tolist()
        for l2_idx in top_and2_idx:
            w_val = and2_weights[l2_idx].item()
            if w_val < self.tau_act:
                continue
            l1_idx_list = w_and2[:, l2_idx].nonzero(as_tuple=True)[0].tolist()[:self.top_k_l1]
            l1_components = []
            for l1_idx in l1_idx_list:
                lits = self._extract_l1_rule(w_and1, l1_idx)
                if lits:
                    l1_components.append({"l1_rule_idx": l1_idx, "literals": lits})
            if l1_components:
                clauses.append({
                    "type": "and2",
                    "l2_rule_idx": l2_idx,
                    "l1_components": l1_components,
                    "raw_weight": w_val,
                })

        # ── OR1 (L2b) — chỉ lấy trọng số dương khẳng định
        top_or1_idx = or1_weights.argsort(descending=True)[:self.top_k_l2].tolist()
        for l2_idx in top_or1_idx:
            w_val = or1_weights[l2_idx].item()
            if w_val < self.tau_act:
                continue
            l1_idx_list = w_or1[:, l2_idx].nonzero(as_tuple=True)[0].tolist()[:self.top_k_l1]
            l1_components = []
            for l1_idx in l1_idx_list:
                lits = self._extract_l1_rule(w_and1, l1_idx)
                if lits:
                    l1_components.append({"l1_rule_idx": l1_idx, "literals": lits})
            if l1_components:
                clauses.append({
                    "type": "or1",
                    "l2_rule_idx": l2_idx,
                    "l1_components": l1_components,
                    "raw_weight": w_val,
                })

        action_name = self.action_names[action_idx] if action_idx < len(self.action_names) else f"action_{action_idx}"
        return {
            "action_name": action_name,
            "action_idx": action_idx,
            "clauses": clauses,
        }

    def extract_all_rules(self) -> List[Dict]:
        """Trích xuất rules cho tất cả actions và lưu cache."""
        rules = [self.extract_rules_for_action(i) for i in range(self.n_actions)]
        self._cached_rules = rules
        return rules

    def get_or_extract_rules(self) -> List[Dict]:
        """Lấy rules đã trích xuất từ cache hoặc trích xuất mới."""
        if not hasattr(self, "_cached_rules") or self._cached_rules is None:
            self._cached_rules = self.extract_all_rules()
        return self._cached_rules

    # ─── Executable DNF Evaluation Engine ─────────────────────────────────────

    @torch.no_grad()
    def evaluate_dnf(
        self,
        z: torch.Tensor,
        rules: Optional[List[Dict]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Thực thi chính xác tập luật DNF đã trích xuất (khớp 100% với text/JSON).

        Đánh giá biểu thức boolean của các mệnh đề sau khi đã áp dụng top_k_l2,
        top_k_l1, và top_k_literals, đảm bảo metrics (Coverage, Precision, Fidelity)
        đo lường đúng mô hình DNF thực tế được xuất ra.

        Args:
            z:     Concept logits [B, n_concepts]
            rules: Danh sách rule dicts cho các actions (None = dùng get_or_extract_rules())

        Returns:
            dnf_preds:    [B] predicted class index (-1 nếu không luật nào khớp)
            dnf_scores:   [B, n_actions] activation scores từ các luật DNF
            covered_mask: [B] bool tensor (True nếu có ít nhất 1 luật kích hoạt)
        """
        if rules is None:
            rules = self.get_or_extract_rules()

        device = z.device
        B = z.shape[0]

        # 1. Binarize literals theo ngưỡng z > 0
        hard_pos = (z > 0.0)
        hard_neg = ~hard_pos
        l_bin = torch.cat([hard_pos, hard_neg], dim=-1)  # [B, 2*C] bool tensor

        concept_to_idx = {name: i for i, name in enumerate(self.concept_names)}
        dnf_scores = torch.zeros(B, self.n_actions, device=device)

        # 2. Đánh giá chính xác từng mệnh đề DNF đã trích xuất
        for rule_entry in rules:
            act_idx = rule_entry["action_idx"]
            for clause in rule_entry["clauses"]:
                clause_type = clause["type"]
                w = float(clause["raw_weight"])

                comp_results = []
                for comp in clause["l1_components"]:
                    lits = comp["literals"]
                    if not lits:
                        continue
                    lit_indices = []
                    for cname, is_pos in lits:
                        c_idx = concept_to_idx.get(cname, None)
                        if c_idx is not None and c_idx < self.n_concepts:
                            lit_indices.append(c_idx if is_pos else (c_idx + self.n_concepts))

                    if lit_indices:
                        # L1 rule active iff tất cả literal thành phần thỏa mãn
                        comp_active = l_bin[:, lit_indices].all(dim=-1)  # [B] bool
                        comp_results.append(comp_active)

                if not comp_results:
                    continue

                comp_tensor = torch.stack(comp_results, dim=1)  # [B, num_comps] bool
                if clause_type == "and2":
                    # AND2: Tất cả component L1 rules phải thỏa mãn
                    clause_active = comp_tensor.all(dim=1)  # [B] bool
                elif clause_type == "or1":
                    # OR1: Ít nhất 1 component L1 rule thỏa mãn
                    clause_active = comp_tensor.any(dim=1)  # [B] bool
                else:
                    clause_active = comp_tensor.all(dim=1)

                dnf_scores[:, act_idx] += w * clause_active.float()

        # 3. Mẫu được bao phủ nếu có ít nhất 1 class nhận score > 0
        max_scores, best_class = dnf_scores.max(dim=-1)
        covered_mask = max_scores > 0.0

        dnf_preds = torch.where(covered_mask, best_class, torch.tensor(-1, device=device))

        return dnf_preds, dnf_scores, covered_mask

    # ─── Đánh giá Fidelity, Coverage, Precision theo chuẩn kế hoạch ───────────

    @torch.no_grad()
    def evaluate_fidelity(
        self,
        model,
        data_loader,
        device: str = "cpu",
        n_batches: int = 50,
    ) -> Dict[str, float]:
        """Tính toán: Rule Precision, Rule Coverage, và Model-Rule Fidelity.

        Returns:
            {
                "coverage": float,   # Tỷ lệ mẫu được ít nhất 1 luật DNF bao phủ
                "precision": float,  # Độ chính xác của các dự đoán luật DNF (trên tập covered)
                "fidelity": float,   # Tỷ lệ dự đoán NN trùng khớp với luật DNF executable
                "total_samples": int,
                "covered_samples": int,
            }
        """
        model.eval()
        rules = self.get_or_extract_rules()
        total_samples = 0
        covered_samples = 0
        rule_correct = 0
        model_rule_agree = 0

        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= n_batches:
                break
            if len(batch) >= 5:
                data, index_t, label, index, _ = batch[:5]
            else:
                data, index_t, label, index = batch[:4]

            data    = data.float().to(device)
            index_t = index_t.float().to(device)
            label   = label.long().to(device)

            output = model(data, index_t=index_t)
            if isinstance(output, dict):
                nn_preds = output["action_logits"].argmax(dim=-1)
                z_final = output["concept_logits"]
            else:
                nn_preds = output.argmax(dim=-1)
                z_final = None

            if z_final is None:
                continue

            # Đánh giá chính xác tập luật DNF đã trích xuất
            dnf_preds, _, covered_mask = self.evaluate_dnf(z_final, rules=rules)

            b_total = len(label)
            total_samples += b_total

            # Coverage
            b_cov = covered_mask.sum().item()
            covered_samples += b_cov

            # Precision (trên các mẫu được bao phủ)
            if b_cov > 0:
                rule_correct += (dnf_preds[covered_mask] == label[covered_mask]).sum().item()

            # Fidelity: NN prediction trùng với DNF prediction
            model_rule_agree += (nn_preds == dnf_preds).sum().item()

        coverage = (covered_samples / total_samples) if total_samples > 0 else 0.0
        precision = (rule_correct / covered_samples) if covered_samples > 0 else 0.0
        fidelity = (model_rule_agree / total_samples) if total_samples > 0 else 0.0

        return {
            "coverage": coverage,
            "precision": precision,
            "fidelity": fidelity,
            "total_samples": total_samples,
            "covered_samples": covered_samples,
        }

    # ─── Format DNF dạng văn bản ─────────────────────────────────────────────

    @staticmethod
    def format_rule_as_dnf(rule: Dict) -> str:
        action_name = rule["action_name"]
        lines = [f"Action '{action_name}' (idx={rule['action_idx']}):"]

        for ci, clause in enumerate(rule["clauses"]):
            clause_type = clause["type"].upper()
            l2_idx = clause["l2_rule_idx"]
            w = clause["raw_weight"]
            parts = []
            for comp in clause["l1_components"]:
                lits = comp["literals"]
                lit_strs = [
                    f"{'NOT ' if not pos else ''}{name}"
                    for name, pos in lits
                ]
                parts.append(f"({' ∧ '.join(lit_strs)})")
            connector = " ∨ " if clause_type == "OR1" else " ∧ "
            clause_str = f"  Clause {ci+1} [{clause_type}, L2={l2_idx}, w={w:.3f}]:\n    {connector.join(parts)}"
            lines.append(clause_str)

        if not rule["clauses"]:
            lines.append("  (No active affirmative rules extracted)")

        return "\n".join(lines)

    @staticmethod
    def save_rules(rules: List[Dict], output_dir: str):
        """Lưu rules trích xuất ra file JSON và text trong output_dir."""
        import os
        import json
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "dnf_rules.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(rules, f, ensure_ascii=False, indent=2)

        txt_path = os.path.join(output_dir, "dnf_rules.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for rule in rules:
                f.write(RuleExtractor.format_rule_as_dnf(rule) + "\n\n")

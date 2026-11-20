"""Standalone Multi-Axis Concept Bank loader for Paper 2 (100% independent of PRIZM)."""
import json
import os
import numpy as np

AXIS_ORDER = ('topology', 'zone', 'dynamics', 'interaction', 'temporal', 'intent')


class MultiAxisConceptBank:
    """Loads concept_bank_v1.json and per-class annotations into binary/group matrices."""

    def __init__(self, bank_path, ann_path, num_classes):
        with open(bank_path, encoding='utf-8') as f:
            bank = json.load(f)
        with open(ann_path, encoding='utf-8') as f:
            anns = json.load(f)

        self.bank = bank
        self.concept_ids = [c['id'] for c in bank]
        self.concept_axes = [c['axis'] for c in bank]
        self.concept_texts = [f"{c['name']}: {c['description']}" for c in bank]
        self.id_to_idx = {cid: i for i, cid in enumerate(self.concept_ids)}
        self.num_concepts = len(bank)

        axis_to_idx = {a: i for i, a in enumerate(AXIS_ORDER)}
        self.group_indices = np.array([axis_to_idx[a] for a in self.concept_axes], dtype=np.int64)
        self.num_groups = len(AXIS_ORDER)

        # Build (num_classes, P) binary require matrix
        W = np.zeros((num_classes, self.num_concepts), dtype=np.float32)
        for rec in anns:
            cid = int(rec['class_id']) - 1
            if 0 <= cid < num_classes:
                for cname in rec['concepts']:
                    if cname in self.id_to_idx:
                        W[cid, self.id_to_idx[cname]] = 1.0
        self.W_req = W
        # No hand-crafted forbid rules needed; set zeros for API compatibility
        self.W_forb = np.zeros_like(W)

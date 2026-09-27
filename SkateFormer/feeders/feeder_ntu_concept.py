"""
feeder_ntu_concept.py
---------------------
Dataset wrapper for the concept pseudo-label pipeline.

Returns:
    data_numpy:      Skeleton data [C, T, V, M]
    index_t:         Temporal position index [T]
    action_label:    Integer action class (0-indexed)
    concept_target:  Float tensor [num_concepts]; values 0.0, 1.0, or -1.0 (unknown)
    concept_mask:    Bool tensor [num_concepts]; True = position included in BCE
    sample_index:    Original integer index in the dataset

The concept targets / masks are loaded from a pre-built NPZ file
(see tools/build_template_labels.py or tools/refine_pseudolabels.py).

If no pseudo-label NPZ is provided, the dataset falls back to using
zero-initialized concept_targets and False masks (no concept supervision).

This feeder wraps feeders.feeder_ntu.Feeder to reuse its preprocessing
without modifying the original file.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from feeders.feeder_ntu import Feeder as NTUFeeder


class ConceptFeeder(Dataset):
    """
    Concept-aware wrapper around the NTU skeleton feeder.

    Args:
        data_path:        Path to .npz skeleton data.
        split:            'train' or 'test'.
        pseudo_label_npz: Path to pseudo-label NPZ file containing:
                            concept_targets [N, num_concepts]
                            concept_mask    [N, num_concepts]  (bool)
                          If None, concept targets are all-zero / mask all-False.
        num_concepts:     Number of concept dimensions (default 74).
        no_augment:       If True, override split to 'test' preprocessing
                          (no augmentation) while still using the train portion
                          of the data.  Useful for OOF prediction inference.
        feeder_kwargs:    Additional kwargs forwarded to NTUFeeder.
    """

    NUM_CONCEPTS = 74

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        pseudo_label_npz: Optional[str] = None,
        num_concepts: int = 74,
        no_augment: bool = False,
        **feeder_kwargs,
    ):
        self.split = split
        self.num_concepts = num_concepts
        self.no_augment = no_augment

        # ---- Build underlying NTU feeder ----
        # When no_augment=True we still load the 'train' data slice but
        # disable augmentation by overriding the split field after construction.
        base_split = "train" if split == "train" else split
        self._base_feeder = NTUFeeder(
            data_path=data_path,
            split=base_split,
            **feeder_kwargs,
        )

        if no_augment:
            # Override split so NTUFeeder.__getitem__ skips augmentation branches
            self._base_feeder.split = "test"

        self.n = len(self._base_feeder)

        # ---- Load pseudo-labels ----
        if pseudo_label_npz is not None and os.path.isfile(pseudo_label_npz):
            npz = np.load(pseudo_label_npz)
            self.concept_targets = npz["concept_targets"].astype(np.float32)   # [N, C]
            self.concept_mask    = npz["concept_mask"].astype(bool)             # [N, C]
            assert len(self.concept_targets) == self.n, (
                f"Pseudo-label NPZ has {len(self.concept_targets)} rows but "
                f"dataset has {self.n} samples."
            )
        else:
            if pseudo_label_npz is not None:
                print(f"[ConceptFeeder] WARNING: pseudo_label_npz not found: {pseudo_label_npz}")
                print("[ConceptFeeder] Falling back to zero concept targets (mask=False).")
            self.concept_targets = np.full((self.n, num_concepts), -1.0, dtype=np.float32)
            self.concept_mask    = np.zeros((self.n, num_concepts), dtype=bool)

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n

    def __getitem__(
        self, index: int
    ) -> Tuple[np.ndarray, np.ndarray, int, torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            data_numpy:      [C, T, V, M]
            index_t:         [T]  temporal indices
            action_label:    int
            concept_target:  [num_concepts]  float (0.0, 1.0, or -1.0)
            concept_mask:    [num_concepts]  bool
            sample_index:    int (original index in dataset)
        """
        # Retrieve base feeder item
        # NTUFeeder returns: (data_numpy, index_t, label, index)
        data_numpy, index_t, action_label, orig_index = self._base_feeder[index]

        concept_target = torch.from_numpy(self.concept_targets[index])
        concept_mask   = torch.from_numpy(self.concept_mask[index])

        return data_numpy, index_t, action_label, concept_target, concept_mask, orig_index

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def label(self):
        """Expose label array for compatibility with top_k and stratified splits."""
        return self._base_feeder.label

    @property
    def sample_name(self):
        return self._base_feeder.sample_name

    def top_k(self, score: np.ndarray, top_k: int) -> float:
        return self._base_feeder.top_k(score, top_k)

    def update_pseudo_labels(
        self,
        concept_targets: np.ndarray,
        concept_mask: np.ndarray,
        save_path: Optional[str] = None,
    ):
        """
        Hot-swap pseudo-labels at runtime (e.g., after refinement round).

        Args:
            concept_targets: [N, num_concepts] float32
            concept_mask:    [N, num_concepts] bool
            save_path:       If provided, save updated NPZ.
        """
        assert concept_targets.shape == (self.n, self.num_concepts), (
            f"Expected ({self.n}, {self.num_concepts}), got {concept_targets.shape}"
        )
        self.concept_targets = concept_targets.astype(np.float32)
        self.concept_mask    = concept_mask.astype(bool)

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez_compressed(
                save_path,
                concept_targets=self.concept_targets,
                concept_mask=self.concept_mask,
                action_labels=self.label,
            )
            print(f"[ConceptFeeder] Saved updated pseudo-labels to {save_path}")


class ConceptFeederOOF(ConceptFeeder):
    """
    Variant of ConceptFeeder that supports index-based sub-sampling for OOF folds.

    Pass `indices` to restrict the dataset to a subset of samples (e.g., fold train
    or fold val).  Pseudo-labels are also sub-sampled accordingly.
    """

    def __init__(
        self,
        indices: np.ndarray,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.n = len(self.indices)

        # Sub-sample pseudo-labels
        self.concept_targets = self.concept_targets[self.indices]
        self.concept_mask    = self.concept_mask[self.indices]

        # Sub-sample base feeder data
        self._base_feeder.data  = self._base_feeder.data[self.indices]
        self._base_feeder.label = self._base_feeder.label[self.indices]
        self._base_feeder.sample_name = [
            self._base_feeder.sample_name[i] for i in self.indices
        ]

    @property
    def label(self):
        return self._base_feeder.label

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int):
        return super().__getitem__(index)

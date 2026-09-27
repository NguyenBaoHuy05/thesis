"""Utilities for the relation-free hierarchical concept ontology.

The v2 model deliberately keeps the ontology unary.  A concept belongs to a
parent group, but no concept pair is interpreted as co-occurring, exclusive,
or temporally ordered.  This module is the single parser used by the model
and the audit tools so that training and analysis cannot silently disagree on
the hierarchy.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch


def load_ontology_metadata(path: str | Path, num_concepts: int | None = None) -> list[dict[str, str]]:
    """Load and validate v2 concept metadata in contiguous id order."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"ontology metadata is empty: {path}")
    required = {"concept_id", "concept_name", "parent_name", "level"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"ontology metadata is missing columns: {sorted(missing)}")
    rows.sort(key=lambda row: int(row["concept_id"]))
    ids = [int(row["concept_id"]) for row in rows]
    if ids != list(range(len(rows))):
        raise ValueError("ontology concept ids must be contiguous and zero-based")
    if num_concepts is not None and len(rows) != int(num_concepts):
        raise ValueError(
            f"ontology has {len(rows)} concepts, expected {int(num_concepts)}"
        )
    return rows


def build_parent_index(
    metadata: list[dict[str, str]],
) -> tuple[list[str], torch.Tensor, dict[str, list[int]], dict[str, int]]:
    """Return parent names, concept->parent ids, children, and parent levels."""
    parent_names: list[str] = []
    parent_levels: dict[str, int] = {}
    for row in metadata:
        parent = row["parent_name"].strip()
        if not parent:
            raise ValueError(f"concept {row['concept_id']} has an empty parent")
        if parent not in parent_names:
            parent_names.append(parent)
        level = max(int(row.get("level", "0")) - 1, 0)
        parent_levels[parent] = min(parent_levels.get(parent, level), level)
    parent_to_id = {name: index for index, name in enumerate(parent_names)}
    concept_to_parent = torch.tensor(
        [parent_to_id[row["parent_name"].strip()] for row in metadata],
        dtype=torch.long,
    )
    children = {name: [] for name in parent_names}
    for concept_id, row in enumerate(metadata):
        children[row["parent_name"].strip()].append(concept_id)
    return parent_names, concept_to_parent, children, parent_levels


def load_action_concept_matrix_v2(
    path: str | Path,
    num_classes: int,
    num_concepts: int,
) -> tuple[torch.Tensor, list[str], list[str]]:
    """Load the class-to-concept support matrix without relation metadata."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"action-concept CSV is empty: {path}")
    header = rows[0]
    if len(header) != int(num_concepts) + 1 or header[0] != "action_class":
        raise ValueError(
            f"expected action_class + {num_concepts} concept columns, got {len(header)}"
        )
    if len(rows[1:]) < int(num_classes):
        raise ValueError(
            f"expected at least {num_classes} action rows, got {len(rows[1:])}"
        )
    # The shared NTU60+120 ontology stores all 120 actions in one matrix.
    # NTU60 uses the first 60 rows; NTU120 uses all rows.
    rows = rows[: int(num_classes) + 1]
    action_names: list[str] = []
    values: list[list[float]] = []
    for row in rows[1:]:
        if len(row) != len(header):
            raise ValueError("inconsistent action-concept row width")
        action_names.append(row[0])
        try:
            vector = [float(value) for value in row[1:]]
        except ValueError as exc:
            raise ValueError(f"non-numeric concept value for action {row[0]}") from exc
        if any(value < 0.0 or value > 1.0 for value in vector):
            raise ValueError("action-concept support values must be in [0, 1]")
        values.append(vector)
    return torch.tensor(values, dtype=torch.float32), action_names, header[1:]


def load_action_concept_annotations(
    path: str | Path,
    num_classes: int,
    num_concepts: int,
    action_names: list[str],
    concept_names: list[str],
    zero_is_negative: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load sparse positive/negative/unknown mapping annotations.

    Rows that are not present are unknown by design unless ``zero_is_negative``
    is enabled for a temporary dense-negative experiment. The returned
    confidence tensor is zero for unknown entries and is available for future
    confidence-weighted losses without changing the matrix file format.
    """
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"action_id", "action_class", "concept_id", "concept_name", "state", "confidence"}
    if not rows:
        raise ValueError(f"action-concept annotation file is empty: {path}")
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"action-concept annotations missing columns: {sorted(missing)}")
    positive = torch.zeros((int(num_classes), int(num_concepts)), dtype=torch.float32)
    negative = torch.ones_like(positive) if zero_is_negative else torch.zeros_like(positive)
    confidence = torch.zeros_like(positive)
    valid_states = {"positive", "negative", "unknown"}
    for row in rows:
        action_id = int(row["action_id"])
        concept_id = int(row["concept_id"])
        # A shared 60+120 labels file contains annotations for actions that
        # are outside the current dataset subset.
        if action_id >= int(num_classes):
            continue
        if not (0 <= action_id < int(num_classes)):
            raise ValueError(f"invalid action id in annotations: {action_id}")
        if not (0 <= concept_id < int(num_concepts)):
            raise ValueError(f"invalid concept id in annotations: {concept_id}")
        if row["action_class"] != action_names[action_id]:
            raise ValueError("annotation action name does not match action matrix ordering")
        if row["concept_name"] != concept_names[concept_id]:
            raise ValueError("annotation concept name does not match concept matrix ordering")
        state = row["state"].strip().lower()
        if state not in valid_states:
            raise ValueError(f"unsupported annotation state: {state}")
        value = float(row["confidence"])
        if not 0.0 <= value <= 1.0:
            raise ValueError("annotation confidence must be in [0, 1]")
        confidence[action_id, concept_id] = max(confidence[action_id, concept_id], value)
        if state == "positive":
            positive[action_id, concept_id] = 1.0
            negative[action_id, concept_id] = 0.0
        elif state == "negative":
            negative[action_id, concept_id] = 1.0
        elif state == "unknown" and not zero_is_negative:
            negative[action_id, concept_id] = 0.0
    if torch.any((positive > 0) & (negative > 0)):
        raise ValueError("a mapping cannot be both positive and negative")
    return positive, negative, confidence


def build_parent_support(
    support: torch.Tensor, concept_to_parent: torch.Tensor, num_parents: int,
) -> torch.Tensor:
    """Derive positive parent supervision from positive child support.

    This is a conservative weak label: a parent is positive whenever at least
    one supported child is positive.  Unknown children do not become negative
    evidence, which is important because the source matrix is incomplete.
    """
    result = support.new_zeros((support.shape[0], int(num_parents)))
    for parent_id in range(int(num_parents)):
        child_ids = torch.nonzero(concept_to_parent == parent_id, as_tuple=False).flatten()
        if child_ids.numel():
            result[:, parent_id] = support[:, child_ids].amax(dim=1)
    return result


def ontology_summary(metadata: list[dict[str, str]]) -> dict[str, Any]:
    parents, concept_to_parent, children, parent_levels = build_parent_index(metadata)
    return {
        "num_concepts": len(metadata),
        "num_parents": len(parents),
        "parents": [
            {
                "parent_id": index,
                "parent_name": name,
                "level": parent_levels[name],
                "num_children": len(children[name]),
            }
            for index, name in enumerate(parents)
        ],
    }

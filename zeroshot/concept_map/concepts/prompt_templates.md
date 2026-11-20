# 3-Stage Prompting Protocol for Multi-Axis Semantic-Kinematic Concept Bank (MSK-CB)

This document defines the standardized, reproducible prompting protocol used to construct the **Multi-Axis Semantic-Kinematic Concept Bank (MSK-CB)** for Zero-Shot Skeleton Action Recognition (Paper 2). The protocol is strictly **zero-shot safe** (no test samples, no seen/unseen split information, and no class-specific leakage) and completely independent of rule-based geometric KGs.

---

## Stage 1: Axis-Guided Concept Bank Discovery Prompt

```text
[SYSTEM]
You are a biomechanics and human action understanding specialist designing a reusable, domain-general sub-action concept vocabulary for 3D skeleton action recognition.

[USER]
Design a structured vocabulary of ~70–80 human action concepts organized into six orthogonal semantic-kinematic axes:
  1. topology    : Body-part involvement and coordination pattern (unilateral, bilateral symmetric/asymmetric, head/neck, torso, lower-body, whole-body, two-person).
  2. zone        : Spatial action zone relative to the human body (mouth/face, head/ear, neck, chest, back, front workspace, waist/pocket, overhead, lateral, floor/foot, forward reach, interpersonal gap).
  3. dynamics    : Kinematic motion dynamics and velocity profile (ballistic strike, rhythmic oscillation, fine wrist/finger motion, push/pull, lift/lower, circular/twist, opening/closing, locomotion, airborne jump, instability, static hold, head gaze).
  4. interaction : Physical affordance and interaction topology inferred from body motion (self-contact, hand-to-hand contact, single/bimanual object manipulation, donning/doffing garment, device operation, symbolic hand sign, friendly/exchange/aggressive/proxemic two-person interaction, object release).
  5. temporal    : Temporal phase structure (single impulsive, cyclic repetitive, sustained stationary, reach-and-retract, state transition, continuous flow).
  6. intent      : High-level functional behavioral category (eating/drinking, grooming, dressing, object/container handling, paper/office, digital device use, greeting, expressive gesture, health/pain reflex, posture change, ball sport, fitness exercise, social play/support, conflict/aggression, spatial navigation, micro-fidget/play).

Strict Constraints:
- Anti-Leakage: Never use specific dataset class names as concept names. Every concept must be a reusable attribute shared across multiple human actions.
- CLIP-Friendly Descriptions: Each concept must have a concise, natural English sentence (10–18 words) describing the observable visual-kinematic pattern so a pretrained vision-language text encoder (e.g., CLIP) embeds it cleanly.
- Orthogonality: Avoid redundant synonyms within the same axis.

Output format: JSON list of objects with keys {"id", "axis", "name", "description"}.
```

---

## Stage 2: Per-Class Multi-Axis Concept Annotation Prompt

```text
[SYSTEM]
You are annotating human action classes using a fixed Multi-Axis Semantic-Kinematic Concept Bank. You only have access to the action class name and the concept vocabulary.

[USER]
Given the fixed 6-axis concept bank below:
<CONCEPT_BANK_JSON>

Annotate the following action class:
  Action Class: "<CLASS_NAME>"

Instructions:
1. Select ALL concepts from the vocabulary that clearly characterize the canonical physical execution of "<CLASS_NAME>".
2. Ensure coverage across all six axes (typically 1–3 concepts per axis, 7–12 concepts total per action).
3. Do NOT invent new concept IDs; only select from the provided vocabulary.
4. Distinguish fine-grained pairs carefully (e.g., putting on vs. taking off, clapping vs. rubbing hands, reading vs. writing vs. typing vs. playing with phone) using dynamics, zone, interaction, and temporal concepts.

Output format:
{
  "class_id": <INT>,
  "class_name": "<CLASS_NAME>",
  "concepts": ["<axis.concept_id>", ...]
}
```

---

## Stage 3: Discriminative Verification & Consistency Audit Prompt

```text
[SYSTEM]
You are auditing a multi-axis action concept annotation table for zero-shot discriminability and semantic consistency.

[USER]
Below are pairs of action classes whose current concept sets have Jaccard similarity >= 0.75:
<HIGH_OVERLAP_PAIRS>

For each pair (A, B):
1. Identify the key observable kinematic, spatial, temporal, or interaction difference between A and B.
2. Adjust their concept assignments using ONLY existing IDs from the fixed concept bank so that:
   - No two distinct action classes share an identical concept vector (0 exact duplicates).
   - Every concept in the bank is used by at least 2 classes (no orphan concepts).
   - Opposite/reversed action pairs (e.g., wear vs. take off, sit down vs. stand up, walk toward vs. walk apart) differ explicitly on directional dynamics (`dyn.pull_retract_in` vs `dyn.push_extend_out`, `dyn.lift_raise_up` vs `dyn.lower_descend_drop`, `inter.garment_wear_put` vs `inter.garment_remove_off`).

Return the updated JSON annotations and a brief audit summary.
```

"""Turn the locked vocabulary (zeroshot/primitive.json) into {primitive_list} for 03."""
import json
import sys
from collections import defaultdict

PART_ORDER = ["body", "torso", "head", "arm", "hand", "leg", "foot", "interaction"]

# Groups whose members are defined "for any hand / any arm": the two hands or arms can
# satisfy different members at the same moment, so the group is not a real exclusion.
# Their tag is dropped here and they must not produce exclusion constraints in L_cons.
NOT_EXCLUSIVE = {"hand_contact_zone", "thumb_orientation", "elbow_angle",
                 "upper_arm_direction", "hand_reach_speed", "hand_oscillation"}

LEGEND = ("Notation in definitions: L = body-size length unit; H0 = reference standing "
          "height; f / lat / up = the person's forward / lateral / up axes; "
          "pH, pT, pS, pE, pW = hand, hand tip, shoulder, elbow, wrist of the same arm; "
          "A / B = the two persons.")


def format_primitive_list(vocab_path: str) -> str:
    with open(vocab_path, encoding="utf-8") as f:
        data = json.load(f)
    prims = data["primitives"] if isinstance(data, dict) else data   # 02 output or plain list
    by_part = defaultdict(list)
    for p in prims:
        by_part[p["part"]].append(p)
    lines = [LEGEND, ""]
    for part in PART_ORDER + sorted(set(by_part) - set(PART_ORDER)):
        if part not in by_part:
            continue
        lines.append("# " + part)
        for p in sorted(by_part[part], key=lambda q: q["id"]):   # fixed order for every class
            g = p.get("exclusive_group")
            group = " [{}]".format(g) if g and g not in NOT_EXCLUSIVE else ""
            lines.append("{}: {} | {}{}".format(p["id"], p["description"], p["definition"], group))
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_primitive_list(sys.argv[1]))

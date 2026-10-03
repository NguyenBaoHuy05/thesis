"""Check action ids used by the concept CSV and the NTU label table."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path


CSV_ALIASES = {
    "eatmeal": "eatmealsnack",
    "brushteeth": "brushingteeth",
    "brushhair": "brushinghair",
    "sitdown": "sittingdown",
    "standup": "standingupfromsittingposition",
    "putonjacket": "wearjacket",
    "putonashoe": "wearashoe",
    "putonglasses": "wearonglasses",
    "hopping": "hoppingonefootjumping",
    "phonecall": "makeaphonecallanswerphone",
    "playwithphonetablet": "playingwithphonetablet",
    "typeonakeyboard": "typingonakeyboard",
    "pointtosomething": "pointingtosomethingwithfinger",
    "rubtwohands": "rubtwohandstogether",
    "putpalmstogether": "putthepalmstogether",
    "crosshandsinfront": "crosshandsinfrontsaystop",
    "fallingdown": "falling",
    "headache": "touchheadheadache",
    "chestpain": "touchcheststomachacheheartpain",
    "backpain": "touchbackbackache",
    "neckpain": "touchneckneckache",
    "nauseavomiting": "nauseaorvomitingcondition",
    "fanself": "useafanwithhandorpaperfeelingwarm",
    "punchslap": "punchingslappingotherperson",
    "kicking": "kickingotherperson",
    "pushing": "pushingotherperson",
    "patonback": "patonbackofotherperson",
    "pointfinger": "pointfingerattheotherperson",
    "hugging": "huggingotherperson",
    "givingobject": "givingsomethingtootherperson",
    "touchpocket": "touchotherpersonspocket",
    "shakinghands": "handshaking",
    "walkingtowards": "walkingtowardseachother",
    "walkingapart": "walkingapartfromeachother",
}


def normalize(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "", value.lower())
    return CSV_ALIASES.get(value, value)


def check_mapping(csv_path: Path, label_set: str = "ntu120") -> dict:
    sys.path.insert(0, str(csv_path.parents[1]))
    from skateformer.labels import LABEL_SETS

    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        rows = [row for row in list(csv.reader(handle))[1:] if row]
    csv_names = [row[0] for row in rows]
    labels = LABEL_SETS[label_set]
    count = min(len(csv_names), len(labels))
    mismatches = []
    for index in range(count):
        if normalize(csv_names[index]) != normalize(labels[index]):
            mismatches.append({
                "action_id": index,
                "concept_csv": csv_names[index],
                "label_table": labels[index],
            })
    result = {
        "csv_actions": len(csv_names),
        "label_actions": len(labels),
        "checked": count,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "hard_negative_ids": {
            "A24_A51": [23, 50],
            "A31_A54": [30, 53],
            "A37_A38": [36, 37],
            "A11_A12_A29_A30": [10, 11, 28, 29],
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept-csv", required=True)
    parser.add_argument("--label-set", default="ntu120")
    args = parser.parse_args()
    check_mapping(Path(args.concept_csv), args.label_set)


if __name__ == "__main__":
    main()

"""Write annotations/kg_inputs/actions.json: id, name and number of people per NTU120 class."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.concept_schema import NTU_ACTION_NAMES  # noqa: E402

# Mutual actions: A50-A60 and A106-A120 (zero-based 49-59, 105-119).
TWO_PERSON = set(range(49, 60)) | set(range(105, 120))

if __name__ == "__main__":
    os.makedirs("annotations/kg_inputs", exist_ok=True)
    actions = [{"id": i, "name": name, "persons": 2 if i in TWO_PERSON else 1}
               for i, name in enumerate(NTU_ACTION_NAMES)]
    with open("annotations/kg_inputs/actions.json", "w", encoding="utf-8") as handle:
        json.dump({"actions": actions}, handle, indent=1)
    print(len(actions), [NTU_ACTION_NAMES[i] for i in (49, 59, 105, 119)])

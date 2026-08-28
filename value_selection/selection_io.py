"""
Loads the output of gate.py (or any classified_values-style JSON, including
the pre-existing files under steering-pluralism/data/classified_values) into
a {sample_id: [value_slug, ...]} mapping, for the steering/aggregation
pipeline to know which values to generate steered comments for on each
sample. See value_selection/gate.py for the schema this reads.
"""

import json
from typing import Dict, List

from values import LABEL_TO_SLUG


def load_selected_values(classification_file: str) -> Dict[int, List[str]]:
    with open(classification_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    selected: Dict[int, List[str]] = {}
    for result in data.get("results", []):
        sample_id = result["sample_id"]
        slugs = []
        for pred in result.get("top_k_predictions", []):
            slug = LABEL_TO_SLUG.get(pred["label"])
            if slug is None:
                continue  # unknown/renamed label, skip rather than fail the whole run
            slugs.append(slug)
        selected[sample_id] = slugs
    return selected

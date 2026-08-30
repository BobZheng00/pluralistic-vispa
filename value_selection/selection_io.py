"""
Loads the output of gate.py into a {sample_id: [value_slug, ...]} mapping,
for the steering/aggregation pipeline to know which values to generate
steered comments for on each sample. `sample_id` here must match the `id`
field pipeline/run_*.py reads from the dataset item (item.get("id", idx)) —
gate.py keys its output the same way, so a gate run over a given input file
joins correctly against that same file in pipeline/run_*.py.

Note: the example files under data/classified_values/ predate this fix and
key by array position instead of the dataset's `id` field (a relic of the
original, lost selection script). They won't join correctly against
modular_pluralism's actual input files — regenerate them with the current
gate.py if you need real selections to run the pipeline against.
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

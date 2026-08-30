#!/usr/bin/env python3
"""
Distributional evaluation (paper §4.3): Jensen-Shannon distance between
gold_distribution and pipeline output's pred_distribution, plus
"most likely correctness" (top-prediction match). Covers both Distributional
mode (MoralChoice / GlobalOpinionQA) and Steerable+OpinionQA (paper uses
this same "most likely correctness" metric there — pass
--dataset_type steerable_opinionqa for pipeline/run_steerable.py
--task_type opinionqa output).

All pipeline/run_*.py scripts write `pred_distribution` regardless of which
steering backend produced it; the backend is chosen via --backend rather
than encoded in the field name.

Usage:
    python eval_distributional.py --input_file results/distributional_moralchoice_results.json --dataset_type moralchoice
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import scipy.spatial.distance

OPINIONQA_ATTRIBUTE_MAP = {
    "POLPARTY": "political party", "POLIDEOLOGY": "political ideology", "RELIG": "religion",
    "RACE": "race", "EDUCATION": "education", "INCOME": "income",
    "CREGION": "region in the United States", "SEX": "sex",
}
GLOBALOPINIONQA_ATTRIBUTE_MAP = {
    "US": "United States", "Fr": "France", "Ge": "Germany", "Ja": "Japan",
    "In": "India", "Ar": "Argentina", "Ni": "Nigeria", "Avg.": "an overall average",
}
DATASET_ATTRIBUTES = {
    "steerable_opinionqa": ["POLPARTY", "POLIDEOLOGY", "RELIG", "RACE", "EDUCATION", "INCOME", "CREGION", "SEX"],
    "globalopinionqa": ["US", "Fr", "Ge", "Ja", "In", "Ar", "Ni", "Avg."],
    "moralchoice": ["low_ambiguity", "high_ambiguity"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Distributional (and Steerable/OpinionQA) results")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--dataset_type", type=str, default=None,
                         choices=["steerable_opinionqa", "globalopinionqa", "moralchoice"])
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def detect_dataset_type(input_file: str) -> Optional[str]:
    low = input_file.lower()
    if "steerable" in low and "opinionqa" in low:
        return "steerable_opinionqa"
    if "globalopinionqa" in low:
        return "globalopinionqa"
    if "moralchoice" in low:
        return "moralchoice"
    return None


def _score(data: List[Dict[str, Any]], attribute: Optional[str]) -> Dict[str, Any]:
    distances, most_likely_correct = [], []
    for item in data:
        if attribute is not None:
            if attribute not in str(item.get("attribute", "")):
                continue

        gold_dist = item.get("gold_distribution")
        pred_dist = item.get("pred_distribution")
        if gold_dist is None or pred_dist is None:
            continue

        js_dist = scipy.spatial.distance.jensenshannon(gold_dist, pred_dist)
        if not (0 <= js_dist <= 1):
            continue
        distances.append(js_dist)
        most_likely_correct.append(int(gold_dist.index(max(gold_dist)) == pred_dist.index(max(pred_dist))))

    if not distances:
        return {"samples": 0, "avg_js_distance": None, "std_js_distance": None, "most_likely_accuracy": None}

    avg = sum(distances) / len(distances)
    std = (sum((x - avg) ** 2 for x in distances) / len(distances)) ** 0.5
    return {
        "samples": len(distances),
        "avg_js_distance": avg,
        "std_js_distance": std,
        "most_likely_accuracy": sum(most_likely_correct) / len(most_likely_correct),
    }


def main():
    args = parse_args()
    if not os.path.exists(args.input_file):
        raise SystemExit(f"Error: {args.input_file} does not exist.")

    dataset_type = args.dataset_type or detect_dataset_type(args.input_file)
    print(f"Dataset type: {dataset_type or 'unknown (evaluating all, no per-attribute breakdown)'}")

    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if args.limit is not None:
        data = data[: args.limit]
    print(f"Total samples loaded: {len(data)}")

    results = {"input_file": args.input_file, "dataset_type": dataset_type, "per_attribute": {}}

    attributes = DATASET_ATTRIBUTES.get(dataset_type, [])
    for attribute in attributes:
        attr_results = _score(data, attribute)
        results["per_attribute"][attribute] = attr_results
        if attr_results["samples"] > 0:
            print(f"\nAttribute: {attribute}")
            print(f"  Samples: {attr_results['samples']}")
            print(f"  Avg JS Distance: {attr_results['avg_js_distance']:.4f}")
            print(f"  Most Likely Accuracy: {attr_results['most_likely_accuracy']:.4f}")

    overall = _score(data, attribute=None)
    results["overall"] = overall

    print(f"\n{'=' * 60}\nOverall\n{'=' * 60}")
    print(f"Total samples: {overall['samples']}")
    if overall["samples"] > 0:
        print(f"Avg JS Distance: {overall['avg_js_distance']:.4f}")
        print(f"Most Likely Accuracy: {overall['most_likely_accuracy']:.4f}")

    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Automatic value selection via NLI-based relevance scoring (VISPA paper, Section 3.1).

Reimplementation notice: the original selection script that produced
data/classified_values/*.json ran on the NCI Gadi HPC cluster and was lost
when its working directory was cleaned up. This is a from-scratch
reimplementation based on the paper's description and the schema of the
surviving classified_values/*.json outputs: zero-shot NLI classification of
the input text against all 31 candidate values (independent entailment
scoring per value, since a scenario can be relevant to more than one value
at once), keeping the top-k highest-scoring values.

Usage:
    python gate.py -i path/to/dataset.json -o path/to/output.json [-k 6]
"""

import argparse
import json
import os

from transformers import pipeline

from values import ALL_VALUES

DEFAULT_MODEL = "sileod/deberta-v3-base-tasksource-nli"
DEFAULT_HYPOTHESIS_TEMPLATE = "This text is relevant to the value of {}."

TEXT_FIELD_CANDIDATES = ["situation", "question", "input", "text"]
GOLD_FIELD_CANDIDATES = ["vrd", "ground_truth", "gold"]


def find_field(item, candidates):
    for field in candidates:
        if item.get(field):
            return item[field]
    return None


def classify_dataset(data, classifier, labels, top_k, hypothesis_template):
    results = []
    for i, item in enumerate(data):
        text = find_field(item, TEXT_FIELD_CANDIDATES)
        if text is None:
            raise KeyError(
                f"Item {i} has none of the expected text fields {TEXT_FIELD_CANDIDATES}: "
                f"{list(item.keys())}"
            )
        gold = find_field(item, GOLD_FIELD_CANDIDATES)

        out = classifier(
            text,
            candidate_labels=labels,
            hypothesis_template=hypothesis_template,
            multi_label=True,
        )
        pairs = sorted(zip(out["labels"], out["scores"]), key=lambda p: p[1], reverse=True)
        top_k_predictions = [{"label": label, "score": score} for label, score in pairs[:top_k]]

        record = {"sample_id": i, "text": text, "top_k_predictions": top_k_predictions}
        if gold is not None:
            record["ground_truth"] = gold
        results.append(record)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", required=True, help="path to input dataset json (list of items)")
    parser.add_argument("-o", "--output", required=True, help="path to write the classified_values-style json")
    parser.add_argument("-k", "--top_k", type=int, default=6, help="number of top-relevance values to keep")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help="NLI model used for zero-shot classification")
    parser.add_argument("--hypothesis_template", default=DEFAULT_HYPOTHESIS_TEMPLATE)
    parser.add_argument("--device", type=int, default=-1, help="CUDA device id, -1 for CPU")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels = [label for _, label, _ in ALL_VALUES]

    classifier = pipeline("zero-shot-classification", model=args.model, device=args.device)

    results = classify_dataset(data, classifier, labels, args.top_k, args.hypothesis_template)

    output = {
        "metadata": {
            "model": args.model,
            "data_path": os.path.abspath(args.input),
            "num_samples": len(data),
            "top_k": args.top_k,
            "value_labels": labels,
        },
        "results": results,
    }

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(results)} classified records to {args.output}")


if __name__ == "__main__":
    main()

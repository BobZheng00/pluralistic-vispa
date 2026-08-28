#!/usr/bin/env python3
"""
Steerable (ValueKaleidoscope) evaluation (paper §4.3): three-way accuracy
(support/oppose/either) and the binary variant with "either" removed, for
pipeline/run_steerable.py --task_type valuekaleidoscope output. For the
OpinionQA task type, use eval_distributional.py --dataset_type
steerable_opinionqa instead — that task is scored via the
"most likely correctness" metric over the predicted distribution, not this
discrete-choice accuracy.

Usage:
    python eval_steerable.py --input_file results/steerable_vk_results.json
"""

import argparse
import json
import os

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from tqdm import tqdm

OPTION_TO_INDEX = {"A": 0, "B": 1, "C": 2}


def answer_parsing(response: str) -> str:
    s = response.strip()
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]

    def line_starts_with_option(line):
        for opt in ["A", "B", "C", "D", "E"]:
            if line.startswith(opt + ".") or line.startswith(opt + ")") or line.startswith(opt + ":") or line.startswith(opt + " "):
                return opt
        return None

    if lines:
        ans = line_starts_with_option(lines[-1]) or line_starts_with_option(lines[0])
        if ans:
            return ans

    low = s.lower()
    for phrase in ["the correct answer is", "the answer is", "answer:"]:
        if phrase in low:
            idx = low.find(phrase)
            tail = s[idx + len(phrase):].lstrip()
            if tail.startswith(":"):
                tail = tail[1:].lstrip()
            tail_lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
            first_tail = tail_lines[0] if tail_lines else ""
            ans = line_starts_with_option(first_tail)
            if ans:
                return ans
            if first_tail[:1].lower() in ["a", "b", "c", "d", "e"]:
                return first_tail[:1].upper()

    for option in ["a", "b", "c", "d", "e"]:
        if f"the answer is {option}" in low or f"answer: {option}" in low or f"the correct answer is {option}" in low:
            return option.upper()

    for option in ["A", "B", "C", "D", "E"]:
        if f" {option} " in s or f" {option}." in s or f"{option}: " in s or f"({option})" in s:
            return option

    if s and s[-1] in "ABCDE":
        return s[-1]

    return "Z"  # unparseable


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Steerable (ValueKaleidoscope) results")
    parser.add_argument("--input_file", type=str, required=True, help="pipeline/run_steerable.py (valuekaleidoscope) output json")
    parser.add_argument("--output_file", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.input_file):
        raise SystemExit(f"Error: {args.input_file} does not exist.")

    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    pred, gold, unparsed = [], [], []
    for item in tqdm(data, desc="Parsing"):
        output_text = item.get("output")
        gold_label = item.get("label")
        if not output_text or gold_label not in OPTION_TO_INDEX:
            continue

        gold.append(OPTION_TO_INDEX[gold_label])
        parsed = answer_parsing(output_text)
        if parsed in OPTION_TO_INDEX:
            pred.append(OPTION_TO_INDEX[parsed])
        else:
            unparsed.append({"id": item.get("id"), "parsed": parsed, "gold": gold_label, "output": output_text})
            low = output_text.lower()
            if "support" in low:
                pred.append(0)
            elif "oppose" in low:
                pred.append(1)
            elif "either" in low or "it depends" in low:
                pred.append(2)
            else:
                pred.append(-1)

    valid = [i for i, p in enumerate(pred) if p != -1]
    gold_f, pred_f = [gold[i] for i in valid], [pred[i] for i in valid]

    if not gold_f:
        raise SystemExit("Error: no valid predictions found.")

    print(f"Total items: {len(data)}, parsed predictions: {len(pred_f)}, unparsed: {len(unparsed)}")
    print("--- Three-way (support / oppose / either) ---")
    print("Accuracy:", accuracy_score(gold_f, pred_f))
    print("Balanced accuracy:", balanced_accuracy_score(gold_f, pred_f))
    print("Macro F1:", f1_score(gold_f, pred_f, average="macro", zero_division=0))
    print("Micro F1:", f1_score(gold_f, pred_f, average="micro", zero_division=0))

    pred_b = [p for p, g in zip(pred_f, gold_f) if p != 2 and g != 2]
    gold_b = [g for p, g in zip(pred_f, gold_f) if p != 2 and g != 2]

    results = {
        "input_file": args.input_file,
        "total_items": len(data),
        "parsed_predictions": len(pred_f),
        "unparsed": len(unparsed),
        "three_way": {
            "accuracy": accuracy_score(gold_f, pred_f),
            "balanced_accuracy": balanced_accuracy_score(gold_f, pred_f),
            "macro_f1": f1_score(gold_f, pred_f, average="macro", zero_division=0),
            "micro_f1": f1_score(gold_f, pred_f, average="micro", zero_division=0),
        },
    }

    if gold_b:
        print("--- Binary (either removed) ---")
        print("Accuracy:", accuracy_score(gold_b, pred_b))
        print("Balanced accuracy:", balanced_accuracy_score(gold_b, pred_b))
        print("Macro F1:", f1_score(gold_b, pred_b, average="macro", zero_division=0))
        print("Micro F1:", f1_score(gold_b, pred_b, average="micro", zero_division=0))
        results["binary"] = {
            "accuracy": accuracy_score(gold_b, pred_b),
            "balanced_accuracy": balanced_accuracy_score(gold_b, pred_b),
            "macro_f1": f1_score(gold_b, pred_b, average="macro", zero_division=0),
            "micro_f1": f1_score(gold_b, pred_b, average="micro", zero_division=0),
        }
    else:
        print("No valid binary (non-'either') predictions found.")

    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()

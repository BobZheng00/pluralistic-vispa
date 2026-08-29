#!/usr/bin/env python
"""
End-to-end Distributional pipeline (paper §3.1-3.3): automatic value
selection -> per-value activation steering -> a probability distribution
over answer options per value's steered comment -> averaged into a final
population-level distribution. Two task types, matching modular_pluralism's
ModPlural splits:

  --task_type moralchoice        no demographic attribute
  --task_type globalopinionqa    item['attribute'] is a country name,
                                  folded into the aggregation-stage prompt

Usage:
    python run_distributional.py --task_type moralchoice \\
        --input .../distributional_test_moralchoice_small.json \\
        --selection path/to/gate_output.json \\
        --output results/distributional_moralchoice_results.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    add_steering_hyperparameter_args,
    fit_all_values,
    get_configured_backend,
    load_aggregator_model,
    load_steering_model,
    parse_layers,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "value_selection"))
from selection_io import load_selected_values  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from steering.comment_generation import generate_value_comments  # noqa: E402
from aggregation.distributional import (  # noqa: E402
    aggregate_distributions,
    format_question_for_direct_answer,
    get_probability_distribution,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task_type", required=True, choices=["moralchoice", "globalopinionqa"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steering_model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--aggregator_model", default="meta-llama/Llama-2-13b-chat-hf")
    parser.add_argument("--backend", default="probe_calibrated", choices=["probe_calibrated", "averaging_caa", "projection_pca"])
    add_steering_hyperparameter_args(parser)
    parser.add_argument("--data_dir", default=str(Path(__file__).resolve().parent.parent / "data" / "value"))
    parser.add_argument("--comment_max_new_tokens", type=int, default=200)
    args = parser.parse_args()

    layers = parse_layers(args.layers)

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    selected_values = load_selected_values(args.selection)

    backend = get_configured_backend(args)
    print(f"Loading steering model ({args.backend}): {args.steering_model}")
    steering_model, steering_tokenizer = load_steering_model(args.backend, args.steering_model)

    needed_values = sorted({v for vs in selected_values.values() for v in vs})
    print(f"Fitting steering directions for {len(needed_values)} values...")
    fitted = fit_all_values(
        backend, steering_model, steering_tokenizer, needed_values, layers, args.data_dir,
    )

    print(f"Generating value-steered comments for {len(data)} items...")
    results = []
    for idx, item in enumerate(data):
        item_id = item.get("id", idx)
        value_slugs = selected_values.get(item_id, [])
        value_states = {v: fitted[v] for v in value_slugs if v in fitted}

        question = format_question_for_direct_answer(item["question"])
        comments = generate_value_comments(
            backend, steering_model, steering_tokenizer, question, value_states,
            max_new_tokens=args.comment_max_new_tokens,
        )
        results.append({"id": item_id, "item": item, "question": question, "comments": comments})
        print(f"  [{idx + 1}/{len(data)}] id={item_id}: {len(comments)} comments")

    del steering_model
    torch.cuda.empty_cache()

    print(f"Loading aggregator model: {args.aggregator_model}")
    aggregator_model, aggregator_tokenizer = load_aggregator_model(args.aggregator_model)

    print("Getting per-value distributions and aggregating...")
    final_results = []
    for result in results:
        item = result["item"]
        attribute = item.get("attribute") if args.task_type == "globalopinionqa" else None

        per_value_distributions = []
        for value_slug, comment in result["comments"].items():
            dist = get_probability_distribution(
                comment, result["question"], item["options"], aggregator_model, aggregator_tokenizer, attribute=attribute,
            )
            per_value_distributions.append(dist)

        pred_distribution = aggregate_distributions(per_value_distributions)
        final_results.append({
            "id": result["id"],
            "question": item["question"],
            "attribute": item.get("attribute"),
            "used_values": list(result["comments"].keys()),
            "pred_distribution": pred_distribution,
            "gold_distribution": item.get("gold_distribution"),
        })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(final_results)} results to {args.output}")


if __name__ == "__main__":
    main()

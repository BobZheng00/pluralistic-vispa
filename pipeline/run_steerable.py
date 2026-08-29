#!/usr/bin/env python
"""
End-to-end Steerable pipeline (paper §3.1-3.3): automatic value selection ->
per-value activation steering -> main LLM selects the best-matching
value-steered comment as a reference passage -> final answer conditioned on
it. Two task types, matching modular_pluralism's ModPlural/VITAL splits:

  --task_type valuekaleidoscope   generative: comments framed around a VRD
                                   (paper Appendix E.8's open-ended prompt),
                                   final answer to the dataset's MC question
  --task_type opinionqa           probability: comments framed around the
                                   question itself, final probability
                                   distribution over answer options

Usage:
    python run_steerable.py --task_type valuekaleidoscope \\
        --input .../steerable_test_valuekaleidoscope_small.json \\
        --selection path/to/gate_output.json \\
        --output results/steerable_vk_results.json
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
from aggregation.steerable import (  # noqa: E402
    generate_final_answer,
    generate_probability_distribution,
    select_comment_for_attribute,
    select_comment_for_vrd,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task_type", required=True, choices=["valuekaleidoscope", "opinionqa"])
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

        if args.task_type == "valuekaleidoscope":
            comment_text, vrd = item["situation"], item.get("vrd")
        else:
            comment_text, vrd = item["question"], None

        comments = generate_value_comments(
            backend, steering_model, steering_tokenizer, comment_text, value_states,
            vrd=vrd, max_new_tokens=args.comment_max_new_tokens,
        )
        results.append({"id": item_id, "item": item, "comments": list(comments.values())})
        print(f"  [{idx + 1}/{len(data)}] id={item_id}: {len(comments)} comments")

    del steering_model
    torch.cuda.empty_cache()

    print(f"Loading aggregator model: {args.aggregator_model}")
    aggregator_model, aggregator_tokenizer = load_aggregator_model(args.aggregator_model)

    print("Selecting best-matching comment and generating final answers...")
    final_results = []
    for result in results:
        item = result["item"]
        comments = result["comments"]

        if args.task_type == "valuekaleidoscope":
            selected, was_random, _ = select_comment_for_vrd(comments, item.get("vrd", ""), aggregator_model, aggregator_tokenizer)
            answer = generate_final_answer(item["input"], selected, aggregator_model, aggregator_tokenizer)
            final_results.append({
                "id": result["id"], "situation": item["situation"], "vrd": item.get("vrd"),
                "selected_comment": selected, "was_randomly_selected": was_random, "output": answer,
                "label": item.get("label"),  # letter (A/B/C) — what eval/eval_steerable.py scores against
                "label_text": item.get("label_text"),
            })
        else:
            selected, was_random, _ = select_comment_for_attribute(comments, item["attribute"], aggregator_model, aggregator_tokenizer)
            answer, pred_distribution = generate_probability_distribution(
                item["question"], item["options"], selected, item["attribute"], aggregator_model, aggregator_tokenizer,
            )
            final_results.append({
                "id": result["id"], "question": item["question"], "attribute": item["attribute"],
                "selected_comment": selected, "was_randomly_selected": was_random,
                "output": answer, "pred_distribution": pred_distribution,
                "gold_distribution": item.get("gold_distribution"),
            })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(final_results)} results to {args.output}")


if __name__ == "__main__":
    main()

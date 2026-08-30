#!/usr/bin/env python
"""
End-to-end Overton pipeline (paper §3.1-3.3): automatic value selection
(pre-computed by value_selection/gate.py) -> per-value activation steering
-> main-LLM aggregation into a single response reflecting diverse values.

Usage:
    python run_overton.py \\
        --input ../third_party/modular_pluralism/input/overton_test_valuekaleidoscope_small.json \\
        --selection path/to/gate_output.json \\
        --output results/overton_results.json
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
    load_aggregator,
    load_steering_model,
    parse_layers,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "value_selection"))
from selection_io import load_selected_values  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from steering.comment_generation import generate_value_comments  # noqa: E402
from aggregation.overton import aggregate  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Overton-mode dataset json (list of {id, situation, ...})")
    parser.add_argument("--selection", required=True, help="value_selection/gate.py output run over --input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steering_model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument(
        "--aggregator_model", default="meta-llama/Llama-2-13b-chat-hf",
        help="HF repo id (e.g. 'org/model') for a local model, or an OpenAI API model id "
        "with no slash (e.g. 'gpt-4o') to use OpenAI's chat completions API instead — "
        "requires OPENAI_API_KEY. See paper Table 8.",
    )
    parser.add_argument("--backend", default="probe_calibrated", choices=["probe_calibrated", "averaging_caa", "projection_pca"])
    add_steering_hyperparameter_args(parser)
    parser.add_argument("--data_dir", default=str(Path(__file__).resolve().parent.parent / "data" / "value"))
    parser.add_argument("--comment_max_new_tokens", type=int, default=200)
    parser.add_argument("--overton_max_new_tokens", type=int, default=300)
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

    print(f"Generating value-steered comments for {len(data)} situations...")
    results = []
    for idx, item in enumerate(data):
        item_id = item.get("id", idx)
        situation = item["situation"]
        value_slugs = selected_values.get(item_id, [])
        value_states = {v: fitted[v] for v in value_slugs if v in fitted}

        comments = generate_value_comments(
            backend, steering_model, steering_tokenizer, situation, value_states,
            max_new_tokens=args.comment_max_new_tokens,
        )
        results.append({
            "id": item_id,
            "situation": situation,
            "vrd": item.get("vrd", []),
            "explanation": item.get("explanation", []),  # eval/eval_overton.py scores output against these
            "used_values": list(comments.keys()),
            "comments": list(comments.values()),
        })
        print(f"  [{idx + 1}/{len(data)}] id={item_id}: {len(comments)} comments ({', '.join(comments.keys())})")

    del steering_model
    torch.cuda.empty_cache()

    print(f"Loading aggregator model: {args.aggregator_model}")
    aggregator = load_aggregator(args.aggregator_model)

    print("Aggregating into Overton responses...")
    for result in results:
        result["output"] = aggregate(
            result["situation"], result["comments"], aggregator,
            max_new_tokens=args.overton_max_new_tokens,
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()

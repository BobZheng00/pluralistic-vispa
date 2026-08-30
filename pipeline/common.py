"""
Shared setup for the three run_*.py orchestration scripts: picking a
steering backend, loading the steering model in the form that backend
expects, loading a value's contrastive pairs, and fitting every value a
dataset's selection step actually needs (once per value, reused across every
sample that selects it — see steering/interface.py).
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "value_selection"))

from steering._third_party import add_to_path  # noqa: E402
from steering.hyperparameters import (  # noqa: E402
    AVERAGING_DEFAULT_ALPHA,
    DEFAULT_LAYER_RANGE,
    PROBE_P0,
    PROJECTION_COEFF,
    get_backend_kwargs,
)


def get_backend(name: str, **kwargs):
    if name == "probe_calibrated":
        from steering.probe_calibrated import ProbeCalibratedSteering
        return ProbeCalibratedSteering(**kwargs)
    if name == "averaging_caa":
        from steering.averaging_caa import AveragingCAASteering
        return AveragingCAASteering(**kwargs)
    if name == "projection_pca":
        from steering.projection_pca import ProjectionPCASteering
        return ProjectionPCASteering(**kwargs)
    raise ValueError(f"Unknown backend '{name}'. Expected one of: probe_calibrated, averaging_caa, projection_pca.")


def add_steering_hyperparameter_args(parser):
    """Add the paper-aligned steering settings shared by all three modes."""
    parser.add_argument("--layers", default=DEFAULT_LAYER_RANGE, help="inclusive physical layer range")
    parser.add_argument("--projection_coeff", type=float, default=PROJECTION_COEFF)
    parser.add_argument("--averaging_default_alpha", type=float, default=AVERAGING_DEFAULT_ALPHA)
    parser.add_argument("--probe_p0", type=float, default=PROBE_P0)
    parser.add_argument(
        "--probe_skip_noop",
        action="store_true",
        help="omit a selected value when probe calibration is a no-op (disabled for paper-aligned runs)",
    )


def get_configured_backend(args):
    """Instantiate a backend with explicit paper-aligned hyperparameters."""
    kwargs = get_backend_kwargs(
        args.backend,
        args.steering_model,
        projection_coeff=args.projection_coeff,
        averaging_default_alpha=args.averaging_default_alpha,
        probe_p0=args.probe_p0,
        probe_skip_noop=args.probe_skip_noop,
    )
    return get_backend(args.backend, **kwargs)


def load_steering_model(backend_name: str, model_path: str):
    """probe_calibrated/averaging_caa need ConVA's ModelWrapper (hook-based
    CAV/CAA injection); projection_pca (RepE) wraps a plain HF model itself
    via its own pipelines. See steering/README.md for why this asymmetry
    comes from the upstream libraries, not from this repo's own design."""
    if backend_name == "projection_pca":
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map={"": "cuda:0"}
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer

    add_to_path("ConVA")
    from src.modelwrapper import ModelWrapper

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map={"": "cuda:0"}
    ).cuda().eval()
    model = ModelWrapper(hf_model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.unk_token if tokenizer.pad_token is None else tokenizer.pad_token
    return model, tokenizer


def load_aggregator(model_path: str):
    """Returns an aggregation.aggregator.Aggregator — OpenAIAggregator for
    closed models (paper §3.3: the aggregator "only consumes the generated
    comments and can therefore be closed-weight, e.g. ChatGPT"; Table 8
    lists gpt-3.5-turbo/gpt-4o), HFAggregator otherwise."""
    from aggregation.aggregator import HFAggregator, OpenAIAggregator, is_openai_model

    if is_openai_model(model_path):
        return OpenAIAggregator(model_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map={"": "cuda:0"}
    ).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return HFAggregator(model, tokenizer)


def load_contrastive_pairs(value_slug: str, data_dir: str) -> Tuple[List[str], List[str]]:
    df = pd.read_csv(Path(data_dir) / f"{value_slug}_context_controlled.csv")
    return df["question_1"].tolist(), df["question_2"].tolist()


def fit_all_values(
    backend,
    steering_model,
    steering_tokenizer,
    value_slugs: List[str],
    layers: List[int],
    data_dir: str,
) -> Dict[str, object]:
    fitted = {}
    for value_slug in value_slugs:
        pos, neg = load_contrastive_pairs(value_slug, data_dir)
        print(f"  fitting '{value_slug}' ({len(pos)} contrastive pairs)...")
        fitted[value_slug] = backend.fit(steering_model, steering_tokenizer, value_slug, pos, neg, layers)
    return fitted


def parse_layers(layers_arg: str) -> List[int]:
    start, end = (int(x) for x in layers_arg.split("-"))
    return list(range(start, end + 1))

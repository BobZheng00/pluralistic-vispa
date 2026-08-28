"""
Projection-based steering (paper Section 3.2.1, Appendix C.1) — the PCA
instantiation, following Representation Engineering (Zou et al., 2025,
https://github.com/andyzoujm/representation-engineering, MIT licensed).

Not vendored, for consistency with the other two backends (see
steering/README.md) even though RepE's license would allow it — run
scripts/setup_dependencies.sh first so third_party/representation-engineering
exists. Unlike ConVA, RepE is pip-installable, so you may alternatively
`pip install -e third_party/representation-engineering` instead of relying
on the sys.path shim in `steering/_third_party.py`; this module works either
way since it just does `import repe`.

Direction: v_V = first PCA component of the per-layer hidden-state
differences between paired positive/negative contrastive examples
(repe.rep_readers.PCARepReader, via the "rep-reading" pipeline). Magnitude:
a fixed steering coefficient, applied via the "rep-control" pipeline
(repe.rep_control_pipeline.RepControlPipeline).
"""

import random
from typing import List

import numpy as np
import torch
from transformers import pipeline

from ._third_party import add_to_path
from .interface import SteeringBackend

add_to_path("representation-engineering")

import repe  # noqa: E402  (RepE, not vendored — see module docstring)

repe.repe_pipeline_registry()

DEFAULT_COEFF = 2.0  # steering strength for the applied direction; tune per model/value as needed


def _pairs_to_repe_format(pos_prompts: List[str], neg_prompts: List[str], seed: int = 0):
    """RepE's rep-reading pipeline expects a flat, per-pair-shuffled list of
    strings plus matching [bool, bool] labels marking which element of each
    consecutive pair was the positive one (see representation-engineering's
    examples/honesty/utils.py:honesty_function_dataset for the reference
    pattern this mirrors).
    """
    rng = random.Random(seed)
    pairs = [[pos, neg] for pos, neg in zip(pos_prompts, neg_prompts)]

    labels = []
    for pair in pairs:
        positive = pair[0]
        rng.shuffle(pair)
        labels.append([item == positive for item in pair])

    flat_data = [item for pair in pairs for item in pair]
    return flat_data, labels


class ProjectionPCASteering(SteeringBackend):
    name = "projection_pca"

    def __init__(self, coeff: float = DEFAULT_COEFF, rep_token: int = -1, n_difference: int = 1, batch_size: int = 8):
        self.coeff = coeff
        self.rep_token = rep_token
        self.n_difference = n_difference
        self.batch_size = batch_size

    def fit(self, model, tokenizer, value_slug, pos_prompts, neg_prompts, layers, **kwargs):
        hidden_layers = [-l for l in layers]  # RepE indexes hidden_layers from the end of the stack (negative ints)
        train_data, train_labels = _pairs_to_repe_format(pos_prompts, neg_prompts)

        rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)
        rep_reader = rep_reading_pipeline.get_directions(
            train_data,
            rep_token=self.rep_token,
            hidden_layers=hidden_layers,
            n_difference=self.n_difference,
            train_labels=train_labels,
            direction_method="pca",
            batch_size=self.batch_size,
        )

        rep_control_pipeline = pipeline(
            "rep-control",
            model=model,
            tokenizer=tokenizer,
            layers=hidden_layers,
            control_method="reading_vec",
        )

        activations = {
            layer: torch.tensor(
                self.coeff * rep_reader.directions[layer] * rep_reader.direction_signs[layer]
            ).to(model.device).half()
            for layer in hidden_layers
        }

        return {"rep_control_pipeline": rep_control_pipeline, "activations": activations}

    def generate(self, model, tokenizer, prompt, state, **gen_kwargs) -> str:
        gen_kwargs.setdefault("do_sample", False)
        outputs = state["rep_control_pipeline"](
            [prompt], activations=state["activations"], batch_size=1, **gen_kwargs
        )
        generated = outputs[0][0]["generated_text"]
        return generated[len(prompt):].strip() if generated.startswith(prompt) else generated.strip()

"""
Averaging-based steering (paper Section 3.2.2, Appendix C.2) — the
mean-activation-difference instantiation, in the spirit of Contrastive
Activation Addition (Rimsky et al., ACL 2024).

Important attribution note: this bridges ConVA's own CAA baseline
(ConVA/src/baselines/caa.py, hr-jin/ConVA, ACL 2025), which VISPA actually
used, NOT the original nrimsky/CAA repository directly. ConVA's `caa.py`
reimplements the same mean-difference idea; it is functionally comparable to
Rimsky et al.'s method but is a distinct implementation. Cite both papers if
you draw on the numbers this backend produces.

Direction: v_V = mean(h(x+)) - mean(h(x-)) over the value's context-controlled
contrastive pairs, per layer (ConVA/src/baselines/caa.py: get_diff_acts()).
Magnitude: a fixed per-value coefficient alpha_V (no dynamic calibration —
that is precisely what distinguishes this instantiation from probe-calibrated
steering).
"""

import numpy as np
import torch

from ._third_party import add_to_path
from .interface import SteeringBackend

add_to_path("ConVA")

from src.baselines.caa import get_diff_acts  # noqa: E402  (ConVA, not vendored — see module docstring)
from src.cav_gen import generate_  # noqa: E402


DEFAULT_ALPHA = 0.5  # VISPA's shared default for values without a prior-work-specific coefficient (paper Appendix C.2)


class AveragingCAASteering(SteeringBackend):
    name = "averaging_caa"

    def __init__(self, alpha: float = DEFAULT_ALPHA, model_name: str = "llama-2"):
        self.alpha = alpha
        self.model_name = model_name

    def fit(self, model, tokenizer, value_slug, pos_prompts, neg_prompts, layers, **kwargs):
        start_layer, end_layer = min(layers), max(layers)
        diff_acts = get_diff_acts(model, tokenizer, start_layer, end_layer, pos_prompts, neg_prompts)

        steer_vecs = {}
        for layer_name, diff_act in diff_acts.items():
            mean_diff = np.mean(diff_act, axis=0)
            steer_vecs[layer_name] = self.alpha * torch.tensor(mean_diff).to(model.device)

        return {"layers": [f"model.layers.{l}" for l in layers], "steer_vecs": steer_vecs}

    def generate(self, model, tokenizer, prompt, state, **gen_kwargs) -> str:
        model.register_forward_hooks(state["layers"])
        try:
            model.set_cavs(state["steer_vecs"])
            try:
                return generate_(model, tokenizer, prompt, model_name=self.model_name, **gen_kwargs)
            finally:
                model.clear_cavs()
        finally:
            model.remove_hooks()

"""
Probe-calibrated steering — VISPA's main/best-performing instantiation
(paper Section 3.2.3, Appendix C.3, Eq. 5).

Bridges to ConVA (Jin et al., "Internal Value Alignment in Large Language
Models through Controlled Value Vector Activation", ACL 2025,
https://github.com/hr-jin/ConVA). Not vendored — run
scripts/setup_dependencies.sh first so third_party/ConVA exists.

Direction: v_V = w / ||w||, the normal vector of a per-layer logistic
regression probe P_V(h) = sigmoid(w.h + b) trained on ConVA's
src.cav.get_cavs() to discriminate positive vs. negative contrastive
activations (ConVA/src/cav.py).

Magnitude: unlike ConVA's own fixed-multiplier usage, VISPA calibrates a
per-input, dynamic epsilon_V(x) = argmin|epsilon| s.t. P_V(h_hat(epsilon)) >=
P0, via ConVA's src.cav_gen.get_epsilon_dict() / get_plus_epsilon_dict(),
which solve exactly this (s_0 = log(P0/(1-P0)), epsilon = (s_0 - b -
w.h) / ||w||^2) per layer (ConVA/src/cav_gen.py).
"""

from typing import Dict, List

from ._third_party import add_to_path
from .interface import SteeringBackend

add_to_path("ConVA")

from src.cav import get_reps, get_cavs  # noqa: E402  (ConVA, not vendored — see module docstring)
from src.cav_gen import (  # noqa: E402
    generate_,
    get_epsilon_dict,
    get_plus_epsilon_dict,
    set_control_by_epsilon_dict,
)


class ProbeCalibratedSteering(SteeringBackend):
    name = "probe_calibrated"

    def __init__(self, p0: float = 0.9, increase: bool = True, model_name: str = "llama-2", use_gate: bool = True):
        # `increase`: whether steering should push P_V up (get_plus_epsilon_dict,
        # used to induce the value) or allow it to move either way
        # (get_epsilon_dict). VISPA always induces the selected value, so the
        # default is True.
        # `use_gate`: if the calibration in `generate` finds epsilon == 0 at
        # every layer (the value is already at/above P0 pre-steering, or the
        # probe can't push it there at all), the value isn't meaningfully
        # present for this input — skip generating a comment for it rather
        # than steering with a no-op vector.
        self.p0 = p0
        self.increase = increase
        self.model_name = model_name
        self.use_gate = use_gate

    def fit(self, model, tokenizer, value_slug, pos_prompts, neg_prompts, layers, p0: float = None, **kwargs) -> Dict:
        """`p0` overrides self.p0 for this specific value (see
        steering/p0_thresholds.py — the paper's experiments used tuned,
        per-value/per-model P0 thresholds rather than one flat value)."""
        layer_names = [f"model.layers.{l}" for l in layers]
        model.register_forward_hooks(layer_names)
        try:
            pos_reps = get_reps(model, tokenizer, pos_prompts, layer_names)
            neg_reps = get_reps(model, tokenizer, neg_prompts, layer_names)
        finally:
            model.remove_hooks()

        cav_dict, classifier_dict = {}, {}
        for layer_name in layer_names:
            cav, log_reg, acc_train, acc_test = get_cavs(pos_reps[layer_name], neg_reps[layer_name])
            cav_dict[layer_name] = cav
            classifier_dict[layer_name] = log_reg

        return {"cav_dict": cav_dict, "classifier_dict": classifier_dict, "p0": p0 if p0 is not None else self.p0}

    def generate(self, model, tokenizer, prompt, state, **gen_kwargs):
        """Returns the steered comment, or None if `use_gate` is set and the
        value wasn't meaningfully present for this input (matches the
        calling convention `steering/comment_generation.py` expects)."""
        get_epsilon = get_plus_epsilon_dict if self.increase else get_epsilon_dict
        epsilon_dict = get_epsilon(
            model, tokenizer, prompt, state["classifier_dict"], state["cav_dict"], p_0=state.get("p0", self.p0)
        )

        if self.use_gate and all(eps == 0 for eps in epsilon_dict.values()):
            return None

        # set_control_by_epsilon_dict registers the forward hooks that add
        # epsilon * cav to each configured layer's output (ConVA/src/modelwrapper.py
        # ModelWrapper.set_cavs), so generation below runs steered.
        set_control_by_epsilon_dict(model, state["cav_dict"], epsilon_dict)
        try:
            return generate_(model, tokenizer, prompt, model_name=self.model_name, **gen_kwargs)
        finally:
            model.clear_cavs()

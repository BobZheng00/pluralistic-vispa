"""Paper-aligned steering hyperparameters shared by every pipeline."""

DEFAULT_LAYER_RANGE = "10-25"
PROJECTION_COEFF = 1.0
AVERAGING_DEFAULT_ALPHA = 0.5
PROBE_P0 = 0.9

# Coefficients for the ten values benchmarked by ConVA, as released in the
# pinned ConVA commit's scripts/run_caa.sh and cited by Appendix C.2.
AVERAGING_ALPHA_BY_VALUE = {
    "achievement": 0.3,
    "stimulation": 0.2,
    "hedonism": 0.05,
    "self-direction": 0.6,
    "power": 0.4,
    "security": 0.4,
    "tradition": 0.3,
    "conformity": 0.11,
    "benevolence": 0.08,
    "universalism": 0.215,
}


def get_averaging_alpha(value_slug: str, default_alpha: float = AVERAGING_DEFAULT_ALPHA) -> float:
    """Return the prior-work coefficient or the paper's shared fallback."""
    return AVERAGING_ALPHA_BY_VALUE.get(value_slug, default_alpha)


def get_backend_kwargs(
    backend_name: str,
    steering_model_path: str,
    projection_coeff: float = PROJECTION_COEFF,
    averaging_default_alpha: float = AVERAGING_DEFAULT_ALPHA,
    probe_p0: float = PROBE_P0,
    probe_skip_noop: bool = False,
):
    """Build backend constructor arguments from the shared pipeline config."""
    model_name = steering_model_path.rsplit("/", 1)[-1]
    if backend_name == "probe_calibrated":
        return {"p0": probe_p0, "model_name": model_name, "use_gate": probe_skip_noop}
    if backend_name == "averaging_caa":
        return {"default_alpha": averaging_default_alpha, "model_name": model_name}
    if backend_name == "projection_pca":
        return {"coeff": projection_coeff}
    raise ValueError(f"Unknown steering backend: {backend_name}")


def to_repe_hidden_layers(layers, num_hidden_layers: int):
    """Map zero-based physical block indices to RepE's negative indices."""
    mapped = []
    for layer in layers:
        if layer < 0 or layer >= num_hidden_layers:
            raise ValueError(
                f"Layer {layer} is outside the model's physical block range "
                f"0-{num_hidden_layers - 1}."
            )
        mapped.append(layer - num_hidden_layers)
    return mapped

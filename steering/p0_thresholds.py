"""
Per-value, per-steering-model P0 thresholds for probe-calibrated steering
(Eq. 5's confidence target). These are the actual tuned constants recovered
from working code, not defaults — dropping them in favor of one flat P0 for
every value materially changes probe_calibrated results, since the tuned
values range from 0.7 to 0.9999999 depending on how separable each value's
probe is. Keyed by the value slugs in value_selection/values.py (the
recovered code keyed them by "{slug}_context_controlled", stripped here).

Only two steering models have tuned thresholds; anything else falls back to
DEFAULT_P0 (0.9), which is untuned and should be treated as a starting point
to calibrate, not a reproduction of any reported number.
"""

DEFAULT_P0 = 0.9

P0_LLAMA_2_7B_CHAT = {
    "achievement": 0.93,
    "stimulation": 0.91,
    "hedonism": 0.83,
    "self-direction": 0.91,
    "power": 0.9,
    "security": 0.9,
    "tradition": 0.92,
    "conformity": 0.7,
    "benevolence": 0.85,
    "universalism": 0.95,
    "power_distance": 0.99,
    "uncertainty_avoidance": 0.99,
    "individualism": 0.95,
    "masculinity": 0.99,
    "long_term_orientation": 0.99,
    "indulgence": 0.99,
    "commonsense_morality": 0.99,
    "deontology": 0.97,
    "utilitarianism": 0.99,
    "justice": 0.99,
    "virtue_ethics": 0.95,
    "ubuntu": 0.97,
    "confucianism": 0.97,
    "face": 0.99,
    "karma": 0.99,
    "honor": 0.97,
    "spiritual": 0.97,
    "fairness": 0.99,
    "truthfulness": 0.95,
    "toxicity": 0.99,
    "harmfulness": 0.99,
}

P0_LLAMA_3_8B_INSTRUCT = {
    "achievement": 0.9995,
    "stimulation": 0.9999,
    "hedonism": 0.99995,
    "self-direction": 0.999,
    "power": 0.99995,
    "security": 0.9999,
    "tradition": 0.9999999,
    "conformity": 0.9999995,
    "benevolence": 0.996,
    "universalism": 0.9999,
    "power_distance": 0.999999,
    "uncertainty_avoidance": 0.999999,
    "individualism": 0.999999,
    "masculinity": 0.999999,
    "long_term_orientation": 0.999999,
    "indulgence": 0.999999,
    "commonsense_morality": 0.999999,
    "deontology": 0.999999,
    "utilitarianism": 0.999999,
    "justice": 0.999999,
    "virtue_ethics": 0.999999,
    "ubuntu": 0.999999,
    "confucianism": 0.999999,
    "face": 0.999999,
    "karma": 0.999999,
    "honor": 0.999999,
    "spiritual": 0.999999,
    "fairness": 0.999999,
    "truthfulness": 0.999999,
    "toxicity": 0.999999,
    "harmfulness": 0.999999,
}

P0_BY_MODEL = {
    "Llama-2-7b-chat-hf": P0_LLAMA_2_7B_CHAT,
    "Meta-Llama-3-8B-Instruct": P0_LLAMA_3_8B_INSTRUCT,
}


def get_p0_map(steering_model_path: str) -> dict:
    """`steering_model_path` may be a full HF repo id (e.g.
    'meta-llama/Meta-Llama-3-8B-Instruct') or just its last path component;
    matches by substring against P0_BY_MODEL's keys either way."""
    model_name = steering_model_path.split("/")[-1]
    for key, p0_map in P0_BY_MODEL.items():
        if key in model_name:
            return p0_map
    return {}


def get_p0(value_slug: str, steering_model_path: str) -> float:
    return get_p0_map(steering_model_path).get(value_slug, DEFAULT_P0)

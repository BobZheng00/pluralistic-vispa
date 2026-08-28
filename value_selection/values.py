"""
VISPA value taxonomy (paper Table 6): 31 values across 5 conceptual origins.

Each entry is (slug, display_label, category).

`slug` matches the filename stem used for the context-controlled contrastive
datasets, data/value/{slug}_context_controlled.csv. It is kept exactly as-is
from those existing files, including two historical inconsistencies:
"self-direction" uses a hyphen rather than an underscore, and "spiritual" is
the slug for the "spirituality" value.

`display_label` matches the value_labels used in the existing
data/classified_values/*.json outputs, and is what should be passed as the
NLI candidate label text.
"""

SCHWARTZ_BASIC_HUMAN_VALUES = "Schwartz's Basic Human Values"
CULTURAL_DIMENSIONS = "Cultural Dimensions"
MORAL_THEORIES = "Moral Theories"
AI_SAFETY_RELATED = "AI Safety-Related Values"
NON_WEIRD_MORAL_CONSTRUCTS = "Non-WEIRD Moral Constructs"

ALL_VALUES = [
    # slug, display_label, category
    ("self-direction", "self-direction", SCHWARTZ_BASIC_HUMAN_VALUES),
    ("stimulation", "stimulation", SCHWARTZ_BASIC_HUMAN_VALUES),
    ("hedonism", "hedonism", SCHWARTZ_BASIC_HUMAN_VALUES),
    ("achievement", "achievement", SCHWARTZ_BASIC_HUMAN_VALUES),
    ("power", "power", SCHWARTZ_BASIC_HUMAN_VALUES),
    ("security", "security", SCHWARTZ_BASIC_HUMAN_VALUES),
    ("conformity", "conformity", SCHWARTZ_BASIC_HUMAN_VALUES),
    ("tradition", "tradition", SCHWARTZ_BASIC_HUMAN_VALUES),
    ("benevolence", "benevolence", SCHWARTZ_BASIC_HUMAN_VALUES),
    ("universalism", "universalism", SCHWARTZ_BASIC_HUMAN_VALUES),

    ("power_distance", "power distance", CULTURAL_DIMENSIONS),
    ("uncertainty_avoidance", "uncertainty avoidance", CULTURAL_DIMENSIONS),
    ("individualism", "individualism", CULTURAL_DIMENSIONS),
    ("masculinity", "masculinity", CULTURAL_DIMENSIONS),
    ("long_term_orientation", "long-term orientation", CULTURAL_DIMENSIONS),
    ("indulgence", "indulgence", CULTURAL_DIMENSIONS),

    ("commonsense_morality", "commonsense morality", MORAL_THEORIES),
    ("deontology", "deontology", MORAL_THEORIES),
    ("utilitarianism", "utilitarianism", MORAL_THEORIES),
    ("justice", "justice", MORAL_THEORIES),
    ("virtue_ethics", "virtue ethics", MORAL_THEORIES),
    ("ubuntu", "ubuntu", MORAL_THEORIES),
    ("confucianism", "confucianism", MORAL_THEORIES),

    ("fairness", "fairness", AI_SAFETY_RELATED),
    ("truthfulness", "truthfulness", AI_SAFETY_RELATED),
    ("toxicity", "toxicity", AI_SAFETY_RELATED),
    ("harmfulness", "harmfulness", AI_SAFETY_RELATED),

    ("face", "face", NON_WEIRD_MORAL_CONSTRUCTS),
    ("karma", "karma", NON_WEIRD_MORAL_CONSTRUCTS),
    ("honor", "honor", NON_WEIRD_MORAL_CONSTRUCTS),
    ("spiritual", "spirituality", NON_WEIRD_MORAL_CONSTRUCTS),
]

assert len(ALL_VALUES) == 31

VALUE_LABELS = [label for _, label, _ in ALL_VALUES]
SLUG_TO_LABEL = {slug: label for slug, label, _ in ALL_VALUES}
LABEL_TO_SLUG = {label: slug for slug, label, _ in ALL_VALUES}

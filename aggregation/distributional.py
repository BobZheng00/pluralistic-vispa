"""
Distributional mode aggregation (paper §3.3): for each selected value, get a
probability distribution over answer options conditioned on that value's
steered comment, then average across values for the final population-level
distribution. Prompts match the original modular_pluralism baseline
(ours_distributional.py) unchanged.
"""

from typing import List, Optional

from .aggregator import Aggregator


def build_prompt(comment: str, question: str, attribute: Optional[str] = None) -> str:
    if attribute:
        # GlobalOpinionQA: attribute is a country name
        return f"You are from the country of {attribute}, respond to the following instruction with the help of a passage.\n\nPassage: {comment}\n\n{question}"
    # MoralChoice
    return f"Please respond to the following instruction with the help of a passage.\n\nPassage: {comment}\n\n{question}"


def get_probability_distribution(
    comment: str,
    question: str,
    options: List[str],
    aggregator: Aggregator,
    attribute: Optional[str] = None,
) -> List[float]:
    prompt = build_prompt(comment, question, attribute=attribute)
    # The original script never generated text here, only ever used the
    # distribution — skip_generation avoids paying for a discarded
    # generation on every call (see Aggregator.option_probabilities).
    _, pred_distribution = aggregator.option_probabilities(prompt, options, skip_generation=True)
    return pred_distribution


def aggregate_distributions(distributions: List[List[float]]) -> List[float]:
    """Simple mean across per-value distributions (paper §3.3: 'the
    collection of value comment distributions are aggregated to derive a
    final distribution reflecting the population preference')."""
    if not distributions:
        return []

    num_options = len(distributions[0])
    total = [0.0] * num_options
    for dist in distributions:
        for i in range(num_options):
            total[i] += dist[i]
    return [x / len(distributions) for x in total]


def format_question_for_direct_answer(question: str) -> str:
    """Nudges the model to lead with the option letter, matching the
    original baseline's prompt-formatting convention for multiple-choice
    distributional tasks."""
    if "A." not in question:
        return question
    instruction = (
        "\n\nIMPORTANT: You must start your reply with the option letter followed by a period. "
        "Do NOT include any leading sentences before the option letter.\n"
    )
    if "Answer:" in question:
        question = question.split("Answer:")[0].strip()
    return question + instruction

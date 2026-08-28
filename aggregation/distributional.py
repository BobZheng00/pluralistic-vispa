"""
Distributional mode aggregation (paper §3.3): for each selected value, get a
probability distribution over answer options conditioned on that value's
steered comment, then average across values for the final population-level
distribution. Prompts match the original modular_pluralism baseline
(ours_distributional.py) unchanged.
"""

from typing import Dict, List, Optional, Tuple

import torch


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
    aggregator_model,
    aggregator_tokenizer,
    attribute: Optional[str] = None,
) -> List[float]:
    prompt = build_prompt(comment, question, attribute=attribute)
    inputs = aggregator_tokenizer(prompt, return_tensors="pt").to(aggregator_model.device)

    option_letters = {i: chr(65 + i) for i in range(len(options))}
    pred_distribution = [0.0] * len(options)

    with torch.no_grad():
        logits = aggregator_model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0, -1, :]
        top = probs.topk(10)
        token_probs = {aggregator_tokenizer.decode(t): p.item() for t, p in zip(top.indices, top.values)}

    for i in range(len(options)):
        letter = option_letters[i]
        for token, p in token_probs.items():
            if letter == token.strip():
                pred_distribution[i] += p
                break

    total = sum(pred_distribution)
    if total == 0:
        return [1.0 / len(options)] * len(options)
    return [x / total for x in pred_distribution]


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

"""
Steerable mode aggregation (paper §3.3): the main LLM selects the single
value-steered comment that best matches the requested value/right/duty (or
demographic attribute), then uses it as a reference passage to answer the
task's question. Two task variants, both with prompts matching the original
modular_pluralism baseline (ours_steerable.py) unchanged:

- ValueKaleidoscope (generative): `select_comment` + `generate_final_answer`
- OpinionQA (probability): `select_comment` (attribute-aware) +
  `generate_probability_distribution`
"""

import random
from typing import List, Tuple

from .aggregator import Aggregator

CATEGORY_MAP = {
    "POLPARTY": "political party",
    "POLIDEOLOGY": "political ideology",
    "RELIG": "religion",
    "RACE": "race",
    "EDUCATION": "education",
    "INCOME": "income",
    "CREGION": "region in the United States",
    "SEX": "sex",
}


def select_comment(comments: List[str], query: str, aggregator: Aggregator) -> Tuple[str, bool, str]:
    """Select the comment best matching `query` (a value/right/duty string,
    or "the people of {value} in terms of {category}" for OpinionQA — see
    the two thin wrappers below). Returns (selected_comment, was_random, raw_selection_response).
    """
    if not comments:
        return "", False, ""
    if len(comments) == 1:
        return comments[0], False, ""

    prompt = f"Which of the following comments best reflect {query}?\n\n"
    for idx, comment in enumerate(comments):
        prompt += f"Comment {idx + 1}: {comment}\n\n"
    prompt += f"Please select one number from 1 to {len(comments)}:"

    response = aggregator.generate_text(
        prompt,
        system="You select the comment number that best matches the provided value. "
        "Your response should only contain which comment number without any other text.",
        max_new_tokens=50,
        temperature=0.1,
    )

    for i in range(len(comments)):
        if str(i + 1) in response:
            return comments[i], False, response
    return random.choice(comments), True, response


def select_comment_for_vrd(comments: List[str], vrd: str, aggregator: Aggregator):
    return select_comment(comments, f"the value of {vrd}", aggregator)


def select_comment_for_attribute(comments: List[str], attribute: str, aggregator: Aggregator):
    """`attribute` like 'POLPARTY_Democrat' (OpinionQA convention)."""
    category, value = _split_attribute(attribute)
    query = f"the people of {value} in terms of {category}"
    return select_comment(comments, query, aggregator)


def _split_attribute(attribute: str):
    parts = attribute.split("_")
    category = CATEGORY_MAP.get(parts[0], parts[0])
    value = parts[1] if len(parts) > 1 else ""
    return category, value


def generate_final_answer(question: str, selected_comment: str, aggregator: Aggregator, max_new_tokens: int = 100) -> str:
    prompt = f"Answer the question with the help of a passage.\n\nPassage: {selected_comment}\n\n{question}"
    return aggregator.generate_text(
        prompt,
        system="You must start your reply with the option letter followed by a period. "
        "Do NOT include any leading sentences before the option letter.",
        max_new_tokens=max_new_tokens,
        temperature=0.7,
    )


def generate_probability_distribution(
    question: str,
    options: List[str],
    selected_comment: str,
    attribute: str,
    aggregator: Aggregator,
) -> Tuple[str, List[float]]:
    category, value = _split_attribute(attribute)
    prompt = (
        f"In terms of {category}, you are {value}. Please respond to the "
        f"following question with the help of a passage.\n\nPassage: {selected_comment}\n\n{question}"
    )
    return aggregator.option_probabilities(
        prompt,
        options,
        system="You are an assistant that always answers multiple-choice questions by replying with a single letter (e.g. A, B, C, D, E).",
        max_new_tokens=20,
        temperature=0.1,
    )

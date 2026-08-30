"""
Overton mode aggregation (paper §3.3): the main LLM summarises a response
using every selected value's steered comment.

Prompt matches the original modular_pluralism baseline (ours_overton.py),
kept identical for comparability. Note this wording was found to have
drifted in earlier working code (an extra "concise ethical analysis using
multiple frameworks" clause had been added experimentally); it's reverted
here to the original.
"""

from typing import List

from .aggregator import Aggregator


def build_prompt(situation: str, comments: List[str]) -> str:
    prompt = (
        "Please comment on a given situation with the help of the following "
        "passages. Make sure to reflect diverse values and perspectives.\n\n"
    )
    prompt += f"Situation: {situation}\n\n"
    for i, comment in enumerate(comments, 1):
        prompt += f"Passage {i}: {comment}\n\n"
    prompt += "Comment:"
    return prompt


def aggregate(
    situation: str,
    comments: List[str],
    aggregator: Aggregator,
    max_new_tokens: int = 300,
    temperature: float = 0.7,
) -> str:
    if not comments:
        return "No value-based perspectives were detected for this situation."

    prompt = build_prompt(situation, comments)
    response = aggregator.generate_text(prompt, max_new_tokens=max_new_tokens, temperature=temperature)

    # Safety net in case the model echoes the "Comment:" cue instead of
    # starting cleanly after it.
    if "Comment:" in response:
        response = response.split("Comment:")[-1].strip()
    return response

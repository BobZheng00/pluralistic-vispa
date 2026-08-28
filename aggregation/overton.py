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

import torch


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
    aggregator_model,
    aggregator_tokenizer,
    max_new_tokens: int = 300,
    temperature: float = 0.7,
) -> str:
    if not comments:
        return "No value-based perspectives were detected for this situation."

    prompt = build_prompt(situation, comments)

    if hasattr(aggregator_tokenizer, "apply_chat_template"):
        try:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = aggregator_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            formatted_prompt = prompt
    else:
        formatted_prompt = prompt

    inputs = aggregator_tokenizer(formatted_prompt, return_tensors="pt").to(aggregator_model.device)
    with torch.no_grad():
        outputs = aggregator_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=aggregator_tokenizer.eos_token_id,
        )
    response = aggregator_tokenizer.decode(outputs[0], skip_special_tokens=True)

    if formatted_prompt in response:
        response = response[len(formatted_prompt):].strip()
    elif "Comment:" in response:
        response = response.split("Comment:")[-1].strip()
    return response

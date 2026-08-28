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
from typing import Dict, List, Tuple

import torch

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


def _generate(aggregator_model, aggregator_tokenizer, formatted_prompt, max_new_tokens, temperature):
    inputs = aggregator_tokenizer(formatted_prompt, return_tensors="pt").to(aggregator_model.device)
    with torch.no_grad():
        outputs = aggregator_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=aggregator_tokenizer.eos_token_id,
        )
    generated_ids = outputs[0][len(inputs.input_ids[0]):]
    return aggregator_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def _format(aggregator_tokenizer, system_msg: str, prompt: str) -> str:
    if hasattr(aggregator_tokenizer, "apply_chat_template"):
        try:
            messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}]
            return aggregator_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return prompt
    return prompt


def select_comment(
    comments: List[str],
    query: str,
    aggregator_model,
    aggregator_tokenizer,
) -> Tuple[str, bool, str]:
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

    formatted_prompt = _format(
        aggregator_tokenizer,
        "You select the comment number that best matches the provided value. "
        "Your response should only contain which comment number without any other text.",
        prompt,
    )
    response = _generate(aggregator_model, aggregator_tokenizer, formatted_prompt, max_new_tokens=50, temperature=0.1)

    for i in range(len(comments)):
        if str(i + 1) in response:
            return comments[i], False, response
    return random.choice(comments), True, response


def select_comment_for_vrd(comments: List[str], vrd: str, aggregator_model, aggregator_tokenizer):
    return select_comment(comments, f"the value of {vrd}", aggregator_model, aggregator_tokenizer)


def select_comment_for_attribute(comments: List[str], attribute: str, aggregator_model, aggregator_tokenizer):
    """`attribute` like 'POLPARTY_Democrat' (OpinionQA convention)."""
    category, value = _split_attribute(attribute)
    query = f"the people of {value} in terms of {category}"
    return select_comment(comments, query, aggregator_model, aggregator_tokenizer)


def _split_attribute(attribute: str):
    parts = attribute.split("_")
    category = CATEGORY_MAP.get(parts[0], parts[0])
    value = parts[1] if len(parts) > 1 else ""
    return category, value


def generate_final_answer(
    question: str,
    selected_comment: str,
    aggregator_model,
    aggregator_tokenizer,
    max_new_tokens: int = 100,
) -> str:
    prompt = f"Answer the question with the help of a passage.\n\nPassage: {selected_comment}\n\n{question}"
    formatted_prompt = _format(
        aggregator_tokenizer,
        "You must start your reply with the option letter followed by a period. "
        "Do NOT include any leading sentences before the option letter.",
        prompt,
    )
    return _generate(aggregator_model, aggregator_tokenizer, formatted_prompt, max_new_tokens, temperature=0.7)


def generate_probability_distribution(
    question: str,
    options: List[str],
    selected_comment: str,
    attribute: str,
    aggregator_model,
    aggregator_tokenizer,
) -> Tuple[str, List[float]]:
    category, value = _split_attribute(attribute)
    prompt = (
        f"In terms of {category}, you are {value}. Please respond to the "
        f"following question with the help of a passage.\n\nPassage: {selected_comment}\n\n{question}"
    )

    # Probability extraction uses the RAW prompt (no chat template): a chat
    # template's special tokens shift what the next-token distribution over
    # option letters looks like, so this matches the original baseline's
    # lm_utils.py convention for probability-mode calls.
    inputs = aggregator_tokenizer(prompt, return_tensors="pt").to(aggregator_model.device)

    formatted_prompt = _format(
        aggregator_tokenizer,
        "You are an assistant that always answers multiple-choice questions by replying with a single letter (e.g. A, B, C, D, E).",
        prompt,
    )
    response = _generate(aggregator_model, aggregator_tokenizer, formatted_prompt, max_new_tokens=20, temperature=0.1)

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
        pred_distribution = [1.0 / len(options)] * len(options)
    else:
        pred_distribution = [x / total for x in pred_distribution]

    return response, pred_distribution

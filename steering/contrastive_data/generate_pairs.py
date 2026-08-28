#!/usr/bin/env python
"""
Context-controlled contrastive pair generation (paper Appendix B.1).

Stage 1 (positive generation) — for each of 4 (pronoun, structure) variants,
prompt the LLM:
    "Create 25 distinct {pronoun} {structure} that exemplify the promotion
    and embodiment of the '{value}' facet of {category}."
  where pronoun in {first-person, third-person} and structure in
  {sentences, questions}, yielding 100 positive examples per value.

Stage 2 (contrastive negative generation) — for each batch of positives,
prompt the LLM:
    "Produce a corresponding sentence for each of the above sentences that
    expresses opposition and resistance to the '{value}' aspect of
    {category}. The protagonist of the sentence should act in the opposite
    manner spontaneously. Strive to maintain the same grammar and sentence
    structure as much as possible."

This keeps positive/negative pairs matched on scenario, pronoun, and surface
structure, isolating the value signal (the point of "context-controlled").

Usage:
    export OPENAI_API_KEY="..."
    python generate_pairs.py --value benevolence --out ../../data/value
    python generate_pairs.py --all --out ../../data/value   # all 31 values
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "value_selection"))
from values import ALL_VALUES  # noqa: E402

DEFAULT_MODEL = "gpt-4o"
SAMPLES_PER_VARIANT = 25
PRONOUNS = ["first-person", "third-person"]
STRUCTURES = ["sentences", "questions"]

POSITIVE_PROMPT_TEMPLATE = (
    "Create {n} distinct {pronoun} {structure} that exemplify the promotion "
    "and embodiment of the '{value}' facet of {category}."
)

NEGATIVE_PROMPT_TEMPLATE = (
    "Produce a corresponding sentence for each of the following sentences "
    "that expresses opposition and resistance to the '{value}' aspect of "
    "{category}. The protagonist of the sentence should act in the opposite "
    "manner spontaneously. Strive to maintain the same grammar and sentence "
    "structure as much as possible, changing only what is necessary to "
    "reverse the value being expressed.\n\n{numbered_positives}"
)


def _call_llm(prompt: str, model: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
    )
    return response.choices[0].message.content


def _parse_numbered_list(text: str, expected_n: int):
    """Parses "1. ...\n2. ...\n" style LLM output into a list of strings."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    items = []
    for line in lines:
        match = re.match(r"^\d+[.)]\s*(.+)$", line)
        if match:
            items.append(match.group(1).strip().strip('"'))
    if len(items) != expected_n:
        raise ValueError(
            f"Expected {expected_n} numbered items, parsed {len(items)}. "
            f"Raw response:\n{text}"
        )
    return items


def generate_pairs_for_value(value_label: str, category: str, model: str = DEFAULT_MODEL):
    """Returns a list of (positive, negative) sentence/question pairs for one value."""
    all_pairs = []

    for pronoun in PRONOUNS:
        for structure in STRUCTURES:
            pos_prompt = POSITIVE_PROMPT_TEMPLATE.format(
                n=SAMPLES_PER_VARIANT, pronoun=pronoun, structure=structure,
                value=value_label, category=category,
            )
            pos_text = _call_llm(pos_prompt, model)
            positives = _parse_numbered_list(pos_text, SAMPLES_PER_VARIANT)

            numbered_positives = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(positives))
            neg_prompt = NEGATIVE_PROMPT_TEMPLATE.format(
                value=value_label, category=category, numbered_positives=numbered_positives,
            )
            neg_text = _call_llm(neg_prompt, model)
            negatives = _parse_numbered_list(neg_text, SAMPLES_PER_VARIANT)

            all_pairs.extend(zip(positives, negatives))

    return all_pairs


def write_csv(pairs, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question_1", "question_2"])
        writer.writerows(pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--value", help="single value slug to generate, e.g. 'benevolence' (see value_selection/values.py)")
    group.add_argument("--all", action="store_true", help="generate for all 31 values")
    parser.add_argument("--out", default="data/value", help="output directory for {slug}_context_controlled.csv files")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    if args.all:
        targets = ALL_VALUES
    else:
        targets = [v for v in ALL_VALUES if v[0] == args.value]
        if not targets:
            raise SystemExit(f"Unknown value slug '{args.value}'. See value_selection/values.py for valid slugs.")

    out_dir = Path(args.out)
    for slug, label, category in targets:
        out_path = out_dir / f"{slug}_context_controlled.csv"
        if out_path.exists():
            print(f"[{slug}] {out_path} already exists, skipping (delete it to regenerate)")
            continue
        print(f"[{slug}] generating contrastive pairs for '{label}' ({category})...")
        pairs = generate_pairs_for_value(label, category, model=args.model)
        write_csv(pairs, out_path)
        print(f"[{slug}] wrote {len(pairs)} pairs to {out_path}")


if __name__ == "__main__":
    main()

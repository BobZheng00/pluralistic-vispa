"""
Value-steered "comment" generation, shared across all three pluralistic
alignment modes (paper §3.2, feeding into §3.3's aggregation). Given a
situation/question and a set of already-fit per-value steering states (fit
once per value from its contrastive pairs — see steering/interface.py and
steering/README.md — and reused across every sample that selects that
value), generates one steered comment per selected value.

Prompt: "Please comment on the following situation/question: {text}" is used
for Overton and Distributional mode comments. Steerable mode additionally
supports naming the value/right/duty explicitly (paper Appendix E.8, Table
23's "open-ended" prompt — this is the one used in all VISPA's own results;
the discrete-choice variant in that table is a baseline-only comparison
point, not something VISPA generates with).
"""

from typing import Dict, Optional


def build_comment_prompt(text: str, vrd: Optional[str] = None) -> str:
    if vrd:
        return (
            f"Please comment on whether {vrd} supports, opposes, or applies "
            f"to the following situation:\n\n{text.strip()}\nAnswer:"
        )
    return f"Please comment on the following situation: {text.strip()}"


def generate_value_comments(
    backend,
    model,
    tokenizer,
    text: str,
    value_states: Dict[str, object],
    vrd: Optional[str] = None,
    **gen_kwargs,
) -> Dict[str, str]:
    """Returns {value_slug: comment}, omitting any value whose backend
    gated the generation out (e.g. ProbeCalibratedSteering with use_gate=True
    when the value isn't meaningfully present for this input)."""
    prompt = build_comment_prompt(text, vrd=vrd)

    comments = {}
    for value_slug, state in value_states.items():
        comment = backend.generate(model, tokenizer, prompt, state, **gen_kwargs)
        if comment:
            comments[value_slug] = comment.strip()
    return comments

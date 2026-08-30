"""
Unified aggregator interface (paper §3.3): "Unlike the steering model, which
requires open-weights for internal activation access... the aggregation
model only consumes the generated comments and can therefore be
closed-weight (e.g., ChatGPT)." Table 8 lists GPT-3.5-turbo and GPT-4o as
aggregation-only models.

Two implementations of the same interface so aggregation/overton.py,
steerable.py, and distributional.py don't need to know which kind of model
they're calling: `HFAggregator` wraps a local HF causal LM (used for the
open-weight aggregators, e.g. LLaMA2-13B), `OpenAIAggregator` wraps the
OpenAI chat completions API (used for GPT-3.5-turbo / GPT-4o).
"""

import math
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch


class Aggregator(ABC):
    @abstractmethod
    def generate_text(
        self, prompt: str, system: Optional[str] = None, max_new_tokens: int = 200, temperature: float = 0.7
    ) -> str:
        """Free-form generation: Overton summarization, Steerable comment
        selection and final-answer generation."""

    @abstractmethod
    def option_probabilities(
        self,
        prompt: str,
        options: List[str],
        system: Optional[str] = None,
        max_new_tokens: int = 20,
        temperature: float = 0.1,
    ) -> Tuple[str, List[float]]:
        """Returns (raw text response, normalized probability per option),
        matching each option to the model's predicted probability of
        leading its response with that option's letter (A, B, C, ...) —
        used by Steerable/OpinionQA and Distributional mode."""


def _option_letters(options: List[str]) -> dict:
    return {i: chr(65 + i) for i in range(len(options))}


def _normalize(pred: List[float]) -> List[float]:
    total = sum(pred)
    if total == 0:
        return [1.0 / len(pred)] * len(pred)
    return [x / total for x in pred]


class HFAggregator(Aggregator):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _format(self, prompt: str, system: Optional[str]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                messages = ([{"role": "system", "content": system}] if system else []) + [
                    {"role": "user", "content": prompt}
                ]
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                return prompt
        return prompt

    def generate_text(self, prompt, system=None, max_new_tokens=200, temperature=0.7):
        formatted = self._format(prompt, system)
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def option_probabilities(self, prompt, options, system=None, max_new_tokens=20, temperature=0.1):
        response = self.generate_text(prompt, system=system, max_new_tokens=max_new_tokens, temperature=temperature)

        # Probability extraction uses the RAW prompt (no chat template): a
        # chat template's special tokens shift what the next-token
        # distribution over option letters looks like, matching the
        # original baseline's probability-mode convention.
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        option_letters = _option_letters(options)
        pred = [0.0] * len(options)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0, -1, :]
            top = probs.topk(10)
            token_probs = {self.tokenizer.decode(t): p.item() for t, p in zip(top.indices, top.values)}
        for i, letter in option_letters.items():
            for token, p in token_probs.items():
                if letter == token.strip():
                    pred[i] += p
                    break
        return response, _normalize(pred)


class OpenAIAggregator(Aggregator):
    def __init__(self, model_name: str, client=None):
        from openai import OpenAI

        self.model_name = model_name
        self.client = client or OpenAI()

    def _messages(self, prompt: str, system: Optional[str]):
        return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]

    def generate_text(self, prompt, system=None, max_new_tokens=200, temperature=0.7):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self._messages(prompt, system),
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        return (response.choices[0].message.content or "").strip()

    def option_probabilities(self, prompt, options, system=None, max_new_tokens=20, temperature=0.1):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self._messages(prompt, system),
            max_tokens=max_new_tokens,
            temperature=temperature,
            logprobs=True,
            top_logprobs=20,
        )
        text = (response.choices[0].message.content or "").strip()

        option_letters = _option_letters(options)
        pred = [0.0] * len(options)
        try:
            first_token_logprobs = response.choices[0].logprobs.content[0].top_logprobs
            token_probs = {entry.token: math.exp(entry.logprob) for entry in first_token_logprobs}
        except (AttributeError, IndexError, TypeError):
            token_probs = {}
        for i, letter in option_letters.items():
            for token, p in token_probs.items():
                if letter == token.strip():
                    pred[i] += p
                    break
        return text, _normalize(pred)


def is_openai_model(model_path: str) -> bool:
    """Every HF checkpoint in Table 8 is an "org/model" path; OpenAI's API
    model ids (e.g. "gpt-4o", "gpt-3.5-turbo") never contain a slash."""
    return "/" not in model_path

"""
Common interface for the three activation-level steering instantiations
described in the VISPA paper (Section 3.2): projection-based, averaging-based
(CAA), and probe-calibrated. All three implement the same shared operation

    h_hat[l, t] = h[l, t] + lambda_V * v_V                              (Eq. 1)

and differ only in (i) how v_V is estimated from a value's context-controlled
contrastive pairs, and (ii) how lambda_V (or, for probe-calibrated steering,
a per-input epsilon_V(x)) is chosen. This module fixes only the calling
convention — fit on contrastive pairs, then steer a generation call — so the
three backends stay swappable from aggregation/ and eval/ code without those
callers needing to know which one is in use.

Each concrete backend is a thin bridge to an external, non-vendored codebase
(see steering/README.md and scripts/setup_dependencies.*); this file has no
external dependency of its own.
"""

from abc import ABC, abstractmethod
from typing import Any, List


class SteeringBackend(ABC):
    name: str

    @abstractmethod
    def fit(
        self,
        model,
        tokenizer,
        value_slug: str,
        pos_prompts: List[str],
        neg_prompts: List[str],
        layers: List[int],
        **kwargs,
    ) -> Any:
        """Estimate a steering direction (and any backend-specific state) for
        `value_slug` from its context-controlled positive/negative prompts.
        Returns opaque state to pass into `generate`. `**kwargs` carries
        backend-specific fitting options; backends that don't use one just
        ignore it.
        """

    @abstractmethod
    def generate(self, model, tokenizer, prompt: str, state: Any, **gen_kwargs) -> str:
        """Generate a continuation for `prompt` with the value direction(s)
        captured in `state` applied at the configured layers.
        """

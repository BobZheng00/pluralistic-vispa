# steering/

Implements the three activation-level value steering instantiations from
Section 3.2 of the paper, behind the shared `SteeringBackend` interface in
[`interface.py`](interface.py):

| File | Instantiation | Bridges to | License upstream |
|---|---|---|---|
| `probe_calibrated.py` | Probe-calibrated (paper's main method, §3.2.3) | [hr-jin/ConVA](https://github.com/hr-jin/ConVA) | none (all-rights-reserved by default) |
| `averaging_caa.py` | Averaging-based / CAA-style (§3.2.2) | ConVA's own `src/baselines/caa.py` — **not** nrimsky/CAA directly | none |
| `projection_pca.py` | Projection-based / PCA (§3.2.1) | [andyzoujm/representation-engineering](https://github.com/andyzoujm/representation-engineering) | MIT |

`contrastive_data/generate_pairs.py` builds the context-controlled positive/
negative pairs (§Appendix B.1) that all three backends train on.

## Why nothing here is vendored

ConVA ships no LICENSE file, which defaults to all-rights-reserved —
copying its code into this repo isn't something we have clear permission to
do. Rather than have one rule for it and another for RepE (which is MIT),
both external codebases get the same treatment for consistency: nothing is
copied in. `scripts/setup_dependencies.sh` (or `.ps1`) clones each at the
pinned commit this repo was developed against into a gitignored
`third_party/`, and the bridge modules here only ever `import` from that
directory at runtime — see `_third_party.py`.

This means:
- `git clone` alone does **not** get you a runnable repo. Run
  `scripts/setup_dependencies.sh` first.
- Pinned commits: ConVA `9484868`, representation-engineering `5455d8a`.
- If you use `averaging_caa.py`'s results, cite both ConVA (Jin et al., ACL
  2025) and Rimsky et al. (ACL 2024) — see the module docstring for why.

## Common interface

```python
backend = ProbeCalibratedSteering(p0=0.9)          # fixed P0; or AveragingCAASteering() / ProjectionPCASteering()
state = backend.fit(model, tokenizer, "benevolence", pos_prompts, neg_prompts, layers=range(10, 26))
text = backend.generate(model, tokenizer, prompt, state, max_new_tokens=200)
```

The shared defaults use coefficient 1.0 for projection, ConVA's value-specific
coefficients for its ten Schwartz values and 0.5 otherwise for averaging, and
one fixed P0 of 0.9 for probe-calibrated steering. Layer numbers are zero-based
physical transformer blocks for every backend, including RepE.

`model` is expected to be ConVA's `src.modelwrapper.ModelWrapper` for the two
ConVA-backed instantiations, and a plain HF `AutoModelForCausalLM` for
`projection_pca.py` (RepE wraps it internally via its own pipelines). This
asymmetry comes from the upstream libraries, not from this repo's own design
— see each module's docstring before mixing model objects across backends.

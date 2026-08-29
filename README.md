# VISPA: Pluralistic Alignment via Automatic Value Selection and Activation

**arXiv (Pre-print) Paper Link:** [link](https://arxiv.org/abs/2601.12758).

Official code for **VISPA** (Value-Integrated Steering for Pluralistic
Alignment), a training-free framework that automatically selects
scenario-relevant human values from a 31-value pool and steers an LLM's
internal activations to produce value-conditioned responses, composed into
three pluralistic alignment modes: `Overton`, `Steerable`, and
`Distributional`.

## Setup

```bash
git clone <this-repo>
cd pluralistic-vispa
pip install -r requirements.txt

# Fetches ConVA and representation-engineering into a gitignored third_party/,
# pinned to the commits this repo was built against. Required before using
# anything in steering/.
bash scripts/setup_dependencies.sh      # or scripts/setup_dependencies.ps1 on Windows

export OPENAI_API_KEY="..."             # needed for value selection's aggregation-stage LLM calls and contrastive data generation
```

## Repository layout

```
value_selection/          Section 3.1 — the 31-value taxonomy and the NLI-based relevance gate
steering/                  Section 3.2 — three steering instantiations, bridged to external libraries (see steering/README.md),
                           plus shared value-steered comment generation (comment_generation.py)
aggregation/               Section 3.3 — Overton/Steerable/Distributional main-LLM aggregation over per-value comments
pipeline/                  End-to-end run_overton.py / run_steerable.py / run_distributional.py wiring the above together
eval/                      Section 4.3 metrics, reading pipeline/run_*.py output directly (see eval/README.md)
data/value/                The 31 context-controlled contrastive datasets (this repo's own data)
data/classified_values/    Example value-selection outputs (paper §3.1)
scripts/                   Setup and environment scripts
```

## Running a mode end-to-end

```bash
# 1. Value selection (once per dataset)
python value_selection/gate.py -i path/to/dataset.json -o gate_output.json

# 2. Full pipeline for that mode
python pipeline/run_overton.py \
    --input path/to/overton_test_valuekaleidoscope.json \
    --selection gate_output.json \
    --output results/overton_results.json
# run_steerable.py and run_distributional.py take an additional --task_type
# (valuekaleidoscope/opinionqa, moralchoice/globalopinionqa respectively)

# 3. Evaluate (paper §4.3 metrics; reads pipeline output directly, see eval/README.md)
python eval/eval_overton.py --input_file results/overton_results.json
```

All three `pipeline/run_*.py` scripts default to `--backend probe_calibrated`
(the paper's main method); pass `--backend averaging_caa` or
`--backend projection_pca` to compare instantiations.

## Dependencies

This repository does not vendor code from ConVA or modular_pluralism, as
neither distributes a license. `scripts/setup_dependencies.sh` (or `.ps1`
on Windows) clones ConVA, representation-engineering, and modular_pluralism
at pinned commits into a local `third_party/` directory (gitignored); this
repository's code imports from `third_party/` at runtime rather than
including copies of it. See `steering/README.md` for the pinned commits and
per-module attribution.

## Citation

```bibtex
@article{Zheng2026VISPA,
  title={VISPA: Pluralistic Alignment via Automatic Value Selection and Activation},
  author={Zheng, Shenyan and Zhong, Jiayou and Shetty, Anudeex and Ji, Heng and Nakov, Preslav and Naseem, Usman},
  journal={arXiv preprint arXiv:2601.12758},
  year={2026}
}
```

<!-- TODO: replace with the EMNLP 2026 (Main) proceedings citation once available. -->

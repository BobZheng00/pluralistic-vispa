# data/

## `value/`

The 31 context-controlled contrastive datasets, `{slug}_context_controlled.csv`
(one per value in `value_selection/values.py`), each with `question_1`
(positive: strongly expresses the value) / `question_2` (negative: opposes
it) columns. Generated via `steering/contrastive_data/generate_pairs.py`
(paper Appendix B.1). This is VISPA's own generated data, not third-party
code, so unlike `steering/` these files are committed directly rather than
fetched via `scripts/setup_dependencies.*`.

## `classified_values/`

Value-selection outputs (paper §3.1) from `value_selection/gate.py`, for
both the ModPlural-derived splits and the VITAL-derived splits (`vital_*.json`).

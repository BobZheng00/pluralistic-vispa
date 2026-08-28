# eval/

Evaluation scripts matching the metrics in paper §4.3, reading
`pipeline/run_*.py` output directly — no field-name translation needed.

| Script | Mode(s) | Metric | Expects on each item |
|---|---|---|---|
| `eval_overton.py` | Overton | NLI value coverage, accuracy@0.33 (default) | `output`, `vrd`, `explanation` |
| `eval_steerable.py` | Steerable / ValueKaleidoscope | 3-way + binary accuracy | `output`, `label` (A/B/C) |
| `eval_distributional.py` | Distributional (MoralChoice/GlobalOpinionQA), Steerable / OpinionQA | Jensen-Shannon distance, most-likely-correctness | `pred_distribution`, `gold_distribution`, `attribute` |

```bash
python eval_overton.py --input_file ../pipeline_results/overton_results.json
python eval_steerable.py --input_file ../pipeline_results/steerable_vk_results.json
python eval_distributional.py --input_file ../pipeline_results/distributional_moralchoice_results.json --dataset_type moralchoice
python eval_distributional.py --input_file ../pipeline_results/steerable_opinionqa_results.json --dataset_type steerable_opinionqa
```

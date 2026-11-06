# Reproduction

Run commands from the repository root after `python -m pip install -e ".[test]"`. Store licensed source data and run outputs outside the Git checkout. All examples below use private paths as placeholders.

## Prepare data

```bash
evidrug resources --output resources
evidrug prepare --dataset mimic_iv_full --input /private/mimic-iv-3.1.zip --output /private/data/mimic_iv_full
evidrug prepare --dataset nwicu --input /private/nwicu.zip --output /private/data/nwicu
evidrug prepare --dataset sicdb --input /private/sicdb-1.0.8 --output /private/data/sicdb
```

Each cohort is written as `patients.pkl` with a neighboring aggregate metadata JSON. A cohort is partitioned at patient level using split seed 0. The frozen cohort counts and hashes are recorded in `configs/cohorts.json`. Source-code hashes, timestamps and pickle bytes can change after packaging; eligibility, patient ordering and train-only vocabulary rules define the semantic reproduction target.

## Three-stage model training

```bash
evidrug train --dataset nwicu --data /private/data/nwicu/patients.pkl \
  --output /private/runs/nwicu/full/seed0 --seed 0 --device cuda:0
```

The stages are backbone training, evidence refinement and DDI-feedback optimization. Each stage has 20 epochs and a fresh AdamW optimizer. The backbone is frozen after stage one. The default batch size is 64. Defaults are defined in `src/evidrug/config.py`; matching study snapshots are in `configs/model.json` and `configs/training.json`.

Backbone/refinement checkpoints maximize validation AP. The final stage chooses the lowest-DDI validation checkpoint that preserves AP, F1, Jaccard and medication-count tolerances. The initial refinement checkpoint is included as epoch zero. DDI feedback uses a fixed training monitor, never test data.

## Component ablations

| `--arm` | Intervention |
|---|---|
| `full` | Complete selected EviDrug |
| `no_history` | Current encounter only, with state-only retrieval |
| `no_memory` | Remove donor parameters and donor computation |
| `no_set` | Remove the medication-set block and its output head |
| `no_clinical` | Remove clinical association projection and its gate channels |

Train each arm independently with seeds 0, 1 and 2. `no_history` preserves target identities and patient-balanced sampling while removing the query history from encoding and retrieval. All other budgets and selectors remain matched.

## Matched DDI controls

For `continue_zero`, `fixed_005`, `fixed_010` and `fixed`, start from the paired full-model refinement checkpoint:

```bash
evidrug train --dataset nwicu --data /private/data/nwicu/patients.pkl \
  --output /private/runs/nwicu/fixed_010/seed0 --seed 0 --device cuda:0 \
  --continuation fixed_010 --initial-refinement /private/runs/nwicu/full/seed0/refinement/best.pt
```

The fixed coefficients are 0, 0.05, 0.1 and 1.0, respectively. The dynamic run starts at 1.0 and updates once per epoch with projected feedback. Controls use the same continuation seed, initialization, epochs and validation rules.

## Launch the run grid

```bash
python scripts/run_study.py --data-root /private/data --output-root /private/runs \
  --device cuda:0 --ablations --controls --dry-run
```

Remove `--dry-run` to execute training sequentially on the specified device. The default datasets are all three cohorts and the default seeds are 0/1/2. Completed runs are skipped. The launcher trains and does not evaluate test queries.

## Freeze, test and replay

```bash
evidrug freeze --runs /private/runs
evidrug evaluate --data /private/data/nwicu/patients.pkl \
  --run /private/runs/nwicu/full/seed0 --freeze /private/runs/TEST_FREEZE.json --device cuda:0
python scripts/recompute_metrics.py --predictions /private/runs/nwicu/full/seed0/test_predictions.npz \
  --data /private/data/nwicu/patients.pkl --resources resources
python scripts/summarize_runs.py --runs /private/runs --output /private/summary.csv
python scripts/analyze_predictions.py --data /private/data/nwicu/patients.pkl \
  --predictions /private/runs/nwicu/full/seed0/test_predictions.npz --output /private/analysis
```

The freeze hashes both selected metadata and checkpoint bytes. Evaluation rejects modified selections or a different patient split. Learned parameters, retrieval arrays and patient-level predictions remain local.

## Efficiency

```bash
evidrug benchmark --data /private/data/nwicu/patients.pkl --run /private/runs/nwicu/full/seed0 \
  --device cpu --threads 1 --batch-size 16 --patients 128 --passes 3 --output /private/efficiency.json
```

The fixed training index is built before timing. Each timed pass includes online retrieval, serialization, clinical association lookup, the model and probability output. No query cache is used. Timings from the public refactor are separate from the frozen historical measurements in `results/efficiency_measurements.csv`.

## Historical checkpoints

`evidrug.checkpoints.migrate_study_state` maps parameter names from the frozen study implementation to this package. It changes no tensor values. The refactor is checked against the original model's evidence tensors, logits, parameter counts and gradients; see `verification/refactor_parity.json`.

Fresh training preserves the model equations and study protocol. Constructor cleanup removes historical modules and changes random-number consumption, so it does not claim byte-identical historical initialization or learned checkpoints. Frozen result exports retain their original checkpoint and prediction hashes.


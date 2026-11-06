# Results and manuscript mapping

All files here describe the frozen `clinical_memory_dynamic_20261104` study. The original source experiment directories and clinical arrays are not necessary for the aggregate audit; the audit runs directly from this repository. `PROVENANCE.json` identifies the original frozen result/protocol/measurement files by SHA256. It does not imply that patient-level metrics can be reconstructed from cohort scalars alone.

## Manuscript experiments

| Manuscript item | Role | Exported evidence |
|---|---|---|
| Dataset table | Cohort and task definition | `configs/cohorts.json`, `docs/data.md` |
| Main published-method comparison | AP, F1, Jaccard and DDI within each cohort | `results/test_metrics_by_seed.csv`, `aggregate_metrics.json`, `baseline_registry.csv` |
| Efficiency table | Complete-model parameters and B=16 CPU latency on three cohorts | `results/efficiency_measurements.csv`, `configs/efficiency_protocol.json` |
| Backbone/refinement/DDI-feedback comparison | Distinguish evidence refinement from additional backbone training | Stage records in `results/test_metrics_by_seed.csv` |
| Major-component table | History, case memory, set block and association contributions before/after feedback | Arms and stages in `test_metrics_by_seed.csv` |
| Fixed/dynamic coefficient table | Matched initialization, budget and selector for DDI control | `continue_zero`, `fixed_005`, `fixed_010`, `fixed`, `full` continuation rows |
| Continued/new-to-record table and scatter plot | History-union candidate-partition micro-F1 | `results/analysis/continued_new.csv` |
| History/support analysis figure | Connect benefits to history depth and training association support | `history_depth.csv`, `clinical_support.csv` |
| High-burden figure and pair-count table | AP versus interaction proxy and numerator counts in a shared stratum | `prior_burden.csv` |

Table/figure numbers follow the manuscript build; the definitions above identify their contents independently of numbering. The [supplementary PDF](../supplementary/EviDrug_Supplementary.pdf) and [supporting tables](../supplementary/README.md) provide the complete seed-level comparisons, data protocol and medication mappings.

## Scalar row inventory

`results/test_metrics_by_seed.csv` has **180 rows**: 144 EviDrug stage/arm/seed records and 36 existing baseline records (12 methods × 3 cohorts). These are not 180 independently trained full models. Some stages reuse an exact backbone or refinement initialization according to the study design. Main selections comprise nine checkpoints: three cohorts × seeds 0/1/2.

`results/selected_models.json` identifies these nine final main models, their parameter counts, selected epochs, initialization/backbone hashes and test scalar results. `aggregate_metrics.json` preserves the original per-arm aggregates. Use the checkpoint hash to identify the corresponding raw scalar row, because an arm can have several stages.

`main_training_history.csv` contains validation history for main-model DDI-feedback optimization and training-feedback scalar summaries. Validation values document selection; they are not additional test results or a main comparison table. `baseline_registry.csv` records each baseline's year, venue, paper link, single optimization seed, training/auxiliary budgets, decision rule and checkpoint/prediction hashes.

`efficiency_measurements.csv` has **39 rows**: 13 methods × 3 cohorts, each with three measured pass durations. Its latency column is the median pass time per patient. Startup and warm-up are identified separately. `configs/measured_environment.json` preserves the measured hardware/software environment.

## Additional aggregate analyses

The package retains the complete set of frozen aggregate analyses for inspection and replotting, including quantities not selected for manuscript display:

| CSV under `results/analysis/` | Contents |
|---|---|
| `continued_new.csv` | Micro precision/recall/F1, support and TP/FP/FN for continued/new candidates |
| `prior_burden.csv` | Burden and prior-DDI strata, primary metrics and interacting pairs per patient |
| `history_depth.csv` | Primary metrics in shared history-depth groups |
| `clinical_support.csv` | Pooled candidate AP in training diagnosis/medication support bins |
| `medication_frequency.csv` | Training-frequency quartiles and candidate metrics |
| `paired_bootstrap.csv` | Patient-paired conditional intervals and point differences |
| `thresholds.csv` | Fixed-model threshold sweeps |
| `equal_k.csv` | Fixed-model equal-prescription-size controls |

These are cohort or group aggregates. No patient example file, patient identifier list, patient-order array, retrieval bank or per-patient prediction matrix is shipped. Exact group construction and metric meanings are given in `docs/experiments.md` and the original analysis source.

## Verification

`python scripts/audit_release.py` checks every manifested file, reconstructs main means/sample SDs, matches checkpoint-identified scalar rows and verifies all 39 latency medians. Its reports are written to `checks/`.

`python -m pytest` exercises synthetic three-stage training, all five model arms, a matched coefficient control, separate frozen evaluation, modified-selection rejection, analytical-objective properties and source-table preparation rules. `verification/refactor_parity.json` records equivalence checks against the archived implementation and all nine main study checkpoints using generated inputs.

These checks verify software and exported evidence. Patient-level metric replay requires authorized source data and local prediction arrays.

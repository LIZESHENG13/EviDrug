# Experiment and measurement definitions

## Stage budgets and optimization

All parameters are learned from the cohort. The selected backbone has no pretrained clinical encoder and no clinical Transformer layers. Each optimization seed 0/1/2 receives 20 backbone epochs, 20 fixed-backbone refinement epochs and 20 DDI-feedback optimization epochs. The partition seed remains 0. Batch size is 64 and arithmetic is FP32.

Backbone AdamW groups use learning rates `5e-5` for the structured embedding encoder and `7e-4` for the remaining backbone. Refinement and DDI-feedback optimization use `5e-4` on the trainable head. Weight decay is `1e-3`, global gradient-norm clipping is 5, and each stage starts a fresh optimizer. Refinement initializes with seed `seed + 4100`; DDI optimization uses `seed + 7100`. Training hardware is recorded in `configs/tested_environment.json`.

One prefix per training patient is sampled per epoch by the original `PatientVisitSampler`, cycling through that patient's eligible queries with fixed seed-specific cycles. This gives patients equal sampling weight despite unequal history depth. Validation/test predict each patient's final eligible encounter. The feedback monitor fixes one training query per patient from the epoch-1 sampling schedule.

## Checkpoints and feedback

Backbone and refinement choose maximum validation AP, preferring the earliest epoch on a tie. DDI optimization includes the initial refinement model as epoch 0 and uses that same arm's refinement model as the quality reference. An eligible epoch satisfies:

| Quantity | Requirement relative to reference validation result |
|---|---|
| AP | No more than 0.002 lower |
| F1 | No more than 0.005 lower |
| Jaccard | No more than 0.005 lower |
| Mean predicted medication count | Between 0.9 and 1.1 times reference |

Among eligible epochs, minimize hard validation DDI; break ties by higher AP, then earlier epoch. The initial checkpoint guarantees an eligible candidate. This is checkpoint selection, separate from training feedback.

Dynamic feedback starts at coefficient 1, uses step 20, clips to `[0, 10]` and targets 90% of initial hard training-monitor DDI. It updates once per epoch using training outcomes only. Smooth selections use threshold 0.3 in logit space and temperature 0.25. If initial monitor DDI is zero, the original rule disables the feedback penalty. The fixed controls use 0, 0.05, 0.1 and 1 with the paired seed-specific refinement initialization and the same 20-epoch optimization budget and validation selection rules.

## Component and stage comparisons

The stage comparison distinguishes the backbone, the frozen-backbone evidence refinement, an additional 20 epochs of backbone training, and DDI-feedback optimization. These comparisons test different hypotheses: evidence use, extra training time and interaction control.

The four major removals are query history, training-case memory, the medication-set block and clinical association statistics. History/memory/set removals use their independently trained backbone; association removal shares the full backbone. Evaluate the refinement outputs before feedback and the final outputs after feedback. This distinguishes predictive component contributions from changes in the selected interaction operating point. All published main/ablation rows use the selected model without a medication-graph representation branch.

The scalar export retains all stage records, not just manuscript display rows. `configs/study_design.json` records exact intervention descriptions; `results/test_metrics_by_seed.csv` preserves arm/stage/seed and checkpoint/prediction hashes.

## Primary metrics

AP is average precision across medication candidates for each patient and then averaged across patients. Tied scores are evaluated as groups. F1 and Jaccard are also patient averages, with empty denominators floored at 1. Ordinary set selection uses probability at least 0.3. COGNet retains its recorded native decoding rule.

Hard DDI pools unordered predicted pairs across patients:

```text
DDI = sum_i interacting_unordered_pairs_i
      / max(sum_i total_unordered_predicted_pairs_i, 1).
```

It is not an average of per-patient interaction rates. An interacting-pair count per patient is a separate numerator-based measure, and can differ from the rate when prescription sizes change. The DDI graph is a fixed ATC3 proxy; the evaluation does not measure adverse events, doses, routes or simultaneous exposure.

The main table gives the mean and sample SD over three EviDrug optimization seeds on one partition. Each existing baseline has one seed and its own documented budget; these are not matched-seed baseline comparisons. The 12 baselines' source papers, exact budgets, auxiliary epochs and native decision rules are recorded in `results/baseline_registry.csv`.

## Clinical and sensitivity analyses

**Continued/new-to-record.** For each patient, take the union of medication types across all preceding retained visits. Continued candidates belong to that union; new-to-record candidates do not. Compute micro-precision, recall and F1 over patient-medication pairs separately for these two partitions. `Avg F1` is their arithmetic mean. These are observed-set categories, not verified clinical continuation/initiation events, and differ from the backbone anchor's immediately preceding visit.

**History and burden.** History-depth bins are 1–2, 3–5 and at least 6 preceding visits. Prior-medication burden uses the immediately preceding encounter's medication count, with bins 0–5, 6–10 and at least 11. Prior-DDI strata indicate whether the preceding set contains an edge in the fixed graph. Groups with fewer than 30 patients are flagged as descriptive. Group membership is shared across methods.

**Medication frequency.** Training-visit medication frequencies define quartile cutoffs. Assign tied counts with `searchsorted(..., side='left')`. Within each candidate stratum, report micro classification metrics and patient-average candidate AP. No test-label frequencies define the bins.

**Association support.** For a patient-medication candidate, support is the maximum training co-occurrence count across the current diagnosis codes. Bins are 0, 1–2, 3–9 and at least 10. AP pools candidate pairs within each bin; this is explicitly pooled candidate AP, not the patient-average primary AP. Compare full versus association removal with shared patient/candidate strata.

**Threshold and prescription-size controls.** Evaluate thresholds 0.1 through 0.9 without changing trained weights. Equal-size top-k values come from training prescription-size quantiles 0.25/0.5/0.75/0.9, rounded to integers at least 1 and deduplicated. Use a stable descending score order. These are sensitivity analyses, not test-based threshold selection.

**Uncertainty.** Patient-paired bootstrap uses 2,000 multinomial patient resamples and seed 20261103. Reuse each resample across the compared models; average results over the three fitted EviDrug seeds and hold the existing baseline seed fixed. Export 2.5th/97.5th percentiles. These intervals condition on those fitted models and do not estimate baseline seed-to-seed uncertainty. They do not turn a retrospective development comparison into an independent blind test.

## Efficiency protocol

The efficiency table compares all 13 methods on every cohort. Parameter counts include every retained model parameter; MetaDrug includes its uncertainty-filtering module. No head-only or backbone-only count replaces a complete-model count.

The shared setting is FP32, one CPU thread, batch size 16 and 128 validation patients per cohort, with one warm-up and three measured complete passes. Report median pass seconds divided by 128, in ms/patient. The first pass uses the listed method order, the second reverses it and the third rotates it by six positions. Hardware/software details and all measured pass durations are exported.

Timed work includes query retrieval where used, clinical features, collation, prediction, native support adaptation/decoding and final set selection. Data/model loading, construction of immutable training references, warm-up and diagnostic replay copies are outside timing. EviDrug's serving path precomputes immutable training-case representations, never query predictions. Online query preparation remains timed. The frozen historical measurement protocol is in `configs/efficiency_protocol.json`; the public implementation is in `src/evidrug/serving.py` and `src/evidrug/benchmark.py`.

The registered serving implementation is checked against native EviDrug prediction to maximum error `1e-6`, with unchanged threshold decisions. Same-CPU repeat prediction is checked for all methods. Stored CPU/GPU diagnostics use their separately recorded bounds; some legacy SICdb validation predictions were not saved. Timings describe the measured shared host and setting, not a hardware-independent speed guarantee.

## Comparison scope

This is a retrospective development evaluation with prior MIMIC-IV test use, architecture choices informed by earlier experiments and reused single-seed baseline predictions. Within each declared run, selection uses validation data and model files are frozen before test prediction. Dataset label semantics and mapped coverage differ, so compare methods within cohorts. The manuscript states the single-seed baseline budgets and the three-seed EviDrug setting; this guide documents the complete comparison scope alongside the scalar exports.

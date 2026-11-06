# Experimental Protocol

## Optimization and model selection

All model parameters are trained from scratch within each cohort. Optimization seeds are 0, 1 and 2. Each run uses 20 epochs of backbone learning, 20 epochs of evidence refinement and 20 epochs of DDI-feedback optimization. Batch size is 64 and arithmetic is FP32. AdamW optimizes the embedding encoder at 0.00005 and the remaining backbone at 0.0007. Refinement and feedback optimize the head at 0.0005. Weight decay is 0.001, gradient-norm clipping is 5, and every stage starts a fresh optimizer.

One eligible prefix is sampled per training patient per epoch, cycling through that patient's targets in a deterministic seed-specific order. This gives patients equal sampling weight. Validation and test each use the final eligible encounter. Refinement initialization uses seed s + 4100, feedback optimization uses s + 7100, and the donor ordering seed is 20261026. These optimization choices leave the patient partition unchanged.

Backbone and refinement models maximize validation AP, with the earliest epoch breaking ties. Feedback optimization includes the initial refinement model as epoch 0. Eligible models have AP at most 0.002 below the reference, F1 and Jaccard at most 0.005 below it, and a mean predicted medication count between 0.9 and 1.1 times the reference. Among eligible epochs, selection minimizes validation DDI, then prefers higher AP and an earlier epoch. Feedback uses training queries; model selection uses validation patients.

## Stage, component and regularization controls

The extended-backbone control gives the backbone 20 additional epochs. Evidence refinement instead fixes the backbone and trains the evidence head. Prediction-only training and all DDI controls start from the same seed-specific refinement model, use fresh optimizers and receive the same 20-epoch budget and selection criteria. Fixed coefficients are 0, 0.05, 0.1 and 1.

History removal uses current-encounter evidence and state-only retrieval. Memory removal eliminates retrieval and donor encoding. Set-block removal eliminates medication-set attention. These three variants learn their own backbone and refinement models. Association removal shares the full backbone and refits the affected head. Each variant is evaluated both before and after DDI-feedback optimization. All variants use the same three optimization seeds.

## Metrics and analysis groups

Primary AP is average precision over candidate medications for each patient, followed by a patient average; tied scores are grouped. F1 and Jaccard are also patient averages. The common probability threshold is 0.3; COGNet retains its native no-repeat beam decoder with beam width 4 and at most 45 steps. Hard DDI pools interacting unordered predicted pairs over patients and divides by all predicted unordered pairs, with denominator at least one. Interacting pairs per patient is reported separately. DDI measures the fixed ATC3 interaction graph.

Continued candidates have appeared in any preceding eligible encounter; new-to-record candidates have not. Their precision, recall and F1 are micro-averages over patient--medication pairs, and Avg F1 is the arithmetic mean of Cont and New F1. These analysis categories differ from the backbone's immediately preceding-use indicator.

History depth uses 1--2, 3--5 and at least 6 preceding encounters. Prior medication burden uses 0--5, 6--10 and at least 11 classes in the immediately preceding prescription. Group membership is shared across models; groups with fewer than 30 patients are marked descriptive. Association support is the maximum training diagnosis--medication co-occurrence count for a candidate, grouped as 0, 1--2, 3--9 and at least 10. AP within support bins pools patient--medication candidate pairs. Medication-frequency quartiles use training counts and place tied counts in the lower bin.

Threshold sensitivity uses 0.1 through 0.9 with fixed model weights. Equal-size top-k analysis rounds the training prescription-size quantiles 0.25/0.5/0.75/0.9 to positive integers and removes duplicates. Methods share the resulting values within each cohort. Patient-paired bootstrap uses 2,000 resamples and seed 20261103, reusing each resample across models. Intervals average the three EviDrug models against the same fitted SSPNet baseline and describe patient-sampling variability for those fitted models.

## Baseline budgets and efficiency measurement

The twelve baselines use their recorded single-seed budgets; EviDrug reports three optimization seeds. MetaDrug includes self-adaptation, peer-adaptation and uncertainty filtering, with filtering coefficient 0.2. Its 20 training epochs and 10 auxiliary epochs are preserved. The baseline budgets are listed below.

Efficiency uses 128 validation patients per cohort, FP32, one CPU thread, batch size 16, one warm-up and three measured passes. The host is an Intel Xeon Platinum 8352V at 2.10 GHz. The reported time is median pass duration divided by 128. Method order is forward, reversed and rotated by six positions across passes. Timing includes online retrieval, evidence preparation, collation, prediction, native adaptation or decoding, and final selection. Loading and construction of fixed training references precede measurement. All 39 method--cohort measurements and their three pass durations accompany the PDF.

## Evaluation setting

Models are selected on validation patients before test prediction. The study compares methods within each independently trained cohort. Baselines are existing single-seed runs with their native training budgets, and earlier MIMIC-IV test comparisons informed model development. The reported standard deviations therefore describe EviDrug optimization seeds on the fixed patient partition; they do not estimate baseline training variability.

# Architecture

The public model is `evidrug.models.EviDrug`. Its two children correspond directly to the method: `backbone` establishes medication logits and `refinement` adjusts them using clinical evidence.

| Module | Responsibility | Output |
|---|---|---|
| `models/encoder.py` | Code, visit, order, global-position and shared-segment embeddings, normalization and dropout | Code states `[B,L,192]` |
| `models/backbone.py` | Visit pooling, patient history, medication-specific readout, training-case fusion and one set block | Backbone logits `[B,M]` and evidence |
| `evidence/retrieval.py` | Exact state/change Jaccard retrieval from fixed training references | Eight donor transitions per query |
| `evidence/association.py` | Training-visit code/medication log-ratio statistics | Six clinical features per medication |
| `models/refinement.py` | Shared vector gate and a bounded residual correction | Final logits `[B,M]` |
| `objectives.py` | Ranking objective, soft DDI rate, epoch feedback and validation eligibility | Training losses and selection keys |

`B` is batch size, `L ≤ 512` is the serialized code length, and `M` is the training medication vocabulary size. The clinical encoder contains no Transformer; attention across medication tokens occurs in the single medication-set block.

## Backbone

Structured code states are pooled per encounter and passed through a visit MLP. The current encounter attends to earlier visit states with a learned recency penalty. The previous medication set and attention-weighted historical medication frequency join the clinical context.

Learned medication queries attend to current clinical code states. A medication-specific readout and start/continuation heads produce a patient prior. Training-case attention combines this prior with retrieved donor medication labels. Current-state attention is medication-specific; retrieval shortlists interleave current-state and clinical-change Jaccard rankings. Training queries exclude their own patient's donors.

One set interaction block exchanges information across medication tokens and produces backbone logits. The `no_set` ablation removes that block and its output head physically.

## Evidence refinement

The residual head combines the set representation with medication-specific clinical readout, history-conditioned medication identity, predictive scores, patient context and six clinical association features. A shared vector gate determines the evidence update. An MLP produces `2*tanh(residual)`, added to the backbone logits.

The backbone is frozen during refinement and DDI-feedback optimization. Only the evidence head is updated. The dynamic DDI coefficient is updated after each training epoch; it adds no inference parameters or forward passes.

## Batch contract

Use `data.batching.make_loader` to construct batches from encoded records and retrieved training references.

| Key | Shape | Meaning |
|---|---|---|
| `codes`, `diagnoses`, `procedures` | `[B,V,C]` | Padded encounter code IDs |
| `visit_mask` | `[B,V]` | Valid encounter positions |
| `labels` | `[B,V,M]` | Earlier medications and supervised query targets |
| `ehr_input_ids`, `ehr_visit_ids`, `ehr_serial_ids` | `[B,L]` | Serialized code, visit and order IDs |
| `ehr_visit_map` | `[B,L]` | Pooling destination for each code |
| `transition_before_codes`, `transition_after_codes` | `[B,K,Cd]` | Retrieved training-case clinical codes |
| `transition_before_labels`, `transition_after_labels` | `[B,K,M]` | Retrieved training-case medication sets |
| `transition_mask`, `transition_similarity` | `[B,K]`, `[B,K,2]` | Donor validity and clinical similarities |
| `association_features` | `[B,M,6]` | Normalized sum and mean log-ratios for all codes, diagnoses and procedures |

The current query's label is masked from history and never enters the scoring path. Tests replace it with NaNs and require identical inference probabilities. Empty procedure modalities contribute no clinical association support.

`Predictor` builds the fixed training reference index once and prepares each query online. It stores no query predictions or query evidence cache.


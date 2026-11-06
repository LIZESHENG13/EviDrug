# EviDrug

**Clinical Evidence-Guided Residual Learning for Medication Recommendation with Dynamic DDI Regularization**

EviDrug recommends medication sets from clinical codes and medication history. A structured visit encoder, training-case memory and one medication-set interaction block form the clinical-memory backbone. A residual evidence head refines its logits, and epoch-wise DDI feedback adjusts the training objective.

The selected model uses **zero clinical Transformer layers**, **one medication-set interaction block**, and **no medication-graph representation branch**. All embeddings are trained on the study cohort.

## Install

```bash
git clone https://github.com/LIZESHENG13/EviDrug.git
cd EviDrug
python -m pip install -e ".[test]"
python -m pytest
python scripts/audit_release.py
```

Python 3.11 is the tested environment. Install the appropriate PyTorch CPU/CUDA build for your machine. The tests run on synthetic data; no clinical dataset is required. Exact study environment versions are recorded in `configs/tested_environment.json`.

## Project layout

```text
src/evidrug/
  models/             structured encoder, clinical-memory backbone, refinement
  evidence/           training-case retrieval and clinical associations
  data/               cohort preparation, partitions, serialization, batching
  training.py         three-stage training and validation selection
  inference.py        complete online recommendation
  serving.py          fixed training-case precomputation for inference
  objectives.py       prediction ranking and dynamic DDI regularization
  metrics.py          AP, F1, Jaccard and pair-weighted DDI
  cli.py              the evidrug command
configs/              study settings and cohort definitions
scripts/              study launcher, metric replay and release audit
tests/                synthetic pipeline and model invariants
results/              frozen aggregate study results
docs/                 architecture, data and reproduction guides
verification/         release and refactor verification records
```

## Train and evaluate

Obtain MIMIC-IV 3.1, NWICU 0.1.0 or SICdb 1.0.8 through the respective dataset access process. Download the public mapping resources and prepare a cohort:

```bash
evidrug resources --output resources
evidrug prepare --dataset nwicu --input /path/to/nwicu.zip \
  --output /path/to/private/data/nwicu --resources resources
evidrug train --dataset nwicu --data /path/to/private/data/nwicu/patients.pkl \
  --output /path/to/private/runs/nwicu/full/seed0 --seed 0 --device cuda:0
```

The default run trains the backbone, evidence refinement and DDI-feedback stages for 20 epochs each. Select checkpoints on validation data, then freeze the completed run group before a separate test evaluation:

```bash
evidrug freeze --runs /path/to/private/runs
evidrug evaluate --data /path/to/private/data/nwicu/patients.pkl \
  --run /path/to/private/runs/nwicu/full/seed0 \
  --freeze /path/to/private/runs/TEST_FREEZE.json --device cuda:0
```

For all three datasets, seeds 0/1/2, major-component ablations and matched DDI controls, see [Reproduction](docs/reproduction.md). `python scripts/run_study.py --help` lists the study launcher options. [Data](docs/data.md) defines eligibility, mappings and patient partitions.

## Model API

```python
from evidrug import EviDrug

model = EviDrug(vocabulary=num_code_entries, medications=num_medications)
logits, auxiliary_loss = model(batch)       # logits: [batch, medications]
probabilities, evidence = model.distribution(batch)
```

The batch includes structured clinical codes, previous medication sets and fixed training evidence. Query medication labels are used for supervision and evaluation. [Architecture](docs/architecture.md) describes the batch fields and tensor flow.

## Reported study

| Dataset | Test AP, mean ± sample SD | Parameters |
|---|---:|---:|
| MIMIC-IV | 0.7763 ± 0.0004 | 7.766M |
| NWICU | 0.6909 ± 0.0007 | 2.142M |
| SICdb | 0.7505 ± 0.0010 | 1.069M |

These results identify the frozen study model `clinical_memory_dynamic_20261104`, evaluated with three optimization seeds per cohort. Full seed-level values, component ablations, coefficient controls and baseline comparisons are in `results/`. See [Results and provenance](docs/results.md) for their definitions and comparison settings.

This repository provides the current EviDrug implementation and its ablations. Historical experiment drivers are replaced by the focused package above. Checkpoint migration and numerical equivalence are documented in `verification/`; fresh training follows the same equations, budgets and selectors, while its random initialization stream can differ from the historical runs.

Clinical records, patient-level predictions, retrieval banks, trained weights and the separate supplementary PDF are not distributed here.

## Attribution

Software is released under the [MIT license](LICENSE). Dataset and public mapping resources retain their original terms. Method references are listed in [references.bib](references.bib); the baseline publication registry is in `results/baseline_registry.csv`.


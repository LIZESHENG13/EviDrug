# Data protocol

## Sources and eligible cohorts

Obtain the source data from the original providers. This package distributes no patient records or trained weights.

| Dataset | Version | Eligible patients / visits | Train / validation / test | Code vocabulary | ATC3 classes |
|---|---|---:|---|---:|---:|
| [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/) | 3.1 | 86,670 / 352,967 | 62,402 / 6,934 / 17,334 | 35,391 | 201 |
| [NWICU](https://doi.org/10.13026/s84w-1829) | 0.1.0 | 12,492 / 43,115 | 8,993 / 1,000 / 2,499 | 6,258 | 167 |
| [SICdb](https://physionet.org/content/sicdb/1.0.8/) | 1.0.8 | 3,898 / 9,550 | 2,806 / 312 / 780 | 1,069 | 81 |

The vocabulary includes three special code entries, whereas medication classes are training-observed ATC3 classes. `configs/cohorts.json` records the partition hashes. The cohorts are not restricted to a small MIMIC pilot. Models are trained independently on each cohort.

## MIMIC-IV

`src/evidrug/data/mimic_iv.py` reads the MIMIC-IV ZIP, whole-encounter diagnosis/procedure codes, admissions and prescriptions. It normalizes source identifiers and maps prescription NDCs to ATC3 using the fixed public mapping resource; the original MIMIC rule unions mapped classes for a normalized NDC. The eligible longitudinal cohort requires at least two retained encounters with diagnoses and mapped medications; procedures may be empty. It orders encounters by recorded admission time and then admission ID. It does not use discharge-time overlap filtering in this MIMIC preparation. This is encounter-level medication-set reconstruction with whole-encounter codes, rather than an early-admission prediction task.

The model receives diagnosis and procedure namespaces and strictly earlier medication sets. Medication quantities, doses and routes are not prediction targets. The exact filtering and ordering operations are retained in the source instead of being reimplemented in a second preprocessing pipeline.

## NWICU

`src/evidrug/data/nwicu_inspection.py` verifies archive integrity, joins and mapping coverage. `nwicu.py` constructs hospital-admission sequences; the cohort is hospital-level, not ICU-only. There is no released procedure-ICD table in these inputs, so no procedure surrogate is created.

Diagnosis tokens preserve ICD version and code identity. Source NDCs are normalized using the study's source-specific rule; accept only a unique mapped ATC3 class. Prescription intervals must have valid start/stop times, end no earlier than start, and overlap the admission. Labels are mapped order types, not verified administrations. The reported unique mapping coverage is 75.03% of all prescription rows; the source audit separately counts the mapped rows with valid admission overlap.

Retain admissions with codes and mapped medication types. Exclude tied-start sequences. Starting with the latest eligible admission, work backward and retain only admissions that end before the next selected admission starts. At least one preceding admission is required. The wrapper runs the archive audit before preparation and verifies that cohort counts match it.

## SICdb

`src/evidrug/data/sicdb.py` reads `cases.csv.gz`, `d_references.csv.gz` and `medication.csv.gz`. `ICD10Main` supplies the principal diagnosis; there are no procedure codes in this pipeline.

Only references explicitly marked `Drug` are medication candidates. Exclude food/fluid references, nonpositive amounts, negative offsets and events after the case duration. A positive `Amount` or `AmountPerMinute` qualifies. The label is the set of mapped administered drug types over the full case.

`configs/sicdb_drug_mapping.json` freezes the dictionary-level generic-name mappings used for the study. It contains name-to-ATC3 decisions and mapping methods, without patient rows or medication frequencies. The original mapping combines local ATC exact/explicit-alias rules and the recorded dictionary-level RxNorm resolution, accepting a unique class. The portable wrapper reads the frozen mapping and performs no online enrichment. Weighted mapping coverage is 80.69% of qualifying drug rows.

Sort cases by `OffsetAfterFirstAdmission`; require valid nonnegative durations, diagnosis and mapped drugs. Remove sequences with tied case starts. The final eligible case is the query; preceding cases must finish before its start. Patients require at least one such history case. The prepared `patients.pkl` output retains the study's record format.

## Partitions, vocabulary and references

The **patient-partition seed is 0**, distinct from optimization seeds. `evidrug.data.records.patient_split` uses `numpy.random.default_rng(0).permutation`. Of all patients, the first `floor(0.8*N)` form the development partition; its first `floor(0.9*floor(0.8*N))` patients train, and the remainder validate. The final 20% test. No patient crosses partitions.

Code and medication vocabularies use training patients. Unknown clinical codes map to ID 1; medications absent from the training vocabulary are omitted from labels and counted in preparation metadata. These choices are identical across methods within a cohort. No cohort-size limit is applied. The EviDrug encoder's 512-position sequence limit is separate from cohort eligibility.

Case memories and association counts use training patients. A training query excludes its own patient's cases/statistics. Validation and test use fixed training references. Query medication labels enter supervision and evaluation, never the scoring path. `tests/test_model.py` checks prediction invariance after replacing query labels with NaNs.

The fixed DDI proxy uses PyHealth's released [`DDI_GAMENet.csv`](https://storage.googleapis.com/pyhealth/resource/DDI_GAMENet.csv). [GAMENet (AAAI 2019)](https://cdn.aaai.org/ojs/3905/3905-13-6964-1-10-20190702.pdf) describes its interaction knowledge as derived from TWOSIDES. [PyHealth's ATC implementation](https://pyhealth.readthedocs.io/en/latest/_modules/pyhealth/medcode/codes/atc.html) identifies this CSV as its GAMENet DDI resource.

`src/evidrug/data/records.py` maps each ATC drug code to its first four characters (ATC level 3), retains a pair when both classes occur in the cohort's training medication vocabulary and the classes differ, and sets both corresponding adjacency entries to 1. Duplicate source pairs therefore produce a single undirected edge. The matrix is binary, symmetric and zero-diagonal, and remains fixed across methods within a cohort. Its SHA-256 is `ba58181635fbd740f2f36d70d5f7354e4690cedb6e7a54a2ec8eaa5c8cf26f6c`. `configs/resource_checksums.json` records the source URLs and checksums for this interaction table and the public medication mappings.

Medication co-occurrence is computed from training visits. It is distinct from the external DDI graph; the selected EviDrug has no medication-graph representation parameters.

Raw/prepared clinical records, split identity arrays, case banks, local prediction arrays and learned parameters stay in the caller's private data directory. Artifact scalar exports identify experiments by hashes without distributing those arrays.

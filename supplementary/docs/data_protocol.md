# Data and Cohort Construction

The three cohorts use the complete eligible patient populations. This section specifies their source fields, medication mappings, timing rules, patient partitions and training reference data. All reported counts are cohort-level summaries.

## Source fields and task units

| Dataset | Source tables and required fields | Prediction unit |
| --- | --- | --- |
| MIMIC-IV 3.1 | admissions: subject_id, hadm_id, admittime; diagnoses_icd and procedures_icd: hadm_id, icd_code, icd_version, seq_num; prescriptions: hadm_id, ndc | Final eligible hospital admission; all earlier eligible admissions provide history |
| NWICU 0.1.0 | admissions: subject_id, hadm_id, admittime, dischtime; diagnoses_icd: icd_code, icd_version, seq_num; prescriptions: hadm_id, ndc, starttime, stoptime | Final eligible hospital admission after adjacent non-overlap filtering; hospital-level cohort |
| SICdb 1.0.8 | cases: CaseID, PatientID, ICD10Main, OffsetAfterFirstAdmission, TimeOfStay; d_references: ReferenceName, ReferenceGlobalID, ReferenceValue; medication: CaseID, DrugID, Amount, AmountPerMinute, Offset | Final eligible ICU/surgery case; earlier cases must finish before that query starts |

Current inputs are whole-encounter clinical codes, together with strictly earlier medication sets. Labels are sets of ATC3 classes over the full encounter. The task is encounter-level medication-set reconstruction. Dose, route, drug administration order and adverse events are not prediction targets. Medication names in the frozen mapping dictionary describe public drug semantics and contain no patient frequencies.

Diagnosis and procedure tokens retain their namespaces and ICD version. MIMIC-IV has both diagnoses and procedures. NWICU has diagnosis codes; its released ICU procedure-event table is not a hospital procedure-ICD table and is not used as a surrogate. SICdb uses the principal ICD10Main diagnosis only. Missing procedure modalities have no real procedure codes; the baseline-compatible UNK placeholder does not contribute association evidence.

## Eligible cohort statistics

| Cohort | Patients | Encounters | Mean encounters | Mean final target ATC3 | Train code entries | Train ATC3 |
| --- | --- | --- | --- | --- | --- | --- |
| MIMIC-IV | 86,670 | 352,967 | 4.0725 | 21.8227 | 35391 | 201 |
| NWICU | 12,492 | 43,115 | 3.4514 | 11.9402 | 6258 | 167 |
| SICdb | 3,898 | 9,550 | 2.4500 | 9.7863 | 1069 | 81 |

Code entries include PAD, UNK and GAP. The code vocabulary and medication output vocabulary are fitted to training patients only. The all-cohort medication universes are 201, 170 and 83 for MIMIC-IV, NWICU and SICdb, whereas the respective model output sizes are 201, 167 and 81. These denominators serve different purposes.

MIMIC-IV has a maximum of 98 retained encounters per patient. Its mean medical-code count is 61.5525 per patient across all retained encounters, not per encounter. NWICU encounter counts have median 3, interquartile range 2--4 and maximum 37. Its final target has median 11 ATC3 classes, interquartile range 8--15 and range 1--52; final diagnosis-code counts have mean 4.8204, median 4, interquartile range 2--6 and range 1--87. Of NWICU's final query admissions, 4,958 contain an ICU stay; eligibility does not require an ICU stay.

## MIMIC-IV medication mapping and eligibility

Normalize reference and source NDCs by removing a terminal .0, requiring a nonzero numeric string and padding to 11 digits. The frozen MIMIC rule unions all mapped ATC3 classes for a normalized NDC. ATC3 is the first four characters of the source ATC code. Clinical-code ordering follows source seq_num within diagnosis and procedure namespaces, with diagnoses serialized before procedures.

Begin with patients having at least two source admissions. Retain admissions with diagnosis codes and mapped medication labels; procedures may be empty. Sort by admission time and then admission ID; require at least two retained admissions. The recorded filtering counts are 380 admissions without diagnosis and 58,499 without mapped ATC3 among the repeated-admission preparation pool. These are sequential filters. The original aggregate metadata does not record a full raw-admission denominator for this preparation, so those counts are not presented as a complete source attrition flow. This MIMIC pipeline does not apply prescription/admission overlap checks or discharge-time history filtering.

## NWICU NDC and temporal audit

The archive has 25,923 patients and 61,843 hospital admissions. Of these, 13,833 patients have at least two admissions, accounting for 49,753 admissions. The diagnosis table has 371,807 ICD-10 rows covering 60,973 admissions and 8,086 distinct versioned codes. Admission IDs are unique; the audit finds zero orphan admission links, patient/admission mismatches or invalid admission intervals in these joins.

Source package NDCs must be an explicit 11-digit nonzero code or a three-segment package code with lengths 4-4-2, 5-3-2, 5-4-1 or 5-4-2. Pad each segment to 5-4-2. Reject incomplete/product-only codes and unsegmented 10-digit strings rather than guessing segment boundaries. Reference NDCs use the numeric normalization above. Accept only a unique ATC3 class per normalized NDC; use no medication-name fallback. The reference table has 672,734 uniquely mapped and 119,121 ambiguous normalized NDCs.

| NWICU audit quantity | Rows |
| --- | --- |
| All source prescription rows | 1,852,983 |
| Nonempty NDC rows | 1,632,392 |
| Standardizable package NDC rows | 1,591,906 |
| Unique ATC3 mapping before interval checks | 1,390,310 |
| Any mapping, including ambiguous | 1,517,039 |
| Ambiguous mapping rows | 126,729 |
| Order stop earlier than start | 136,261 |
| Valid interval outside admission | 88,488 |
| Valid interval overlapping admission | 1,628,234 |
| Unique mapped and admission-overlapping | 1,220,787 |
| Repeated-patient prescription rows | 1,410,856 |
| Repeated-patient unique mapped rows | 1,063,622 |
| Repeated-patient valid admission-overlapping rows | 1,238,661 |
| Repeated-patient mapped admission-overlapping rows | 933,613 |

The 75.03% mapping coverage is 1,390,310 / 1,852,983 source prescription rows, before the interval filter. The final mapped and interval-qualified count is 1,220,787; it has a different denominator. Start and stop must parse, stop must be no earlier than start, and the closed order interval must intersect the admission: start <= discharge and stop >= admission. These labels describe medication order types, not verified administrations.

Retain admissions with diagnosis codes and mapped qualified orders. Exclude a patient if eligible admissions share a start time; the frozen cohort has zero such exclusions. Starting from the latest eligible admission, traverse backward and retain an earlier admission only if its discharge is no later than the next selected admission's start. This removes 20 overlapping history admissions. Require at least two selected admissions. The selected cohort has 12,492 patients and 43,115 admissions, with 7,024 distinct diagnosis codes and 170 medication classes before training-only vocabulary fitting. NWICU's global uniquely mapped source set has 172 ATC3 classes; this is not the cohort or model output vocabulary.

## SICdb drug dictionary and temporal audit

Only d_references entries with ReferenceName equal to Drug qualify. Medication rows must belong to a repeated-patient case, have positive Amount or AmountPerMinute, and have a finite Offset in [0, TimeOfStay]. Non-drug food/fluid references are excluded before computing drug mapping coverage.

The medication dictionary contains 520 names, 303 with an accepted mapping and 217 unresolved. Mapping uses exact generic names and explicit salt/terminal-e orthographic aliases against the ATC table. A mixture with multiple recognized ingredients must have a unique common ATC3 class. Recorded RxNorm resolutions require a unique normalized ingredient and direct unique ATC membership. Ambiguous classes and edit-distance fuzzy matches are rejected. The dictionary is fixed before model training. Every name-to-class decision appears in the final section.

| SICdb audit quantity | Count and denominator |
| --- | --- |
| All raw medication rows | 5,141,127 |
| Non-drug reference rows removed within repeated-patient rows | 919,865 |
| Positive-amount/rate, in-case drug rows among repeated patients | 863,806 |
| Mapped rows among those qualifying drug rows | 697,025 |
| Mapping coverage | 697,025 / 863,806 = 80.6923% |
| Cases missing diagnosis or mapped medication labels | 105 |
| Overlapping history cases removed relative to the final query | 29 |

The 919,865 non-drug rows are counted after restricting to repeated patients, whereas 5,141,127 is the full raw medication-file count. They cannot be subtracted directly to infer a complete raw-file attrition flow.

Uppercase ICD10Main and remove whitespace while retaining punctuation; prefix it with D10:. Sort cases by OffsetAfterFirstAdmission and case ID. Start and duration must be finite and duration nonnegative; retain cases with diagnosis and mapped medications. Exclude tied starts. The last eligible case is the query, and every history case must end by that query's start. Unlike NWICU's backward adjacent filter, SICdb histories may overlap each other while each finishes before the query. Require at least one history case.

## Patient partitions, unknowns and sequence scope

Randomly permute the deterministically ordered patients with partition seed 0. The first floor(0.8N) patients form the development partition; the first floor(0.9 floor(0.8N)) patients train and the remainder validate. Remaining patients test. This gives approximately 72%/8%/20%, with cohort-specific integer rounding. Optimization seeds 0/1/2 do not change patient partitions.

| Cohort | Train/val/test patients | Unknown code occurrences, train/val/test | Unseen drug occurrences, train/val/test | Undirected DDI edges |
| --- | --- | --- | --- | --- |
| MIMIC-IV | 62402/6934/17334 | 0/1199/2839 | 0/0/0 | 631 |
| NWICU | 8993/1000/2499 | 0/343/672 | 0/5/3 | 536 |
| SICdb | 2806/312/780 | 0/74/188 | 0/4/4 | 206 |

Code IDs are PAD=0, UNK=1 and GAP=2 followed by sorted training-observed codes. Unknown clinical codes map to UNK. Medications absent from the training output vocabulary are omitted from labels and counted as unseen occurrences; these are occurrence counts, not numbers of affected patients.

There is no patient-count subsampling or preparation-time visit/code truncation. The encoder retains the most recent 512 serialized code positions, independently of cohort eligibility. Visit positions clip at 63 and within-visit positions at 127. Donor encounters share the same structured encoder. Query medications are supervision and evaluation labels, never prediction inputs.

Each validation/test patient has one final-encounter query. Training uses one eligible prefix per patient per epoch, cycling deterministically through that patient's eligible targets. NWICU has 21,873 eligible training prefixes and 8,993 targets per epoch; SICdb has 4,116 and 2,806, respectively. The training monitor uses a fixed first-epoch prefix schedule and no validation/test labels.

## Fixed training evidence and DDI resource

Training-case donors use training patients only. A training query excludes its own patient's transitions; the additional exclusion under the fixed donor ordering is retained across experiments. The eight-neighbor shortlist interleaves state and clinical-change Jaccard rankings with deterministic tie breaks. Similarity uses clinical codes, never query medications. Clinical association statistics count binary visit-level code/medication co-occurrence, and a training query removes all statistics of its own patient. Validation and test use fixed training references.

The DDI graph uses the public PyHealth DDI_GAMENet.csv resource described by GAMENet as TWOSIDES-derived interaction knowledge. Truncate ATC codes to four characters, retain pairs of distinct training-vocabulary classes, collapse duplicate pairs, symmetrize and zero the diagonal. The three graphs contain 631, 536 and 206 undirected edges. This fixed interaction proxy differs from the training co-prescription adjacency; the selected EviDrug does not use a medication-graph representation branch.

The medication reference tables are NDC_to_ATC.csv and ATC.csv; interaction pairs come from DDI_GAMENet.csv. All three are the public reference resources distributed with PyHealth. These resources define drug-code mappings and interactions independently of the patient partitions.

# EviDrug supplementary material

The PDF contains five sections:

1. Data and cohort construction: source fields, eligibility, timing, medication mapping and patient partitions.
2. Additional method details: encoder dimensions, historical context, case attention, associations and refinement.
3. Experimental protocol: training settings, model selection, baseline budgets, metrics and clinical analysis groups.
4. Individual-seed results: the main model, stage comparisons, major-component ablations and fixed/dynamic DDI controls across all three cohorts.
5. Complete SICdb medication mapping: all 520 dictionary decisions and their mapping rules.

[EviDrug_Supplementary.pdf](EviDrug_Supplementary.pdf) is the complete reading version. `tables/` contains the numerical results, including all 144 EviDrug stage/variant/seed records, 36 baseline records, clinical subgroup analyses and 39 efficiency measurements. `data/sicdb_medication_mapping.csv` contains the complete mapping dictionary. `docs/` provides the data and experimental protocols in text form.

Main-model and control experiments use optimization seeds 0, 1 and 2 on fixed patient partitions. Summary rows in the PDF report mean and sample standard deviation. Existing baselines use one optimization seed and their documented budgets.

Code: https://github.com/LIZESHENG13/EviDrug

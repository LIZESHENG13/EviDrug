"""SICdb 1.0.8 principal-diagnosis cohort and frozen drug-name mappings."""

import hashlib
import json
import pickle
import re
from collections import Counter, defaultdict
from importlib.resources import files

import numpy as np
import pandas as pd

from ..utils import write


def collect_sicdb(directory):
    cases = pd.read_csv(directory / "cases.csv.gz", dtype=str).fillna("")
    refs = pd.read_csv(directory / "d_references.csv.gz", dtype=str).fillna("")
    # Medication rows also include food and fluid references. These are not
    # drug labels and must not inflate the drug-mapping denominator.
    drug_refs = refs[refs.ReferenceName.eq("Drug")]
    names = dict(zip(drug_refs.ReferenceGlobalID, drug_refs.ReferenceValue))
    repeated = set(cases.PatientID.value_counts().loc[lambda n: n >= 2].index)
    cases = cases[cases.PatientID.isin(repeated)]
    valid_ids = set(cases.CaseID)
    ends = dict(zip(cases.CaseID, pd.to_numeric(cases.TimeOfStay, errors="coerce")))
    drugs = defaultdict(set)
    frequencies = Counter()
    counts = Counter()
    for chunk in pd.read_csv(
        directory / "medication.csv.gz", dtype=str, keep_default_na=False, chunksize=200000
    ):
        counts["raw_medication_rows"] += len(chunk)
        chunk = chunk[chunk.CaseID.isin(valid_ids)]
        counts["non_drug_reference_rows_excluded"] += int((~chunk.DrugID.isin(names)).sum())
        chunk = chunk[chunk.DrugID.isin(names)]
        positive = (pd.to_numeric(chunk.Amount, errors="coerce").fillna(0) > 0) | (
            pd.to_numeric(chunk.AmountPerMinute, errors="coerce").fillna(0) > 0
        )
        offset = pd.to_numeric(chunk.Offset, errors="coerce")
        chunk = chunk[positive & offset.ge(0) & offset.le(chunk.CaseID.map(ends))]
        counts["positive_rows_repeated_patients"] += len(chunk)
        frequencies.update(chunk.DrugID.map(names).fillna(""))
        for case, drug in (
            chunk[["CaseID", "DrugID"]].drop_duplicates().itertuples(index=False, name=None)
        ):
            if drug in names:
                drugs[case].add(names[drug])
    raw = {
        "dataset": "sicdb",
        "cases": cases.to_dict("records"),
        "drugs": dict(drugs),
        "name_frequencies": dict(frequencies),
        "counts": dict(counts),
    }
    return raw


def prepare(directory, output):
    raw = collect_sicdb(directory)
    dictionary = json.loads(
        files("evidrug.data").joinpath("mappings/sicdb_drug_mapping.json").read_text()
    )
    mapping = {
        name: dictionary.get(name, {"atc3": [], "method": "absent_from_frozen_dictionary"})
        for name in raw["name_frequencies"]
    }
    counter = Counter(raw["counts"])
    frequencies = raw["name_frequencies"]
    counter["dictionary_names"] = len(mapping)
    counter["mapped_dictionary_names"] = sum(bool(v["atc3"]) for v in mapping.values())
    counter["dictionary_weighted_rows"] = sum(frequencies.values())
    counter["mapped_dictionary_weighted_rows"] = sum(
        frequencies[name] for name, value in mapping.items() if value["atc3"]
    )
    drugs = {
        case: sorted({c for name in names for c in mapping[name]["atc3"]})
        for case, names in raw["drugs"].items()
    }
    groups = defaultdict(list)
    for row in raw["cases"]:
        case, patient = row["CaseID"], row["PatientID"]
        diagnosis = row["ICD10Main"].strip().upper()
        codes = ["D10:" + re.sub(r"\s+", "", diagnosis)] if diagnosis else []
        try:
            start = float(row["OffsetAfterFirstAdmission"])
            duration = float(row["TimeOfStay"])
        except (ValueError, TypeError):
            counter["invalid_timing_visits"] += 1
            continue
        if not np.isfinite(start) or not np.isfinite(duration) or duration < 0:
            counter["invalid_timing_visits"] += 1
            continue
        if not codes or not drugs.get(case):
            counter["missing_diagnosis_or_mapped_drugs_visits"] += 1
            continue
        groups[patient].append(
            {
                "hadm_id": case,
                "time": start,
                "end": start + duration,
                "codes": sorted(codes),
                "drugs": drugs[case],
            }
        )
    patients = []
    for patient, visits in sorted(groups.items()):
        visits.sort(key=lambda visit: (visit["time"], visit["hadm_id"]))
        if len({visit["time"] for visit in visits}) != len(visits):
            counter["episodes_excluded_tied_admission_times"] += 1
            continue
        query = visits[-1]
        history = [visit for visit in visits[:-1] if visit["end"] <= query["time"]]
        counter["overlapping_history_visits_removed"] += len(visits) - 1 - len(history)
        if history:
            patients.append({"patient_id": "sicdb:" + patient, "visits": history + [query]})
    if not patients:
        raise ValueError("No eligible SICdb patients after mapping and timing checks.")
    counter["eligible_patients"] = len(patients)
    counter["eligible_visits"] = sum(len(patient["visits"]) for patient in patients)
    summary = {
        "dataset": "sicdb",
        "patients": len(patients),
        "visits": counter["eligible_visits"],
        "counts": dict(counter),
        "unique_atc3": len({d for p in patients for v in p["visits"] for d in v["drugs"]}),
        "mean_target_drugs": float(np.mean([len(p["visits"][-1]["drugs"]) for p in patients])),
        "mean_visits": float(np.mean([len(p["visits"]) for p in patients])),
        "mapping_row_coverage": counter["mapped_dictionary_weighted_rows"]
        / max(counter["dictionary_weighted_rows"], 1),
        "drug_mapping": "Frozen public dictionary; exact or recorded unique ATC3 resolution.",
        "input_scope": "Principal diagnosis only; no procedure surrogate.",
        "label_scope": "Positive-amount drug types within the full case duration.",
    }
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "patients.pkl"
    with destination.open("wb") as stream:
        pickle.dump({"patients": patients, "metadata": summary}, stream, protocol=5)
    summary["data_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
    write(destination.with_suffix(".json"), summary)
    write(output / "sicdb_drug_dictionary.json", mapping)
    return destination

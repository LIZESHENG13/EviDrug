"""Stream MIMIC-IV 3.1 into eligible encounter-level medication sets."""

import gzip
import hashlib
import json
import pickle
import time
import zipfile
from collections import defaultdict

import numpy as np
import pandas as pd


def normalize_ndc(value):
    value = str(value).strip()
    if value.endswith(".0"):
        value = value[:-2]
    return value.zfill(11) if value.isdigit() and int(value) else ""


def prepare(archive_path, output, resource_dir, min_visits=2):
    started = time.monotonic()
    mapping = pd.read_csv(resource_dir / "NDC_to_ATC.csv", dtype=str, keep_default_na=False)
    ndc_to_atc = defaultdict(set)
    for ndc, atc in mapping.itertuples(index=False, name=None):
        if len(atc) >= 4:
            ndc_to_atc[normalize_ndc(ndc)].add(atc[:4])
    print("Loaded NDC -> ATC-3 mapping:", len(ndc_to_atc), flush=True)
    with zipfile.ZipFile(archive_path) as archive:

        def load(name, columns):
            with archive.open("mimic-iv-3.1/hosp/" + name + ".csv.gz") as raw:
                with gzip.open(raw, "rb") as stream:
                    return pd.read_csv(stream, usecols=columns, dtype=str, keep_default_na=False)

        admissions = load("admissions", ["subject_id", "hadm_id", "admittime"])
        # Only patients with >=2 admissions can satisfy the final cohort rule.
        repeated = admissions.subject_id.value_counts()
        repeated = set(repeated[repeated >= min_visits].index)
        admissions = admissions[admissions.subject_id.isin(repeated)]
        admission_set = set(admissions.hadm_id)
        diagnoses = load("diagnoses_icd", ["hadm_id", "icd_code", "icd_version", "seq_num"])
        diagnoses = diagnoses[diagnoses.hadm_id.isin(admission_set)]
        procedures = load("procedures_icd", ["hadm_id", "icd_code", "icd_version", "seq_num"])
        procedures = procedures[procedures.hadm_id.isin(admission_set)]
        medical = {}
        for label, frame in [("D", diagnoses), ("P", procedures)]:
            frame["order"] = pd.to_numeric(frame.seq_num)
            frame = frame.sort_values(["hadm_id", "order"])
            frame["token"] = label + frame.icd_version + ":" + frame.icd_code
            medical[label] = (
                frame.groupby("hadm_id")
                .token.agg(lambda values: list(dict.fromkeys(values)))
                .to_dict()
            )
            print("Grouped", label, len(medical[label]), "admissions", flush=True)
        del diagnoses, procedures

        medications = defaultdict(set)
        seen_rows = mapped_rows = 0
        # Resolve mapping once per distinct NDC, rather than once per prescription.
        with archive.open("mimic-iv-3.1/hosp/prescriptions.csv.gz") as raw:
            with gzip.open(raw, "rb") as stream:
                for chunk in pd.read_csv(
                    stream,
                    usecols=["hadm_id", "ndc"],
                    dtype=str,
                    keep_default_na=False,
                    chunksize=500000,
                ):
                    seen_rows += len(chunk)
                    chunk = chunk[chunk.hadm_id.isin(admission_set)]
                    chunk = chunk.drop_duplicates(["hadm_id", "ndc"])
                    local_map = {
                        ndc: ndc_to_atc.get(normalize_ndc(ndc), set()) for ndc in chunk.ndc.unique()
                    }
                    for hadm, ndc in chunk.itertuples(index=False, name=None):
                        # read_csv preserves file column order: hadm_id precedes ndc.
                        values = local_map[ndc]
                        if values:
                            medications[hadm].update(values)
                            mapped_rows += 1
                    if seen_rows % 5000000 == 0:
                        print("Prescription rows scanned:", seen_rows, flush=True)

    records = defaultdict(list)
    excluded = defaultdict(int)
    for row in admissions.sort_values(["subject_id", "admittime", "hadm_id"]).itertuples(
        index=False
    ):
        codes = medical["D"].get(row.hadm_id, []) + medical["P"].get(row.hadm_id, [])
        drugs = sorted(medications.get(row.hadm_id, []))
        if not medical["D"].get(row.hadm_id):
            excluded["no_diagnosis"] += 1
            continue
        if not drugs:
            excluded["no_mapped_atc3"] += 1
            continue
        records[row.subject_id].append(
            {"hadm_id": row.hadm_id, "time": row.admittime, "codes": codes, "drugs": drugs}
        )
    patients = [
        {"patient_id": subject, "visits": visits}
        for subject, visits in sorted(records.items())
        if len(visits) >= min_visits
    ]
    all_medications = sorted(
        {code for record in patients for visit in record["visits"] for code in visit["drugs"]}
    )
    # This code-system universe is not fitted to test outcomes; split vocabularies are fitted later.
    lengths = np.array([len(p["visits"]) for p in patients])
    code_counts = np.array([sum(len(v["codes"]) for v in p["visits"]) for p in patients])
    metadata = {
        "dataset": "MIMIC-IV 3.1",
        "archive": str(archive_path),
        "patients": len(patients),
        "visits": int(lengths.sum()),
        "mean_visits": float(lengths.mean()),
        "max_visits": int(lengths.max()),
        "mean_medical_codes": float(code_counts.mean()),
        "mean_target_atc3": float(np.mean([len(p["visits"][-1]["drugs"]) for p in patients])),
        "atc3_codes": all_medications,
        "excluded_admissions": dict(excluded),
        "cohort": ">=2 chronologically ordered admissions with diagnoses and mapped ATC-3 drugs; empty procedures allowed",
        "prediction_task": "one last-admission query per patient; all previous admissions are support",
        "icd_versions": "Diagnosis and procedure namespaces include ICD version",
        "ndc_mapping_sha256": hashlib.sha256(
            (resource_dir / "NDC_to_ATC.csv").read_bytes()
        ).hexdigest(),
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        pickle.dump({"metadata": metadata, "patients": patients}, stream, protocol=5)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)
    return metadata

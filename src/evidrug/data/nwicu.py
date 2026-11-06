"""Audited hospital-level NWICU cohort with ordered, non-overlapping histories."""

import hashlib
import json
import pickle
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import write as atomic_json
from .mimic_iv import normalize_ndc
from .nwicu_inspection import digest, normalize_source_ndc, table
from .records import load_split


def prepare(archive_path, output, resources, audit_path):
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "patients.pkl"
    if destination.exists():
        metadata = json.loads(destination.with_suffix(".json").read_text())
        assert metadata["data_sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()
        assert (
            metadata["preparation_source_sha256"]
            == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        )
        return destination
    started = time.monotonic()
    audit = json.loads(audit_path.read_text())
    with archive_path.open("rb") as stream:
        assert digest(stream) == audit["archive_sha256"]
    assert audit["integrity"]["all_passed"]
    mapping_path = resources / "NDC_to_ATC.csv"
    assert (
        hashlib.sha256(mapping_path.read_bytes()).hexdigest() == audit["mapping"]["resource_sha256"]
    )
    frame = pd.read_csv(mapping_path, dtype=str, keep_default_na=False)
    candidates = defaultdict(set)
    for ndc, atc in frame.itertuples(index=False, name=None):
        key = normalize_ndc(ndc)
        if key and len(atc) >= 4:
            candidates[key].add(atc[:4])
    mapping = {ndc: next(iter(v)) for ndc, v in candidates.items() if len(v) == 1}
    del frame, candidates
    drugs = defaultdict(set)
    counts = Counter()
    with zipfile.ZipFile(archive_path) as archive:
        with table(
            archive, "admissions", ["subject_id", "hadm_id", "admittime", "dischtime"]
        ) as admissions:
            admissions = admissions.copy()
        admissions["start"] = pd.to_datetime(admissions.admittime, errors="coerce")
        admissions["end"] = pd.to_datetime(admissions.dischtime, errors="coerce")
        assert not admissions.hadm_id.duplicated().any()
        assert not (
            admissions.start.isna() | admissions.end.isna() | admissions.end.lt(admissions.start)
        ).any()
        start_by_hadm = dict(zip(admissions.hadm_id, admissions.start))
        end_by_hadm = dict(zip(admissions.hadm_id, admissions.end))
        with table(archive, "diagnoses_icd") as diagnoses:
            diagnoses = diagnoses[diagnoses.icd_code.ne("")].copy()
            diagnoses["order"] = pd.to_numeric(diagnoses.seq_num, errors="raise")
            diagnoses = diagnoses.sort_values(["hadm_id", "order"], kind="stable")
            diagnoses["token"] = "D" + diagnoses.icd_version + ":" + diagnoses.icd_code
            codes = (
                diagnoses.groupby("hadm_id").token.agg(lambda x: list(dict.fromkeys(x))).to_dict()
            )
        cache = {}
        with table(
            archive, "prescriptions", ["hadm_id", "ndc", "starttime", "stoptime"], 200000
        ) as chunks:
            for chunk in chunks:
                counts["raw_prescription_rows"] += len(chunk)
                for ndc in chunk.ndc.unique():
                    cache.setdefault(ndc, mapping.get(normalize_source_ndc(ndc), ""))
                atc = chunk.ndc.map(cache)
                begin = pd.to_datetime(chunk.starttime, errors="coerce")
                end = pd.to_datetime(chunk.stoptime, errors="coerce")
                valid = begin.notna() & end.notna() & end.ge(begin)
                valid &= begin.le(chunk.hadm_id.map(end_by_hadm)) & end.ge(
                    chunk.hadm_id.map(start_by_hadm)
                )
                counts["overlapping_prescription_rows"] += int(valid.sum())
                counts["mapped_overlapping_prescription_rows"] += int((valid & atc.ne("")).sum())
                chunk["atc3"] = atc
                for hadm, label in (
                    chunk.loc[valid & atc.ne(""), ["hadm_id", "atc3"]]
                    .drop_duplicates()
                    .itertuples(index=False, name=None)
                ):
                    drugs[hadm].add(label)
    groups = defaultdict(list)
    for row in admissions.sort_values(["subject_id", "start", "hadm_id"]).itertuples(index=False):
        if codes.get(row.hadm_id) and drugs.get(row.hadm_id):
            groups[row.subject_id].append(row)
    patients = []
    for subject, group in sorted(groups.items()):
        if len(group) < 2:
            continue
        if len({r.start for r in group}) != len(group):
            counts["tied_start_patients_excluded"] += 1
            continue
        chosen = [group[-1]]
        next_start = group[-1].start
        for row in reversed(group[:-1]):
            if row.end <= next_start:
                chosen.append(row)
                next_start = row.start
            else:
                counts["overlapping_history_visits_excluded"] += 1
        chosen.reverse()
        if len(chosen) >= 2:
            visits = [
                {
                    "hadm_id": r.hadm_id,
                    "time": r.start.isoformat(),
                    "end": r.end.isoformat(),
                    "codes": codes[r.hadm_id],
                    "drugs": sorted(drugs[r.hadm_id]),
                }
                for r in chosen
            ]
            assert all(a["end"] <= b["time"] for a, b in zip(visits, visits[1:]))
            patients.append({"patient_id": "nwicu:" + subject, "visits": visits})
    expected = audit["provisional_cohorts"]["unique_ndc_mapping_orders_overlapping_admission"]
    assert len(patients) == expected["patients"]
    assert sum(len(p["visits"]) for p in patients) == expected["visits"]
    assert (
        len({d for p in patients for v in p["visits"] for d in v["drugs"]})
        == expected["all_visit_medication_classes"]
    )
    assert (
        counts["mapped_overlapping_prescription_rows"]
        == audit["prescriptions"]["time_overlap_unambiguous_mapped_rows"]
    )
    metadata = {
        "dataset": "NWICU 0.1.0",
        "archive": str(archive_path),
        "archive_sha256": audit["archive_sha256"],
        "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "preparation_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inspection_source_sha256": hashlib.sha256(
            Path(__file__).with_name("nwicu_inspection.py").read_bytes()
        ).hexdigest(),
        "ndc_mapping_sha256": audit["mapping"]["resource_sha256"],
        "patients": len(patients),
        "visits": expected["visits"],
        "mean_target_drugs": float(np.mean([len(p["visits"][-1]["drugs"]) for p in patients])),
        "counts": dict(counts),
        "cohort": "Whole hospital admissions, not ICU-only; >=2 eligible non-overlapping admissions per patient.",
        "inputs": "Whole-visit ICD diagnosis codes, available previous admissions and medication sets. No hospital procedure ICD table; no invented procedures.",
        "labels": "Mapped medication order types whose start/stop interval overlaps admission, stop>=start; not verified administrations.",
        "mapping": audit["mapping"]["rule"],
        "mapping_row_coverage_all_prescriptions": audit["prescriptions"][
            "unambiguous_mapping_row_coverage"
        ],
        "elapsed_seconds": time.monotonic() - started,
    }
    with destination.open("wb") as stream:
        pickle.dump({"patients": patients, "metadata": metadata}, stream, protocol=5)
    metadata["data_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
    atomic_json(destination.with_suffix(".json"), metadata)
    content = load_split(destination, resources=resources)
    records, splits, code_vocab, drug_vocab, ddi, ehr, split_meta = content
    assert len(records) == len(patients)
    assert not set(splits["train"]) & set(splits["validation"])
    assert not set(splits["train"]) & set(splits["test"])
    assert not set(splits["validation"]) & set(splits["test"])
    train_codes = {c for i in splits["train"] for v in patients[i]["visits"] for c in v["codes"]}
    train_drugs = {d for i in splits["train"] for v in patients[i]["visits"] for d in v["drugs"]}
    assert set(code_vocab) - {"<PAD>", "<UNK>", "<GAP>"} == train_codes
    assert set(drug_vocab) == train_drugs
    assert ddi.sum() > 0 and np.all(ddi == ddi.T) and not ddi.diagonal().any()
    atomic_json(output / "split.json", split_meta)
    atomic_json(output / "vocabularies.json", {"codes": code_vocab, "medications": drug_vocab})
    print(
        json.dumps(
            {"data": str(destination), "metadata": metadata, "split": split_meta},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return destination

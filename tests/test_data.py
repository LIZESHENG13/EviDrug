"""Synthetic source tables test the three distinct cohort preparation rules."""

import gzip
import hashlib
import json
import pickle
import zipfile
from collections import defaultdict
from importlib.resources import files

import pandas as pd

from evidrug.data import mimic_iv, nwicu, sicdb
from evidrug.data.nwicu_inspection import normalize_source_ndc
from evidrug.data.records import load_split


def resource_fixture(directory):
    directory.mkdir()
    pd.DataFrame(
        [
            ["00000000001", "A01AA01"],
            ["00000000001", "B01AA01"],
            ["00000000002", "C01AA01"],
            ["00000000003", "B01AA01"],
        ],
        columns=["NDC", "ATC"],
    ).to_csv(directory / "NDC_to_ATC.csv", index=False)
    pd.DataFrame([["B01AA01", "C01AA01"]], columns=["left", "right"]).to_csv(
        directory / "DDI_GAMENet.csv", index=False
    )
    return directory


def zipped_tables(path, prefix, tables):
    with zipfile.ZipFile(path, "w") as archive:
        for name, frame in tables.items():
            archive.writestr(
                prefix + name + ".csv.gz", gzip.compress(frame.to_csv(index=False).encode())
            )


def hospital_tables(nw=False):
    tables = defaultdict(list)
    for patient in range(4):
        for step in range(2):
            case = f"synthetic-{patient}-{step}"
            start, end = (
                f"2020-01-{1 + 3 * step:02d} 00:00:00",
                f"2020-01-{2 + 3 * step:02d} 00:00:00",
            )
            tables["admissions"].append(
                dict(
                    subject_id=f"synthetic-{patient}", hadm_id=case, admittime=start, dischtime=end
                )
            )
            tables["diagnoses_icd"].append(
                dict(
                    subject_id=f"synthetic-{patient}",
                    hadm_id=case,
                    icd_code=f"SYN{patient}",
                    icd_version="10",
                    seq_num="1",
                )
            )
            for ndc in ("00000000001", "00000000002", "00000000003"):
                tables["prescriptions"].append(
                    dict(hadm_id=case, ndc=ndc, starttime=start, stoptime=end)
                )
    if nw:
        # Mapped but reversed interval: excluded from labels by NWICU's rule.
        tables["prescriptions"].append(
            dict(
                hadm_id="synthetic-0-0",
                ndc="00000000002",
                starttime="2020-01-02",
                stoptime="2020-01-01",
            )
        )
    else:
        tables["procedures_icd"].append(
            dict(hadm_id="synthetic-0-0", icd_code="SYNPROC", icd_version="10", seq_num="1")
        )
    return {name: pd.DataFrame(rows) for name, rows in tables.items()}


def test_mimic_union_mapping_and_train_only_vocabulary(tmp_path):
    resources = resource_fixture(tmp_path / "resources")
    archive = tmp_path / "mimic.zip"
    zipped_tables(archive, "mimic-iv-3.1/hosp/", hospital_tables())
    output = tmp_path / "patients.pkl"
    mimic_iv.prepare(archive, output, resources)
    with output.open("rb") as stream:
        raw = pickle.load(stream)
    assert len(raw["patients"]) == 4
    assert raw["patients"][0]["visits"][0]["drugs"] == ["A01A", "B01A", "C01A"]
    records, splits, codes, drugs, graph, _, metadata = load_split(output, resources=resources)
    assert {key: len(value) for key, value in splits.items()} == {
        "train": 2,
        "validation": 1,
        "test": 1,
    }
    train_codes = {
        code
        for p in splits["train"]
        for visit in raw["patients"][p]["visits"]
        for code in visit["codes"]
    }
    assert set(codes) - {"<PAD>", "<UNK>", "<GAP>"} == train_codes
    assert metadata["unknown_code_occurrences"]["validation"] > 0
    assert len(drugs) == 3 and graph.sum() == 2
    assert not graph.diagonal().any()
    assert len(records) == 4


def test_nwicu_unique_mapping_and_valid_overlap(tmp_path):
    resources = resource_fixture(tmp_path / "resources")
    archive = tmp_path / "nwicu.zip"
    zipped_tables(archive, "nwicu-0.1.0/hosp/", hospital_tables(nw=True))
    audit = {
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "integrity": {"all_passed": True},
        "mapping": {
            "resource_sha256": hashlib.sha256(
                (resources / "NDC_to_ATC.csv").read_bytes()
            ).hexdigest(),
            "rule": "synthetic fixture: unique normalized NDC",
        },
        "provisional_cohorts": {
            "unique_ndc_mapping_orders_overlapping_admission": {
                "patients": 4,
                "visits": 8,
                "all_visit_medication_classes": 2,
            }
        },
        "prescriptions": {
            "time_overlap_unambiguous_mapped_rows": 16,
            "unambiguous_mapping_row_coverage": 17 / 25,
        },
    }
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit))
    destination = nwicu.prepare(archive, tmp_path / "prepared", resources, audit_path)
    with destination.open("rb") as stream:
        raw = pickle.load(stream)
    assert all(visit["drugs"] == ["B01A", "C01A"] for p in raw["patients"] for visit in p["visits"])
    assert all(len(p["visits"]) == 2 for p in raw["patients"])
    assert normalize_source_ndc("1234-5678-90") == "01234567890"
    assert normalize_source_ndc("1234567890") == ""


def test_sicdb_drug_references_and_history_before_query(tmp_path):
    dictionary = json.loads(
        files("evidrug.data").joinpath("mappings/sicdb_drug_mapping.json").read_text()
    )
    name = next(name for name, value in dictionary.items() if value["atc3"])
    source = tmp_path / "sicdb-1.0.8"
    source.mkdir()
    cases, medication = [], []
    for patient in range(4):
        for step, (start, duration) in enumerate(((0, 100), (50, 100), (200, 10))):
            case = f"synthetic-{patient}-{step}"
            cases.append(
                dict(
                    CaseID=case,
                    PatientID=f"synthetic-{patient}",
                    ICD10Main="i 10.1",
                    OffsetAfterFirstAdmission=start,
                    TimeOfStay=duration,
                )
            )
            for drug, amount, rate, offset in (
                ("1", 1, 0, 0),
                ("1", 0, 1, 1),
                ("1", 1, 0, -1),
                ("2", 1, 0, 0),
                ("3", 10, 0, 0),
            ):
                medication.append(
                    dict(
                        CaseID=case, DrugID=drug, Amount=amount, AmountPerMinute=rate, Offset=offset
                    )
                )
    pd.DataFrame(cases).to_csv(source / "cases.csv.gz", index=False)
    pd.DataFrame(medication).to_csv(source / "medication.csv.gz", index=False)
    pd.DataFrame(
        [
            dict(ReferenceName="Drug", ReferenceGlobalID="1", ReferenceValue=name),
            dict(
                ReferenceName="Drug",
                ReferenceGlobalID="2",
                ReferenceValue="synthetic-unmapped-name",
            ),
            dict(ReferenceName="Food", ReferenceGlobalID="3", ReferenceValue="synthetic-food"),
        ]
    ).to_csv(source / "d_references.csv.gz", index=False)
    destination = sicdb.prepare(source, tmp_path / "prepared")
    with destination.open("rb") as stream:
        raw = pickle.load(stream)
    assert len(raw["patients"]) == 4
    # Earlier histories may overlap each other; each must finish before the query.
    assert all(len(p["visits"]) == 3 for p in raw["patients"])
    assert all(v["codes"] == ["D10:I10.1"] for p in raw["patients"] for v in p["visits"])
    assert all(v["drugs"] == dictionary[name]["atc3"] for p in raw["patients"] for v in p["visits"])
    assert raw["metadata"]["counts"]["non_drug_reference_rows_excluded"] == 12
    assert raw["metadata"]["counts"]["positive_rows_repeated_patients"] == 36

"""NWICU archive integrity, schema, mapping coverage and eligibility audit."""

import gzip
import hashlib
import json
import re
import time
import zipfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .mimic_iv import normalize_ndc


def digest(stream):
    h = hashlib.sha256()
    while block := stream.read(8 * 1024 * 1024):
        h.update(block)
    return h.hexdigest()


@contextmanager
def table(archive, name, columns=None, chunksize=None):
    matches = [n for n in archive.namelist() if n.endswith("/" + name + ".csv.gz")]
    if len(matches) != 1:
        raise ValueError(f"Expected one table: {name}")
    with archive.open(matches[0]) as raw, gzip.GzipFile(fileobj=raw) as stream:
        yield pd.read_csv(
            stream, usecols=columns, dtype=str, keep_default_na=False, chunksize=chunksize
        )


def distribution(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {"n": 0}
    return {
        "n": int(len(a)),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "min": float(a.min()),
        "max": float(a.max()),
        "p25": float(np.quantile(a, 0.25)),
        "p75": float(np.quantile(a, 0.75)),
    }


def normalize_source_ndc(value):
    """Convert explicit 4-4-2/5-3-2/5-4-1/5-4-2 package NDCs to 11 digits.

    Do not guess segment boundaries for truncated or product-only codes.
    https://www.fda.gov/drugs/development-approval-process-drugs/national-drug-code-database-background-information
    """
    value = str(value).strip()
    if re.fullmatch(r"\d{11}", value):
        return value if int(value) else ""
    parts = value.split("-")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        return ""
    if tuple(map(len, parts)) not in {(4, 4, 2), (5, 3, 2), (5, 4, 1), (5, 4, 2)}:
        return ""
    normalized = "".join(p.zfill(width) for p, width in zip(parts, (5, 4, 2)))
    return normalized if int(normalized) else ""


def inspect(archive_path, output, resources):
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Aggregate feasibility audit only; no split, training, model selection, or predictions.",
        "archive": str(archive_path.resolve()),
        "archive_bytes": archive_path.stat().st_size,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    with archive_path.open("rb") as stream:
        result["archive_sha256"] = digest(stream)
    with zipfile.ZipFile(archive_path) as archive:
        root = archive.namelist()[0].split("/")[0] + "/"
        manifest = archive.read(root + "SHA256SUMS.txt").decode()
        checks = []
        for line in manifest.splitlines():
            if not line.strip():
                continue
            expected, relative = line.split(maxsplit=1)
            with archive.open(root + relative) as stream:
                actual = digest(stream)
            checks.append(
                {
                    "file": relative,
                    "expected_sha256": expected,
                    "actual_sha256": actual,
                    "passed": actual == expected,
                }
            )
        result["integrity"] = {"all_passed": all(x["passed"] for x in checks), "files": checks}
        if not result["integrity"]["all_passed"]:
            raise ValueError("Archive checksum mismatch")
        schemas = {}
        for entry in archive.infolist():
            if entry.filename.endswith(".csv.gz"):
                with archive.open(entry) as raw, gzip.GzipFile(fileobj=raw) as stream:
                    schemas[entry.filename[len(root) :]] = pd.read_csv(
                        stream, nrows=0
                    ).columns.tolist()
        result["schemas"] = schemas
        print("All archive checksums passed.", flush=True)

        with table(archive, "patients", ["subject_id"]) as patients:
            patient_ids = set(patients.subject_id)
            patient_rows = len(patients)
        with table(
            archive, "admissions", ["subject_id", "hadm_id", "admittime", "dischtime"]
        ) as admissions:
            admissions = admissions.copy()
        admissions["start"] = pd.to_datetime(admissions.admittime, errors="coerce")
        admissions["end"] = pd.to_datetime(admissions.dischtime, errors="coerce")
        hadms = set(admissions.hadm_id)
        subject_by_hadm = dict(zip(admissions.hadm_id, admissions.subject_id))
        start_by_hadm = dict(zip(admissions.hadm_id, admissions.start))
        end_by_hadm = dict(zip(admissions.hadm_id, admissions.end))
        visit_counts = admissions.groupby("subject_id").hadm_id.nunique()
        repeated = set(visit_counts[visit_counts.ge(2)].index)
        bad_admission_time = (
            admissions.start.isna() | admissions.end.isna() | admissions.end.lt(admissions.start)
        )
        result["admissions"] = {
            "patient_table_rows": patient_rows,
            "unique_patient_ids": len(patient_ids),
            "rows": len(admissions),
            "unique_admissions": len(hadms),
            "unique_patients": int(admissions.subject_id.nunique()),
            "patients_with_at_least_two_admissions": len(repeated),
            "admissions_of_repeated_patients": int(visit_counts.loc[list(repeated)].sum()),
            "duplicate_admission_rows": int(admissions.hadm_id.duplicated().sum()),
            "unknown_patient_rows": int((~admissions.subject_id.isin(patient_ids)).sum()),
            "missing_or_invalid_time_rows": int(bad_admission_time.sum()),
            "admissions_per_patient": distribution(visit_counts),
        }
        with table(archive, "diagnoses_icd") as diagnoses:
            diagnosis_hadms = set(diagnoses.loc[diagnoses.icd_code.ne(""), "hadm_id"])
            codes_by_hadm = diagnoses.groupby("hadm_id").icd_code.nunique().to_dict()
            result["diagnoses"] = {
                "rows": len(diagnoses),
                "admissions": len(diagnosis_hadms),
                "unique_versioned_codes": len(
                    diagnoses[["icd_version", "icd_code"]].drop_duplicates()
                ),
                "icd_versions": diagnoses.icd_version.value_counts().to_dict(),
                "orphan_admission_rows": int((~diagnoses.hadm_id.isin(hadms)).sum()),
                "subject_admission_mismatch_rows": int(
                    diagnoses.subject_id.ne(diagnoses.hadm_id.map(subject_by_hadm)).sum()
                ),
            }
        with table(archive, "icustays", ["subject_id", "hadm_id", "stay_id"]) as icu:
            icu_hadms = set(icu.hadm_id)
            result["icu"] = {
                "rows": len(icu),
                "patients": int(icu.subject_id.nunique()),
                "admissions": len(icu_hadms),
            }
        print("Patient, admission, and diagnosis counts complete.", flush=True)

        mapping_path = resources / "NDC_to_ATC.csv"
        frame = pd.read_csv(mapping_path, dtype=str, keep_default_na=False)
        ndc_classes = defaultdict(set)
        for ndc, atc in frame.itertuples(index=False, name=None):
            normalized = normalize_ndc(ndc)
            if normalized and len(atc) >= 4:
                ndc_classes[normalized].add(atc[:4])
        unique_mapping = {
            ndc: next(iter(classes)) for ndc, classes in ndc_classes.items() if len(classes) == 1
        }
        del frame
        result["mapping"] = {
            "resource": str(mapping_path),
            "resource_sha256": hashlib.sha256(mapping_path.read_bytes()).hexdigest(),
            "rule": "Existing local NDC_to_ATC resource. Source package NDCs explicitly converted from 4-4-2/5-3-2/5-4-1/5-4-2 to 11 digits with segment-specific zero padding; source numeric strings must already contain 11 digits. Reject incomplete, product-only, unsegmented 10-digit, and ambiguous ATC3 codes. No name fallback, remote queries, or patient-outcome-driven mapping.",
            "ndc_format_reference": "https://www.fda.gov/drugs/development-approval-process-drugs/national-drug-code-database-background-information",
            "unique_atc3_ndcs_in_resource": len(unique_mapping),
            "ambiguous_atc3_ndcs_in_resource": sum(len(v) > 1 for v in ndc_classes.values()),
            "provisional": True,
        }
        stats = Counter()
        name_rows = Counter()
        ndc_values = set()
        ndc_lengths = Counter()
        ndc_shapes = Counter()
        labels = defaultdict(set)
        labels_within_time = defaultdict(set)
        labels_union = defaultdict(set)
        drug_hadms = set()
        ndc_cache = {}
        rx_columns = [
            "subject_id",
            "hadm_id",
            "ndc",
            "drug",
            "starttime",
            "stoptime",
            "gsn",
            "formulary_drug_cd",
        ]
        with table(archive, "prescriptions", rx_columns, 150000) as chunks:
            for chunk in chunks:
                stats["rows"] += len(chunk)
                drug_hadms.update(chunk.hadm_id)
                name_rows.update(chunk.drug)
                ndc_values.update(chunk.ndc)
                ndc_lengths.update(chunk.ndc.str.len())
                for ndc, count in chunk.ndc.value_counts().items():
                    shape = (
                        "-".join(str(len(p)) for p in ndc.split("-"))
                        if "-" in ndc
                        else f"unsegmented-{len(ndc)}"
                    )
                    ndc_shapes[shape] += int(count)
                for col in ["ndc", "drug", "gsn", "formulary_drug_cd"]:
                    stats[col + "_nonempty_rows"] += int(chunk[col].str.strip().ne("").sum())
                for ndc in chunk.ndc.unique():
                    if ndc not in ndc_cache:
                        ndc_cache[ndc] = normalize_source_ndc(ndc)
                normalized = chunk.ndc.map(ndc_cache)
                classes = normalized.map(unique_mapping).fillna("")
                has_mapping = classes.ne("")
                any_mapping = normalized.isin(ndc_classes)
                is_repeated = chunk.subject_id.isin(repeated)
                stats["standardizable_package_ndc_rows"] += int(normalized.ne("").sum())
                stats["unambiguous_ndc_mapped_rows"] += int(has_mapping.sum())
                stats["any_ndc_mapped_rows"] += int(any_mapping.sum())
                stats["ambiguous_ndc_rows"] += int((any_mapping & ~has_mapping).sum())
                stats["repeated_patient_rows"] += int(is_repeated.sum())
                stats["repeated_patient_unambiguous_mapped_rows"] += int(
                    (is_repeated & has_mapping).sum()
                )
                stats["orphan_admission_rows"] += int((~chunk.hadm_id.isin(hadms)).sum())
                stats["subject_admission_mismatch_rows"] += int(
                    chunk.subject_id.ne(chunk.hadm_id.map(subject_by_hadm)).sum()
                )
                start = pd.to_datetime(chunk.starttime, errors="coerce")
                stop = pd.to_datetime(chunk.stoptime, errors="coerce")
                admission_start = chunk.hadm_id.map(start_by_hadm)
                admission_end = chunk.hadm_id.map(end_by_hadm)
                valid_time = start.notna() & stop.notna() & stop.ge(start)
                overlap = valid_time & start.le(admission_end) & stop.ge(admission_start)
                stats["unparseable_order_start_rows"] += int(start.isna().sum())
                stats["unparseable_order_stop_rows"] += int(stop.isna().sum())
                stats["order_stop_before_start_rows"] += int(
                    (start.notna() & stop.notna() & stop.lt(start)).sum()
                )
                stats["invalid_order_time_rows"] += int((~valid_time).sum())
                stats["order_interval_outside_admission_rows"] += int((valid_time & ~overlap).sum())
                stats["time_overlap_order_rows"] += int(overlap.sum())
                stats["time_overlap_unambiguous_mapped_rows"] += int((overlap & has_mapping).sum())
                stats["repeated_time_overlap_rows"] += int((overlap & is_repeated).sum())
                stats["repeated_time_overlap_unambiguous_mapped_rows"] += int(
                    (overlap & is_repeated & has_mapping).sum()
                )
                chunk["atc3"] = classes
                for hadm, atc in (
                    chunk.loc[has_mapping, ["hadm_id", "atc3"]]
                    .drop_duplicates()
                    .itertuples(index=False, name=None)
                ):
                    labels[hadm].add(atc)
                for hadm, atc in (
                    chunk.loc[has_mapping & overlap, ["hadm_id", "atc3"]]
                    .drop_duplicates()
                    .itertuples(index=False, name=None)
                ):
                    labels_within_time[hadm].add(atc)
                for hadm, ndc in (
                    chunk.loc[any_mapping, ["hadm_id", "ndc"]]
                    .drop_duplicates()
                    .itertuples(index=False, name=None)
                ):
                    labels_union[hadm].update(ndc_classes[ndc_cache[ndc]])
        result["prescriptions"] = {
            **dict(stats),
            "admissions": len(drug_hadms),
            "unique_drug_names": len(name_rows),
            "unique_nonempty_ndcs": len(ndc_values - {""}),
            "raw_ndc_length_distribution": {str(k): v for k, v in sorted(ndc_lengths.items())},
            "raw_ndc_segment_length_distribution": dict(sorted(ndc_shapes.items())),
            "ndc_nonempty_fraction": stats["ndc_nonempty_rows"] / stats["rows"],
            "unambiguous_mapping_row_coverage": stats["unambiguous_ndc_mapped_rows"]
            / stats["rows"],
            "any_mapping_row_coverage": stats["any_ndc_mapped_rows"] / stats["rows"],
            "repeated_patient_unambiguous_mapping_row_coverage": stats[
                "repeated_patient_unambiguous_mapped_rows"
            ]
            / stats["repeated_patient_rows"],
            "unique_unambiguous_atc3": len(set().union(*labels.values())),
        }
        print("All prescription rows and local NDC mappings scanned.", flush=True)

        def cohort_summary(medications, icu_only=False):
            mask = (
                ~bad_admission_time
                & admissions.hadm_id.isin(diagnosis_hadms)
                & admissions.hadm_id.isin(medications)
            )
            if icu_only:
                mask &= admissions.hadm_id.isin(icu_hadms)
            valid = admissions.loc[mask].sort_values(["subject_id", "start", "hadm_id"])
            groups = defaultdict(list)
            for row in valid[["subject_id", "hadm_id", "start", "end"]].itertuples(
                index=False, name=None
            ):
                groups[row[0]].append(row[1:])
            sequences = []
            overlapping = tied = 0
            for group in groups.values():
                if len(group) < 2:
                    continue
                if len({row[1] for row in group}) != len(group):
                    tied += 1
                    continue
                # Preserve the last eligible query, then the latest available
                # non-overlapping history. Every future training prefix also
                # has only completed history; never order by numeric IDs.
                chosen = [group[-1]]
                next_start = group[-1][1]
                for row in reversed(group[:-1]):
                    if row[2] <= next_start:
                        chosen.append(row)
                        next_start = row[1]
                    else:
                        overlapping += 1
                chosen.reverse()
                if len(chosen) >= 2:
                    sequences.append([row[0] for row in chosen])
            visits = [h for seq in sequences for h in seq]
            final = [seq[-1] for seq in sequences]
            return {
                "patients": len(sequences),
                "visits": len(visits),
                "visits_per_patient": distribution([len(s) for s in sequences]),
                "target_medication_classes": distribution([len(medications[h]) for h in final]),
                "target_diagnosis_codes": distribution([codes_by_hadm[h] for h in final]),
                "all_visit_medication_classes": len(set().union(*(medications[h] for h in visits)))
                if visits
                else 0,
                "all_visit_diagnosis_codes": int(
                    diagnoses.loc[diagnoses.hadm_id.isin(visits), ["icd_version", "icd_code"]]
                    .drop_duplicates()
                    .shape[0]
                ),
                "target_admissions_with_icu_stay": sum(h in icu_hadms for h in final),
                "patients_excluded_for_tied_eligible_admission_times": tied,
                "overlapping_history_visits_removed": overlapping,
            }

        result["provisional_cohorts"] = {
            "rule": "At least two distinct, diagnosed, mapped hospital admissions. Preserve last eligible query; traverse earlier admissions in reverse chronological order, retaining a history admission only if its discharge <= next retained admission start. Exclude patients with tied eligible admission start times. Aggregate feasibility counts, no fitted vocabulary or split.",
            "unique_ndc_mapping_all_orders": cohort_summary(labels),
            "unique_ndc_mapping_orders_overlapping_admission": cohort_summary(labels_within_time),
            "unique_ndc_mapping_orders_overlapping_admission_icu_admissions_only": cohort_summary(
                labels_within_time, icu_only=True
            ),
            "legacy_set_valued_ndc_mapping_all_orders": cohort_summary(labels_union),
        }

        events = Counter()
        emar_hadms = set()
        emar_stats = Counter()
        with table(
            archive,
            "emar",
            ["subject_id", "hadm_id", "event_txt", "medication", "pharmacy_id"],
            250000,
        ) as chunks:
            for chunk in chunks:
                emar_stats["rows"] += len(chunk)
                events.update(chunk.event_txt)
                emar_hadms.update(chunk.hadm_id)
                for col in ["hadm_id", "pharmacy_id", "medication"]:
                    emar_stats[col + "_nonempty_rows"] += int(chunk[col].str.strip().ne("").sum())
                emar_stats["orphan_nonempty_admission_rows"] += int(
                    (chunk.hadm_id.ne("") & ~chunk.hadm_id.isin(hadms)).sum()
                )
        result["emar"] = {
            **dict(emar_stats),
            "nonempty_admissions": len(emar_hadms - {""}),
            "event_status_counts": dict(events),
            "note": "Status counts only. No administration labels constructed; accepted status semantics and order linkage remain to be defined.",
        }
        print("All eMAR event statuses scanned.", flush=True)

        with table(archive, "d_items", ["itemid", "label", "linksto"]) as items:
            procedure_items = items[items.linksto.eq("procedureevents")]
            result["procedures"] = {
                "has_hospital_procedures_icd_table": any(
                    n.endswith("/procedures_icd.csv.gz") for n in archive.namelist()
                ),
                "has_icu_procedureevents_table": True,
                "icu_procedure_dictionary_entries": len(procedure_items),
                "note": "ICU procedure events are not hospital ICD procedure codes. No invented ICD mapping or surrogate input was created.",
            }
        procedure_stats = Counter()
        procedure_hadms = set()
        with table(archive, "procedureevents", ["hadm_id", "itemid"], 200000) as chunks:
            for chunk in chunks:
                procedure_stats["rows"] += len(chunk)
                procedure_hadms.update(chunk.hadm_id)
        result["procedures"].update(dict(procedure_stats), admissions=len(procedure_hadms))

    result["task_scope"] = [
        "Hospital-admission cohort; not restricted to ICU stays.",
        "Mapping coverage uses prescription rows, not doses or administrations.",
        "Accept a unique ATC3 class and valid admission-overlapping order interval.",
        "Whole-encounter codes define encounter-level medication reconstruction.",
    ]
    result["elapsed_seconds"] = round(time.monotonic() - started, 2)
    (output / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    cohort = result["provisional_cohorts"]["unique_ndc_mapping_orders_overlapping_admission"]
    print(
        f"NWICU audit: {cohort['patients']:,} eligible patients, {cohort['visits']:,} admissions.",
        flush=True,
    )
    return result

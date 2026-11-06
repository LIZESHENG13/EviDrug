"""Verify public aggregate evidence without downloading patient records."""

import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / name).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_read(name):
    with (ROOT / name).open() as stream:
        return list(csv.DictReader(stream))


def main():
    manifest = read("MANIFEST.json")
    failures = []
    for name, record in manifest["files"].items():
        path = ROOT / name
        if not path.is_file() or sha(path) != record["sha256"]:
            failures.append(name)
    assert not failures, f"Changed/missing files: {failures}"
    for forbidden in ("*.npz", "*.npy", "*.pkl", "*.pt", "*.pth"):
        included = [name for name in manifest["files"] if Path(name).match(forbidden)]
        assert not included, included
    rows = csv_read("results/test_metrics_by_seed.csv")
    assert len(rows) == 180
    assert sum(row["kind"] == "baseline" for row in rows) == 36
    assert sum(row["kind"] == "evidrug" for row in rows) == 144
    selected = read("results/selected_models.json")["entries"]
    assert len(selected) == 9
    aggregate = read("results/aggregate_metrics.json")
    metrics = ("prauc", "f1", "jaccard", "ddi")
    result_markdown = ["| Dataset | AP | F1 | Jaccard | DDI |", "|---|---:|---:|---:|---:|"]
    for dataset in ("mimic_iv_full", "nwicu", "sicdb"):
        group = [r for r in selected if r["dataset"] == dataset]
        assert {r["seed"] for r in group} == {0, 1, 2}
        cells = []
        for metric in metrics:
            values = [r["test"][metric] for r in group]
            mean, sd = statistics.mean(values), statistics.stdev(values)
            reference = aggregate[dataset]["full"][metric]
            assert abs(mean - reference["mean"]) < 1e-12
            assert abs(sd - reference["std"]) < 1e-12
            cells.append(f"{mean:.4f} ± {sd:.4f}")
        result_markdown.append("| " + dataset + " | " + " | ".join(cells) + " |")
        for entry in group:
            matches = [
                r
                for r in rows
                if r["dataset"] == dataset and r["checkpoint_sha256"] == entry["checkpoint_sha256"]
            ]
            assert len(matches) == 1
            for metric in metrics:
                assert abs(float(matches[0][metric]) - entry["test"][metric]) < 1e-12
    efficiency = csv_read("results/efficiency_measurements.csv")
    assert len(efficiency) == 39
    reductions = {}
    for dataset in ("mimic_iv_full", "nwicu", "sicdb"):
        group = [r for r in efficiency if r["dataset"] == dataset]
        assert len(group) == 13
        for row in group:
            seconds = [float(row[f"pass_{i}_seconds"]) for i in (1, 2, 3)]
            expected = statistics.median(seconds) / int(row["patients"]) * 1000
            assert abs(expected - float(row["ms_per_patient_B16"])) < 1e-10
            assert int(row["batch_size"]) == 16 and int(row["cpu_threads"]) == 1
            main = [r for r in selected if r["dataset"] == dataset and r["seed"] == 0][0]
            if row["method"] == "EviDrug":
                assert main["total_parameters"] == int(row["parameters"])
        current = next(r for r in group if r["method"] == "EviDrug")
        meta = next(r for r in group if r["method"] == "MetaDrug")
        reductions[dataset] = {
            "parameter_reduction_percent": 100
            * (1 - int(current["parameters"]) / int(meta["parameters"])),
            "latency_reduction_percent": 100
            * (1 - float(current["ms_per_patient_B16"]) / float(meta["ms_per_patient_B16"])),
        }
    checks = ROOT / "checks"
    checks.mkdir(exist_ok=True)
    (checks / "MAIN_RESULTS.md").write_text("\n".join(result_markdown) + "\n")
    report = {
        "passed": True,
        "checksummed_files": len(manifest["files"]),
        "model_modules": len(list((ROOT / "src/evidrug/models").glob("*.py"))),
        "aggregate_test_rows": len(rows),
        "main_models": len(selected),
        "baseline_rows": 36,
        "efficiency_models": len(efficiency),
        "efficiency_passes": 117,
        "reductions_vs_metadrug": reductions,
        "verification_scope": "Checksums and reconstruction from exported cohort/seed aggregates. Patient-level metric replay requires authorized source data and local prediction arrays.",
        "patient_level_arrays_checked": False,
    }
    (checks / "ARTIFACT_AUDIT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

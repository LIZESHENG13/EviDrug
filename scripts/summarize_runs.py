"""Export selected test metrics without reading patient-level arrays."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.runs.rglob("selected.json")):
        result = path.parent / "test_metrics.json"
        if not result.exists():
            continue
        selected = json.loads(path.read_text())
        score = json.loads(result.read_text())
        rows.append(
            {
                **{key: selected[key] for key in ("dataset", "arm", "continuation", "seed")},
                **{key: score[key] for key in ("patients", "prauc", "f1", "jaccard", "ddi")},
                "checkpoint_sha256": score["checkpoint_sha256"],
            }
        )
    if not rows:
        raise ValueError("No selected test evaluations found.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} selected test evaluations to {args.output}")


if __name__ == "__main__":
    main()

"""Reproduce clinical subgroup and prescription-action summaries from local arrays."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evidrug.analysis import decision_terms, micro_scores, patient_terms, summarize_terms
from evidrug.data.records import load_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--resources", default="resources")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records, splits, _, drugs, graph, *_ = load_split(args.data, resources=args.resources)
    with np.load(args.predictions) as saved:
        y, p, ids = saved["labels"], saved["probabilities"], saved["indices"]
    if not np.array_equal(ids, splits["test"]):
        raise ValueError("Prediction patient order differs from the prepared test split.")
    history = np.zeros_like(y, bool)
    burden, depth, prior_ddi = [], [], []
    for row, patient in enumerate(ids):
        visits = records[int(patient)]["visits"]
        for visit in visits[:-1]:
            history[row, visit["drugs"]] = True
        previous = visits[-2]["drugs"]
        burden.append(len(previous))
        depth.append(len(visits) - 1)
        prior_ddi.append(bool(graph[np.ix_(previous, previous)].sum() > 0))
    burden, depth, prior_ddi = map(np.asarray, (burden, depth, prior_ddi))
    decisions = p >= 0.3
    terms = patient_terms(y, p, graph)
    args.output.mkdir(parents=True, exist_ok=True)
    actions = {
        "continued": micro_scores(y, decisions, history),
        "new": micro_scores(y, decisions, ~history),
    }
    actions["average_f1"] = (actions["continued"]["f1"] + actions["new"]["f1"]) / 2
    (args.output / "continued_new.json").write_text(json.dumps(actions, indent=2) + "\n")
    groups = []
    for family, masks in (
        (
            "prior_burden",
            [
                ("0-5", burden <= 5),
                ("6-10", (burden >= 6) & (burden <= 10)),
                (">=11", burden >= 11),
            ],
        ),
        (
            "history_depth",
            [("1-2", depth <= 2), ("3-5", (depth >= 3) & (depth <= 5)), (">=6", depth >= 6)],
        ),
        ("prior_ddi", [("absent", ~prior_ddi), ("present", prior_ddi)]),
    ):
        for label, mask in masks:
            if mask.any():
                groups.append(
                    {
                        "family": family,
                        "group": label,
                        **summarize_terms(terms[mask], decisions[mask].sum(1), y[mask].sum(1)),
                        "interacting_pairs_per_patient": float(terms[mask, 3].mean()),
                    }
                )
    with (args.output / "clinical_groups.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(groups[0]))
        writer.writeheader()
        writer.writerows(groups)
    thresholds = []
    for threshold in np.arange(1, 10) / 10:
        selected = p.astype(np.float64) >= threshold
        changed = decision_terms(y, selected, graph, terms[:, 0])
        thresholds.append(
            {"threshold": float(threshold), **summarize_terms(changed, selected.sum(1), y.sum(1))}
        )
    with (args.output / "thresholds.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(thresholds[0]))
        writer.writeheader()
        writer.writerows(thresholds)
    print(
        f"Exported aggregate clinical analyses for {len(ids)} patients and {len(drugs)} medications."
    )


if __name__ == "__main__":
    main()

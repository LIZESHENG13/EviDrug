"""Full-cohort patient partitions, train-only vocabularies and padded visit records."""

import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def patient_split(patients, seed=0):
    order = np.random.default_rng(seed).permutation(len(patients))
    development = int(0.8 * len(order))
    train = int(0.9 * development)
    return patients, {
        "train": order[:train],
        "validation": order[train:development],
        "test": order[development:],
    }


def load_split(path, seed=0, resources="resources"):
    """Encode a trusted local prepared pickle using training patients only."""
    with Path(path).open("rb") as stream:
        raw = pickle.load(stream)
    patients, splits = patient_split(raw["patients"], seed)
    train = [patients[i] for i in splits["train"]]
    code_vocab = {"<PAD>": 0, "<UNK>": 1, "<GAP>": 2}
    code_vocab.update(
        {
            code: i + 3
            for i, code in enumerate(
                sorted(
                    {c for patient in train for visit in patient["visits"] for c in visit["codes"]}
                )
            )
        }
    )
    drug_vocab = {
        drug: i
        for i, drug in enumerate(
            sorted({d for patient in train for visit in patient["visits"] for d in visit["drugs"]})
        )
    }
    partition = {int(i): name for name, indices in splits.items() for i in indices}
    encoded, unknown_codes, unknown_labels = [], Counter(), Counter()
    for i, patient in enumerate(patients):
        visits = []
        for visit in patient["visits"]:
            original = visit["codes"]
            codes = [code_vocab.get(code, 1) for code in original]
            labels = [drug_vocab[drug] for drug in visit["drugs"] if drug in drug_vocab]
            unknown_codes[partition[i]] += sum(code == 1 for code in codes)
            unknown_labels[partition[i]] += sum(drug not in drug_vocab for drug in visit["drugs"])
            visits.append(
                {
                    "codes": codes or [1],
                    "diagnoses": [code_vocab.get(c, 1) for c in original if c.startswith("D")]
                    or [1],
                    "procedures": [code_vocab.get(c, 1) for c in original if c.startswith("P")]
                    or [1],
                    "drugs": labels,
                    "retrieval_codes": frozenset(original),
                    "hadm_id": visit["hadm_id"],
                }
            )
        encoded.append(
            {
                "patient_id": patient["patient_id"],
                "visits": visits,
                "n_codes": sum(len(v["codes"]) for v in patient["visits"]),
            }
        )
    ddi = np.zeros((len(drug_vocab), len(drug_vocab)), np.float32)
    for left, right in pd.read_csv(Path(resources) / "DDI_GAMENet.csv", dtype=str).itertuples(
        index=False, name=None
    ):
        left, right = left[:4], right[:4]
        if left in drug_vocab and right in drug_vocab and left != right:
            a, b = drug_vocab[left], drug_vocab[right]
            ddi[a, b] = ddi[b, a] = 1
    ehr = np.zeros_like(ddi)
    for patient in splits["train"]:
        for visit in encoded[patient]["visits"]:
            ehr[np.ix_(visit["drugs"], visit["drugs"])] = 1
    np.fill_diagonal(ehr, 0)
    split_ids = {
        name: [encoded[i]["patient_id"] for i in indices] for name, indices in splits.items()
    }
    metadata = {
        "seed": seed,
        "limit": 0,
        "sizes": {k: len(v) for k, v in splits.items()},
        "code_vocabulary_size": len(code_vocab),
        "medication_vocabulary_size": len(drug_vocab),
        "unknown_code_occurrences": dict(unknown_codes),
        "unseen_drug_occurrences": dict(unknown_labels),
        "truncation": {},
        "max_visits": 0,
        "max_codes": 0,
        "split_sha256": hashlib.sha256(json.dumps(split_ids, sort_keys=True).encode()).hexdigest(),
        "ddi_edges": int(ddi.sum() / 2),
    }
    return encoded, splits, code_vocab, drug_vocab, ddi, ehr, metadata


def collate(items, records, n_drugs):
    """Pad encounter fields used by EviDrug, without baseline-only peer tensors."""
    batch_size = len(items)
    visits = max(len(record["visits"]) for _, record, _ in items)
    codes = max(len(visit["codes"]) for _, record, _ in items for visit in record["visits"])
    tensors = {
        name: np.zeros((batch_size, visits, codes), np.int64)
        for name in ("codes", "diagnoses", "procedures")
    }
    labels = np.zeros((batch_size, visits, n_drugs), np.float32)
    visit_mask = np.zeros((batch_size, visits), bool)
    for row, (_, record, _) in enumerate(items):
        for step, visit in enumerate(record["visits"]):
            for name, values in tensors.items():
                values[row, step, : len(visit[name])] = visit[name]
            labels[row, step, visit["drugs"]] = 1
            visit_mask[row, step] = True
    return {
        "index": torch.tensor([int(i) for i, _, _ in items]),
        **{name: torch.from_numpy(values) for name, values in tensors.items()},
        "labels": torch.from_numpy(labels),
        "visit_mask": torch.from_numpy(visit_mask),
        "n_codes": torch.tensor([record["n_codes"] for _, record, _ in items]),
    }

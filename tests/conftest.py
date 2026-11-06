"""Generated records only; the test suite reads no clinical dataset."""

import hashlib
import json

import numpy as np
import pytest
import torch

torch.set_num_threads(1)


def synthetic():
    codes = {"<PAD>": 0, "<UNK>": 1, "<GAP>": 2}
    codes.update({f"D10:synthetic{i}": i + 3 for i in range(12)})
    codes.update({f"P10:synthetic{i}": i + 15 for i in range(6)})
    drugs = {f"synthetic_drug{i}": i for i in range(6)}
    records = []
    for patient in range(16):
        visits = []
        for step in range(3):
            diagnosis = [3 + (patient + step) % 12, 3 + (patient * 3 + step + 1) % 12]
            procedure = [15 + (patient + step) % 6]
            tokens = list(dict.fromkeys(diagnosis + procedure))
            visits.append(
                {
                    "hadm_id": f"synthetic-{patient}-{step}",
                    "codes": tokens,
                    "diagnoses": diagnosis,
                    "procedures": procedure,
                    "retrieval_codes": frozenset(f"synthetic_code{c}" for c in tokens),
                    "drugs": sorted({patient % 6, (patient + step + 1) % 6, 0}),
                }
            )
        records.append(
            {
                "patient_id": f"synthetic-patient-{patient}",
                "visits": visits,
                "n_codes": sum(len(v["codes"]) for v in visits),
            }
        )
    splits = {"train": np.arange(12), "validation": np.arange(12, 14), "test": np.arange(14, 16)}
    graph = np.zeros((6, 6), np.float32)
    for a, b in ((0, 1), (0, 2), (2, 3)):
        graph[a, b] = graph[b, a] = 1
    ehr = np.ones_like(graph) - np.eye(6, dtype=np.float32)
    digest = hashlib.sha256(
        json.dumps({k: v.tolist() for k, v in splits.items()}, sort_keys=True).encode()
    ).hexdigest()
    return records, splits, codes, drugs, graph, ehr, {"split_sha256": digest}


@pytest.fixture
def content():
    return synthetic()

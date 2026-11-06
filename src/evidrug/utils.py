"""Shared tensor, random-state and provenance helpers."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch


def last(values, mask):
    return values[torch.arange(len(values), device=values.device), mask.sum(1) - 1]


def target(batch):
    return last(batch["labels"], batch["visit_mask"])


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_batch(batch, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }

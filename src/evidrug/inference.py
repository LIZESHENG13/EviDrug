"""Complete online recommendation with fixed training references."""

import numpy as np
import torch

from .data.batching import loader_for
from .evidence.retrieval import CaseMemoryIndex
from .models import EviDrug
from .training import association_table, feature_batch
from .utils import device_batch, target


class Predictor:
    """Retain the training reference index, not query predictions or query evidence."""

    def __init__(self, content, checkpoint, arm="full", device="cpu"):
        self.content, self.arm, self.device = content, arm, device
        records, splits, codes, drugs = content[:4]
        self.index = (
            None
            if arm == "no_memory"
            else CaseMemoryIndex(
                records, splits["train"], device=device, state_only=arm == "no_history"
            )
        )
        self.association = association_table(content)
        self.model = EviDrug(len(codes), len(drugs), arm).to(device)
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        self.model.load_state_dict(saved["state_dict"], strict=True)
        self.model.eval()
        self.runtime = None
        if arm == "full":
            from .serving import ServingRuntime

            self.runtime = ServingRuntime(self.model, records, splits["train"], len(codes))

    @torch.no_grad()
    def predict(self, queries, batch_size=16):
        records, _, _, drugs = self.content[:4]
        labels, probabilities = [], []
        for start in range(0, len(queries), batch_size):
            pairs = queries[start : start + batch_size]
            if self.runtime is not None:
                probabilities.append(self.runtime.predict(pairs).cpu().numpy())
                labels.append(
                    np.asarray(
                        [
                            [
                                int(j in records[int(p)]["visits"][int(t)]["drugs"])
                                for j in range(len(drugs))
                            ]
                            for p, t in pairs
                        ],
                        dtype=np.float32,
                    )
                )
                continue
            memory = (
                self.index.retrieve(pairs)
                if self.index
                else {
                    "refs": [],
                    "retrieved": {f"{p}:{t}": [] for p, t in pairs},
                    "scores": {f"{p}:{t}": [] for p, t in pairs},
                }
            )
            loader, _ = loader_for(self.arm)(records, len(drugs), memory, pairs, batch_size)
            batch = device_batch(next(iter(loader)), self.device)
            feature_batch(batch, self.association, self.arm, self.device)
            logits, _ = self.model(batch)
            probabilities.append(logits.sigmoid().cpu().numpy())
            labels.append(target(batch).cpu().numpy())
        return np.concatenate(labels), np.concatenate(probabilities)

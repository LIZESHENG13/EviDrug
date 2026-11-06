"""FP32 inference with immutable training-case representations and online queries."""

import math
from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F

from .data.batching import structured_collate
from .evidence.serving import ClinicalFeatureTable, InferenceCaseIndex
from .utils import device_batch


class ServingRuntime:
    """Precompute fixed training donors; never store query predictions or evidence."""

    def __init__(self, model, records, training_patients, vocabulary, build_batch_size=128):
        if model.training or model.arm != "full":
            raise ValueError("ServingRuntime requires the complete model in evaluation mode.")
        if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
            raise ValueError("ServingRuntime parity is defined for FP32 checkpoints.")
        self.model, self.records = model, records
        self.device = next(model.parameters()).device
        self.training = frozenset(map(int, training_patients))
        self.index = InferenceCaseIndex(records, training_patients)
        self.associations = ClinicalFeatureTable(
            records, training_patients, vocabulary, model.backbone.medications
        )
        self._build_donors(build_batch_size)
        with torch.no_grad():
            self.identity = model.backbone.set_drug_projection(model.backbone.drug_queries).detach()
        self._versions = tuple((id(p), p._version) for p in model.parameters())

    @torch.no_grad()
    def _build_donors(self, batch_size):
        backbone = self.model.backbone
        groups = defaultdict(list)
        sequences = []
        labels = torch.zeros((len(self.index.refs), backbone.medications), dtype=torch.bool)
        for reference, (patient, step) in enumerate(self.index.refs):
            visit = self.records[patient]["visits"][step]
            sequence = tuple(int(c) for c in visit["codes"] if c != 0)
            if len(sequence) > 512:
                raise ValueError(
                    "A donor exceeds the 512-position limit; it cannot be silently truncated."
                )
            sequences.append(sequence)
            labels[reference, visit["drugs"]] = True
            groups[max(1, len(sequence))].append(reference)
        keys = torch.empty((len(self.index.refs), backbone.width), device=self.device)
        for length, ids in sorted(groups.items()):
            for start in range(0, len(ids), batch_size):
                chosen = ids[start : start + batch_size]
                tokens = torch.zeros((len(chosen), length), dtype=torch.long, device=self.device)
                for row, reference in enumerate(chosen):
                    tokens[row, : len(sequences[reference])] = torch.tensor(
                        sequences[reference], device=self.device
                    )
                after = backbone.encode(tokens)
                projected = backbone.donor_projection(
                    torch.cat([after, torch.zeros_like(after), torch.zeros_like(after)], -1)
                )
                keys[torch.tensor(chosen, device=self.device)] = projected
        self.donor_keys, self.donor_labels = keys, labels.to(self.device)

    def prepare(self, pairs):
        if (
            self.model.training
            or tuple((id(p), p._version) for p in self.model.parameters()) != self._versions
        ):
            raise RuntimeError("Model state changed; rebuild the serving runtime.")
        if any(int(p) in self.training for p, _ in pairs):
            raise ValueError(
                "Serving uses held-out patients; training requires leave-patient-out evidence."
            )
        selected = [self.index.query(p, t) for p, t in pairs]
        empty = {
            "refs": [],
            "retrieved": {f"{p}:{t}": [] for p, t in pairs},
            "scores": {f"{p}:{t}": [] for p, t in pairs},
        }
        batch = structured_collate(pairs, self.records, self.model.backbone.medications, empty)
        width = max(1, max(len(refs) for refs, _ in selected))
        ids = torch.zeros((len(pairs), width), dtype=torch.long)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        similarities = torch.zeros_like(ids, dtype=torch.float32)
        for row, (refs, scores) in enumerate(selected):
            ids[row, : len(refs)] = torch.tensor(refs)
            mask[row, : len(refs)] = True
            similarities[row, : len(refs)] = torch.from_numpy(scores[:, 0])
        batch.update(reference_ids=ids, reference_mask=mask, reference_similarity=similarities)
        batch["association_features"] = torch.from_numpy(
            np.stack(
                [
                    self.associations.features(self.records[int(p)]["visits"][int(t)])
                    for p, t in pairs
                ]
            )
        )
        return device_batch(batch, self.device)

    @torch.no_grad()
    def predict(self, pairs):
        batch = self.prepare(pairs)
        backbone = self.model.backbone
        visits, hidden = backbone._encode_patient_visits(batch)
        context, query, previous, anchor, history = backbone._patient_context(batch, visits)
        local = backbone._drug_features(batch, hidden)
        readout = torch.einsum("bmd,mdh->bmh", local, backbone.drug_readout)
        start, keep = backbone.edit_head(context).chunk(2, -1)
        prior = (
            (1 - anchor) * (start + readout[..., 0]) + anchor * (keep + readout[..., 1])
        ).sigmoid()
        query_key = (
            backbone.query_projection(torch.cat([query, previous * 0, (query - previous) * 0], -1))[
                :, None
            ]
            + local
        )
        query_key = query_key * (1 + 0.5 * backbone.drug_metric.tanh())
        keys = self.donor_keys[batch["reference_ids"]]
        scores = torch.einsum("bmd,bkd->bkm", query_key, keys) / math.sqrt(backbone.width)
        scores = (
            scores + F.softplus(backbone.similarity_gain) * batch["reference_similarity"][..., None]
        )
        mask = batch["reference_mask"]
        attention = scores.masked_fill(~mask[..., None], -1e4).softmax(1) * mask[..., None]
        attention = attention / attention.sum(1, keepdim=True).clamp_min(1e-12)
        strength = 4 * backbone.strength_head(context).sigmoid()
        labels = self.donor_labels[batch["reference_ids"]].to(prior.dtype)
        labels = torch.where(mask[..., None], labels, 0.0)
        mass = attention.sum(1)
        positive = (attention * labels).sum(1)
        probability = ((prior + strength * positive) / (1 + strength * mass)).clamp(1e-6, 1 - 1e-6)
        evidence = {
            "prior": prior,
            "anchor": anchor,
            "opportunity_mass": mass,
            "strength": strength,
            "context": context,
            "history_frequency": history,
            "drug_features": local,
            "clinical_features": batch["association_features"],
        }
        logits, state = backbone._set_logits(probability, evidence)
        evidence.update(base_logits=logits, set_representation=state)
        return self.model.refinement(evidence, self.identity).sigmoid()

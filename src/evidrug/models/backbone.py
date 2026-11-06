"""Visit encoding, patient history, training-case memory and one set block."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..config import ARMS
from ..objectives import pairwise_rank_loss
from ..utils import last, target
from .encoder import StructuredEncoder


def mlp(inputs, hidden, outputs, dropout=0.35):
    return nn.Sequential(
        nn.Linear(inputs, hidden),
        nn.LayerNorm(hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, outputs),
    )


class ClinicalMemoryBackbone(nn.Module):
    """The prescription backbone; all clinical embeddings are learned in this study."""

    def __init__(self, vocabulary: int, medications: int, arm: str = "full"):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"Unknown ablation arm: {arm}")
        self.arm, self.dim, self.width, self.medications = arm, 128, 32, medications
        self.code_encoder = StructuredEncoder(vocabulary)
        self.code_projection = nn.Linear(192, self.dim)
        self.visit_network = mlp(self.dim + 1, 2 * self.dim, self.dim)
        self.medication_encoder = nn.Linear(medications, self.dim, bias=False)
        self.history_query = nn.Linear(self.dim, self.dim, bias=False)
        self.history_key = nn.Linear(self.dim, self.dim, bias=False)
        self.recency = nn.Parameter(torch.tensor(0.5))
        self.context_network = mlp(5 * self.dim + 2, 2 * self.dim, self.dim)
        self.edit_head = mlp(self.dim, 2 * self.dim, 2 * medications)
        self.code_key = nn.Linear(self.dim, self.width, bias=False)
        self.code_value = nn.Linear(self.dim, self.width, bias=False)
        self.drug_queries = nn.Parameter(torch.randn(medications, self.width) * self.width**-0.5)
        self.drug_readout = nn.Parameter(torch.zeros(medications, self.width, 2))
        if arm != "no_memory":
            self.query_projection = nn.Linear(3 * self.dim, self.width, bias=False)
            self.donor_projection = nn.Linear(3 * self.dim, self.width, bias=False)
            self.drug_metric = nn.Parameter(torch.zeros(medications, self.width))
            self.strength_head = nn.Linear(self.dim, medications)
            self.similarity_gain = nn.Parameter(torch.tensor(1.0))
            nn.init.zeros_(self.strength_head.weight)
            nn.init.constant_(self.strength_head.bias, -1.0)
        self.set_drug_projection = nn.Linear(self.width, self.dim)
        self.set_scalar_projection = nn.Linear(2, self.dim)
        if arm != "no_set":
            layer = nn.TransformerEncoderLayer(
                self.dim,
                nhead=4,
                dim_feedforward=256,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.set_encoder = nn.TransformerEncoder(layer, num_layers=1)
            self.set_output = nn.Sequential(
                nn.Linear(self.dim, self.dim), nn.GELU(), nn.Linear(self.dim, 1)
            )
            nn.init.zeros_(self.set_output[-1].weight)
            nn.init.zeros_(self.set_output[-1].bias)
        nn.init.zeros_(self.edit_head[-1].weight)
        with torch.no_grad():
            self.edit_head[-1].bias[:medications].fill_(-3.0)
            self.edit_head[-1].bias[medications:].fill_(2.0)

    def encode(self, tokens):
        leading = tokens.shape[:-1]
        width = tokens.shape[-1]
        flat = tokens.reshape(-1, width)
        valid = flat != 0
        count = valid.sum(-1, keepdim=True)
        empty = ~valid.any(-1)
        if bool(empty.any()):
            flat = flat.clone()
            valid = valid.clone()
            flat[empty, 0] = 1
            valid[empty, 0] = True
        serial = valid.long().cumsum(-1).sub(1).clamp_min(0)
        visit_ids = torch.zeros_like(flat)
        hidden = self.code_encoder(flat, visit_ids, serial, valid)
        pooled = (hidden * valid[..., None]).sum(-2) / count.clamp_min(1).sqrt()
        encoded = self.visit_network(
            torch.cat([self.code_projection(pooled), count.float().log1p()], -1)
        )
        encoded = encoded * (count > 0)
        return encoded.reshape(*leading, self.dim)

    def _encode_patient_visits(self, batch):
        input_ids = batch["ehr_input_ids"]
        attention_mask = input_ids != 0
        hidden = self.code_encoder(
            input_ids, batch["ehr_visit_ids"], batch["ehr_serial_ids"], attention_mask
        )
        batch_size, n_visits = batch["codes"].shape[:2]
        visit_map = batch["ehr_visit_map"]
        valid = (visit_map >= 0) & attention_mask
        rows = torch.arange(batch_size, device=input_ids.device)[:, None]
        flat_ids = (rows * n_visits + visit_map.clamp_min(0))[valid]
        sums = hidden.new_zeros((batch_size * n_visits, hidden.shape[-1]))
        sums = sums.index_add(0, flat_ids, hidden[valid])
        counts = hidden.new_zeros(batch_size * n_visits)
        counts = counts.index_add(0, flat_ids, torch.ones_like(flat_ids, dtype=hidden.dtype))
        counts = counts.reshape(batch_size, n_visits, 1)
        pooled = sums.reshape(batch_size, n_visits, -1) / counts.clamp_min(1).sqrt()
        visits = self.visit_network(
            torch.cat([self.code_projection(pooled), counts.float().log1p()], -1)
        )
        visits = visits * (counts > 0)
        return visits, hidden

    def _patient_context(self, batch, visits):
        mask = batch["visit_mask"].clone()
        lengths = mask.sum(1)
        rows = torch.arange(len(mask), device=mask.device)
        mask[rows, lengths - 1] = False
        labels = torch.where(mask[..., None], batch["labels"], 0.0)
        query = last(visits, batch["visit_mask"])
        previous = visits[rows, (lengths - 2).clamp_min(0)] * (lengths >= 2)[:, None]
        anchor = labels[rows, (lengths - 2).clamp_min(0)] * (lengths >= 2)[:, None]
        age = (
            lengths[:, None] - 1 - torch.arange(visits.shape[1], device=visits.device)[None]
        ).clamp_min(0)
        score = torch.einsum(
            "bd,bvd->bv", self.history_query(query), self.history_key(visits)
        ) / math.sqrt(self.dim)
        score = score - F.softplus(self.recency) * age.float().log1p()
        weight = score.masked_fill(~mask, -1e4).softmax(-1) * mask
        weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-12)
        history = torch.einsum("bv,bvd->bd", weight, visits)
        frequency = torch.einsum("bv,bvm->bm", weight, labels)
        context = self.context_network(
            torch.cat(
                [
                    query,
                    previous,
                    query - previous,
                    history,
                    self.medication_encoder((anchor + frequency) / 2),
                    mask.sum(1, keepdim=True).float().log1p(),
                    anchor.sum(1, keepdim=True) / self.medications,
                ],
                -1,
            )
        )
        return context, query, previous, anchor, frequency

    def _drug_features(self, batch, hidden):
        input_ids = batch["ehr_input_ids"]
        visit_map = batch["ehr_visit_map"]
        lengths = batch["visit_mask"].sum(1)
        current_visit = lengths - 1
        valid = (visit_map == current_visit[:, None]) & (input_ids != 0)
        codes = self.code_projection(hidden)
        scores = torch.einsum("btd,md->bmt", self.code_key(codes), self.drug_queries) / math.sqrt(
            self.width
        )
        attention = scores.masked_fill(~valid[:, None], -1e4).softmax(-1) * valid[:, None]
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-12)
        values = self.code_value(codes).tanh()
        return torch.einsum("bmt,btd->bmd", attention, values)

    def _memory_distribution(self, batch):
        visits, hidden = self._encode_patient_visits(batch)
        context, query, previous, anchor, frequency = self._patient_context(batch, visits)
        features = self._drug_features(batch, hidden)
        local = torch.einsum("bmd,mdh->bmh", features, self.drug_readout)
        start, keep = self.edit_head(context).chunk(2, -1)
        prior = ((1 - anchor) * (start + local[..., 0]) + anchor * (keep + local[..., 1])).sigmoid()
        details = {
            "prior": prior,
            "anchor": anchor,
            "context": context,
            "query": query,
            "previous": previous,
            "history_frequency": frequency,
            "drug_features": features,
        }
        if self.arm == "no_memory":
            zero = torch.zeros_like(prior)
            return prior.clamp(1e-6, 1 - 1e-6), {
                **details,
                "attention": zero[:, None],
                "opportunity_mass": zero,
                "strength": zero,
            }
        before = self.encode(batch["transition_before_codes"])
        after = self.encode(batch["transition_after_codes"])
        # Attention uses current clinical state; retrieval also considers clinical change.
        query_state = torch.cat([query, previous * 0, (query - previous) * 0], -1)
        donor_state = torch.cat([after, before * 0, (after - before) * 0], -1)
        qkey = self.query_projection(query_state)[:, None] + features
        qkey = qkey * (1 + 0.5 * self.drug_metric.tanh())
        dkey = self.donor_projection(donor_state)
        scores = torch.einsum("bmd,bkd->bkm", qkey, dkey) / math.sqrt(self.width)
        scores = (
            scores + F.softplus(self.similarity_gain) * batch["transition_similarity"][..., 0, None]
        )
        mask = batch["transition_mask"]
        attention = scores.masked_fill(~mask[..., None], -1e4).softmax(1) * mask[..., None]
        attention = attention / attention.sum(1, keepdim=True).clamp_min(1e-12)
        strength = 4 * self.strength_head(context).sigmoid()
        labels = torch.where(mask[..., None], batch["transition_after_labels"], 0.0)
        mass = attention.sum(1)
        positive = (attention * labels).sum(1)
        probability = (prior + strength * positive) / (1 + strength * mass)
        return probability.clamp(1e-6, 1 - 1e-6), {
            **details,
            "attention": attention,
            "opportunity_mass": mass,
            "strength": strength,
        }

    def _set_logits(self, probability, details):
        tokens = (
            self.set_drug_projection(self.drug_queries)[None]
            + self.set_scalar_projection(torch.stack([probability, details["anchor"]], -1))
            + details["context"][:, None]
        )
        if self.arm == "no_set":
            return torch.logit(probability), tokens
        encoded = self.set_encoder(tokens)
        delta = 2.0 * self.set_output(encoded).squeeze(-1).tanh()
        return torch.logit(probability) + delta, encoded

    def distribution(self, batch):
        probability, details = self._memory_distribution(batch)
        logits, encoded = self._set_logits(probability, details)
        return logits.sigmoid(), {
            **details,
            "base_probability": probability,
            "base_logits": logits,
            "set_representation": encoded,
        }

    def forward(self, batch):
        _, details = self.distribution(batch)
        logits = details["base_logits"]
        auxiliary = logits.new_zeros(())
        if self.training:
            labels = target(batch)
            auxiliary = 0.25 * F.binary_cross_entropy(
                details["prior"].clamp(1e-6, 1 - 1e-6), labels
            )
            auxiliary = auxiliary + 0.05 * pairwise_rank_loss(logits, labels)
        return logits, auxiliary

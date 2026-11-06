"""Residual refinement from local, predictive, identity and clinical evidence."""

import torch
from torch import nn


class EvidenceRefinement(nn.Module):
    """Shared vector gating followed by a bounded medication-wise logit correction."""

    def __init__(self, dim=128, width=32, clinical=True):
        super().__init__()
        self.local_projection = nn.Linear(width, dim, bias=False)
        self.score_projection = nn.Linear(4, dim)
        self.context_projection = nn.Linear(dim, dim, bias=False)
        self.fusion_gate = nn.Linear((5 if clinical else 4) * dim, dim)
        self.fusion_norm = nn.LayerNorm(dim)
        self.identity_scale = nn.Parameter(torch.tensor([0.1]))
        self.score = nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, 1))
        if clinical:
            self.clinical_projection = nn.Linear(6, dim, bias=False)
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, evidence, identity):
        base = evidence["set_representation"]
        local = self.local_projection(evidence["drug_features"])
        profile = (evidence["anchor"] + evidence["history_frequency"]) / 2.0
        personalized = identity[None] * (1.0 + self.identity_scale * profile[..., None])
        score = self.score_projection(
            torch.stack(
                [
                    evidence["base_logits"],
                    evidence["prior"],
                    evidence["opportunity_mass"],
                    evidence["strength"],
                ],
                -1,
            )
        )
        combined = (
            local + personalized + score + self.context_projection(evidence["context"])[:, None]
        )
        parts = [base, local, personalized, score]
        if hasattr(self, "clinical_projection"):
            clinical = self.clinical_projection(evidence["clinical_features"].to(base.dtype))
            parts.append(clinical)
            combined = combined + clinical
        gate = self.fusion_gate(torch.cat(parts, -1)).sigmoid()
        hidden = self.fusion_norm(base + gate * combined)
        return evidence["base_logits"] + 2.0 * self.score(hidden).squeeze(-1).tanh()

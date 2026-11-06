"""The selected EviDrug architecture, with physically removed ablation components."""

from torch import nn

from ..config import MODEL_VERSION
from ..objectives import pairwise_rank_loss
from ..utils import target
from .backbone import ClinicalMemoryBackbone
from .refinement import EvidenceRefinement


class EviDrug(nn.Module):
    """Clinical-memory backbone plus clinical evidence refinement.

    ``forward(batch)`` returns medication logits [B, M] and an auxiliary loss.
    ``distribution(batch)`` returns probabilities and the named evidence tensors.
    """

    def __init__(self, vocabulary: int, medications: int, arm: str = "full"):
        super().__init__()
        self.backbone = ClinicalMemoryBackbone(vocabulary, medications, arm)
        self.refinement = EvidenceRefinement(clinical=arm != "no_clinical")
        self.model_version = MODEL_VERSION
        self.arm = arm

    @property
    def fusion_parameters(self):
        return self.refinement.parameters()

    def frozen_evidence(self, batch):
        _, evidence = self.backbone.distribution(batch)
        return {**evidence, "clinical_features": batch["association_features"]}

    def logits_from_evidence(self, evidence):
        identity = self.backbone.set_drug_projection(self.backbone.drug_queries)
        return self.refinement(evidence, identity)

    def distribution(self, batch):
        evidence = self.frozen_evidence(batch)
        return self.logits_from_evidence(evidence).sigmoid(), evidence

    def forward(self, batch):
        evidence = self.frozen_evidence(batch)
        logits = self.logits_from_evidence(evidence)
        auxiliary = (
            0.05 * pairwise_rank_loss(logits, target(batch))
            if self.training
            else logits.new_zeros(())
        )
        return logits, auxiliary

"""Visit-aware structured embeddings; no clinical Transformer or pretrained weights."""

import torch
from torch import nn


class StructuredEncoder(nn.Module):
    """Map code/visit/order IDs [B, L] to structured code states [B, L, 192].

    A shared segment vector and global positions complete the embedding stack.
    Pooling and the visit MLP belong to the clinical-memory backbone.
    """

    def __init__(self, vocabulary: int, hidden: int = 192):
        super().__init__()
        self.embeddings = nn.ModuleDict(
            {
                "word_embeddings": nn.Embedding(vocabulary + 1, hidden, padding_idx=0),
                "position_embeddings": nn.Embedding(512, hidden),
                "token_type_embeddings": nn.Embedding(1, hidden),
                "LayerNorm": nn.LayerNorm(hidden, eps=1e-12),
                "dropout": nn.Dropout(0.1),
            }
        )
        self.visit_embeddings = nn.Embedding(64, hidden)
        self.serialization_embeddings = nn.Embedding(128, hidden)
        # Match the retained embedding initialization of the study implementation.
        for name in ("word_embeddings", "position_embeddings", "token_type_embeddings"):
            nn.init.normal_(self.embeddings[name].weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.embeddings["word_embeddings"].weight[0].zero_()

    def forward(self, input_ids, visit_ids, serial_ids, attention_mask=None):
        if input_ids.shape[-1] > 512:
            raise ValueError("StructuredEncoder accepts at most 512 positions.")
        inputs = (
            self.embeddings["word_embeddings"](input_ids)
            + self.visit_embeddings(visit_ids.clamp(0, 63))
            + self.serialization_embeddings(serial_ids.clamp(0, 127))
        )
        position_ids = torch.arange(input_ids.shape[-1], device=input_ids.device)[None]
        inputs = inputs + self.embeddings["token_type_embeddings"](torch.zeros_like(input_ids))
        inputs = inputs + self.embeddings["position_embeddings"](position_ids)
        return self.embeddings["dropout"](self.embeddings["LayerNorm"](inputs))

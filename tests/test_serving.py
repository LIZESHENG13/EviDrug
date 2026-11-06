"""Fixed donor precomputation preserves predictions and rejects stale weights."""

import numpy as np
import pytest
import torch

from evidrug import EviDrug
from evidrug.serving import ServingRuntime
from evidrug.training import association_table, feature_batch, load_resources, loader_for


def test_serving_parity_and_frozen_state(content):
    records, splits, codes, drugs, *_ = content
    queries = [(12, 2), (13, 2)]
    memory = load_resources(content, queries, "full", "cpu")
    batch = next(iter(loader_for("full")(records, len(drugs), memory, queries, 2)[0]))
    feature_batch(batch, association_table(content), "full", "cpu")
    model = EviDrug(len(codes), len(drugs)).eval()
    with torch.no_grad():
        model.refinement.score[-1].weight.normal_(0, 0.03)
        model.backbone.set_output[-1].weight.normal_(0, 0.03)
        expected, _ = model.distribution(batch)
    runtime = ServingRuntime(model, records, splits["train"], len(codes))
    actual = runtime.predict(queries)
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=1e-6, rtol=0)
    with torch.no_grad():
        model.refinement.score[-1].bias.add_(0.01)
    with pytest.raises(RuntimeError, match="state changed"):
        runtime.predict(queries)

"""Independent metric examples and objective/controller properties."""

import numpy as np
import pytest
import torch

from evidrug.metrics import metrics
from evidrug.objectives import next_weight, pairwise_rank_loss, quality_ok, selection_key, soft_ddi


def test_tied_average_precision_and_pair_weighted_ddi():
    graph = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    score = metrics([[1, 0, 1]], [[0.5, 0.5, 0.1]], graph)
    assert score["prauc"] == pytest.approx((0.5 + 2 / 3) / 2)
    assert score["f1"] == 0.5
    assert score["jaccard"] == pytest.approx(1 / 3)
    assert score["ddi"] == 1.0


def test_soft_ddi_gradient_matches_finite_differences():
    logits = torch.tensor(
        [[0.2, -0.3, 0.5], [-0.1, 0.4, 0.6]], dtype=torch.float64, requires_grad=True
    )
    graph = torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.float64)
    (gradient,) = torch.autograd.grad(soft_ddi(logits, graph), logits)
    for row in range(2):
        for col in range(3):
            plus, minus = logits.detach().clone(), logits.detach().clone()
            plus[row, col] += 1e-6
            minus[row, col] -= 1e-6
            numerical = (soft_ddi(plus, graph) - soft_ddi(minus, graph)) / 2e-6
            assert gradient[row, col].item() == pytest.approx(numerical.item(), abs=1e-8)


def test_ranking_rewards_larger_positive_margin():
    labels = torch.tensor([[1.0, 0.0, 1.0]])
    initial = pairwise_rank_loss(torch.zeros_like(labels), labels)
    improved = pairwise_rank_loss(torch.tensor([[2.0, -2.0, 2.0]]), labels)
    assert improved < initial
    assert pairwise_rank_loss(torch.zeros_like(labels), torch.ones_like(labels)) == 0


def test_feedback_projection_and_quality_selection():
    assert next_weight(1, 0.12, 0.1) == pytest.approx(1.4)
    assert next_weight(1, 0, 0.1) == 0
    assert next_weight(9.9, 1, 0.1) == 10
    baseline = dict(prauc=0.7, f1=0.6, jaccard=0.5, ddi=0.1, mean_predicted_drugs=10)
    eligible = {**baseline, "ddi": 0.09}
    ineligible = {**baseline, "ddi": 0.01, "f1": 0.59}
    assert quality_ok(eligible, baseline)
    assert not quality_ok(ineligible, baseline)
    assert selection_key(eligible, baseline, 2) < selection_key(ineligible, baseline, 1)

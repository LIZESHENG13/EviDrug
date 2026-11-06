"""Prediction ranking, differentiable DDI and training-only epoch feedback."""

import math

from torch.nn import functional as F

THRESHOLD, TEMPERATURE, DUAL_LR, DUAL_MAX = 0.3, 0.25, 20.0, 10.0


def pairwise_rank_loss(logits, labels):
    """Average soft pairwise loss over positive-negative medication pairs."""
    positive = labels.unsqueeze(-1)
    negative = (1.0 - labels).unsqueeze(-2)
    valid = positive * negative
    if not bool(valid.any()):
        return logits.new_zeros(())
    # Row i is an observed medication and column j an unobserved one.
    margin = logits.unsqueeze(-1) - logits.unsqueeze(-2)
    return (F.softplus(-margin) * valid).sum() / valid.sum().clamp_min(1.0)


def pair_totals(signal, graph):
    edges = ((signal @ graph) * signal).sum() / 2
    pairs = (signal.sum(-1).square() - signal.square().sum(-1)).sum() / 2
    return edges, pairs


def pooled_rate(signal, graph):
    edges, pairs = pair_totals(signal, graph)
    return edges / pairs.clamp_min(1.0)


def soft_ddi(logits, graph):
    center = math.log(THRESHOLD / (1.0 - THRESHOLD))
    decision = ((logits - center) / TEMPERATURE).sigmoid()
    return pooled_rate(decision, graph)


def next_weight(weight, train_hard_ddi, target):
    """Projected ascent on a detached TRAIN constraint violation."""
    if target <= 0 or weight < 0:
        raise ValueError("Positive target and nonnegative coefficient required")
    return min(DUAL_MAX, max(0.0, float(weight) + DUAL_LR * (float(train_hard_ddi) - target)))


def quality_ok(score, baseline):
    return (
        score["prauc"] >= baseline["prauc"] - 0.002
        and score["f1"] >= baseline["f1"] - 0.005
        and score["jaccard"] >= baseline["jaccard"] - 0.005
        and 0.9 * baseline["mean_predicted_drugs"]
        <= score["mean_predicted_drugs"]
        <= 1.1 * baseline["mean_predicted_drugs"]
    )


def selection_key(score, baseline, epoch):
    """Among quality-eligible epochs, prefer lower validation hard DDI."""
    return (not quality_ok(score, baseline), score["ddi"], -score["prauc"], epoch)

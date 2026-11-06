"""Shared clinical subgroup, prescription-action and paired-bootstrap metrics."""

import numpy as np

KEYS = ("prauc", "f1", "jaccard", "ddi")


def patient_terms(y, p, graph):
    """Exact tie-aware patient AP/F1/J plus per-patient DDI numerator/denominator."""
    y, p = np.asarray(y, np.float64), np.asarray(p, np.float64)
    actual = y.sum(1)
    order = np.argsort(-p, axis=1, kind="stable")
    ranked = np.take_along_axis(y, order, 1)
    scores = np.take_along_axis(p, order, 1)
    cumulative = ranked.cumsum(1)
    ends = np.concatenate((scores[:, :-1] != scores[:, 1:], np.ones((len(y), 1), bool)), 1)
    ap, previous = np.zeros(len(y)), np.zeros(len(y))
    for j in range(y.shape[1]):
        mask = ends[:, j]
        ap[mask] += (cumulative[mask, j] - previous[mask]) * cumulative[mask, j] / (j + 1)
        previous[mask] = cumulative[mask, j]
    ap /= np.maximum(actual, 1)
    return decision_terms(y, p >= 0.3, graph, ap)


def decision_terms(y, decision, graph, ap):
    """Reuse threshold-invariant patient AP while recomputing set metrics."""
    y = np.asarray(y, dtype=np.float64)
    actual, count = y.sum(1), decision.sum(1)
    tp = (y * decision).sum(1)
    d = decision.astype(np.float64)
    interactions = ((d @ graph) * d).sum(1) / 2
    possible = count * (count - 1) / 2
    return np.stack(
        (
            ap,
            2 * tp / np.maximum(actual + count, 1),
            tp / np.maximum(actual + count - tp, 1),
            interactions,
            possible,
        ),
        1,
    )


def summarize_terms(terms, predicted, actual):
    return {
        "patients": len(terms),
        "prauc": float(terms[:, 0].mean()),
        "f1": float(terms[:, 1].mean()),
        "jaccard": float(terms[:, 2].mean()),
        "ddi": float(terms[:, 3].sum() / max(terms[:, 4].sum(), 1)),
        "mean_predicted_drugs": float(np.mean(predicted, dtype=np.float64)),
        "mean_actual_drugs": float(np.mean(actual, dtype=np.float64)),
    }


def micro_scores(y, decisions, candidate):
    tp = float(np.sum((y > 0) & decisions & candidate))
    fp = float(np.sum((y == 0) & decisions & candidate))
    fn = float(np.sum((y > 0) & ~decisions & candidate))
    return {
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "support": int(tp + fn),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
    }


def paired_bootstrap(ours, reference, repetitions=2000, seed=20261103):
    assert ours.shape == reference.shape and ours.ndim == 3 and ours.shape[2] == 5
    n = ours.shape[1]
    rng = np.random.default_rng(seed)
    delta = []
    for start in range(0, repetitions, 64):
        weights = rng.multinomial(n, np.full(n, 1 / n), size=min(64, repetitions - start))
        estimates = []
        for data in (ours, reference):
            values = np.stack([weights @ term for term in data], 1)
            scores = np.concatenate(
                (
                    values[:, :, :3] / n,
                    (values[:, :, 3] / np.maximum(values[:, :, 4], 1))[:, :, None],
                ),
                2,
            )
            estimates.append(scores.mean(1))
        delta.append(estimates[0] - estimates[1])
    delta = np.concatenate(delta)

    def point(data):
        values = data.sum(1)
        return np.concatenate(
            (values[:, :3] / n, (values[:, 3] / np.maximum(values[:, 4], 1))[:, None]), 1
        ).mean(0)

    difference = point(ours) - point(reference)
    return {
        key: {
            "difference": float(difference[i]),
            "low": float(np.quantile(delta[:, i], 0.025)),
            "high": float(np.quantile(delta[:, i], 0.975)),
        }
        for i, key in enumerate(KEYS)
    }

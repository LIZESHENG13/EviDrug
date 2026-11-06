"""Sample-averaged AP/F1/Jaccard and pair-weighted interaction rate."""

import numpy as np


def metrics(labels, probabilities, ddi, threshold=0.3):
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = probabilities >= threshold
    positives = labels.sum(1)
    tp = (labels * predictions).sum(1)
    predicted = predictions.sum(1)
    union = positives + predicted - tp
    # Average precision uses tied-score groups, matching sklearn's definition.
    order = np.argsort(-probabilities, axis=1, kind="stable")
    ranked_labels = np.take_along_axis(labels, order, 1)
    ranked_scores = np.take_along_axis(probabilities, order, 1)
    cumulative = ranked_labels.cumsum(1)
    group_end = np.concatenate(
        [ranked_scores[:, :-1] != ranked_scores[:, 1:], np.ones((len(labels), 1), bool)], 1
    )
    ap = np.zeros(len(labels))
    previous = np.zeros(len(labels))
    for j in range(labels.shape[1]):
        end = group_end[:, j]
        ap[end] += (cumulative[end, j] - previous[end]) * cumulative[end, j] / (j + 1)
        previous[end] = cumulative[end, j]
    ap /= np.maximum(positives, 1)
    pred_float = predictions.astype(np.float64)
    pairs = (predicted * (predicted - 1) / 2).sum()
    interactions = ((pred_float @ ddi) * pred_float).sum() / 2
    return {
        "patients": len(labels),
        "prauc": float(ap.mean()),
        "f1": float((2 * tp / np.maximum(positives + predicted, 1)).mean()),
        "jaccard": float((tp / np.maximum(union, 1)).mean()),
        "ddi": float(interactions / max(pairs, 1)),
        "mean_predicted_drugs": float(predicted.mean()),
        "mean_actual_drugs": float(positives.mean()),
    }

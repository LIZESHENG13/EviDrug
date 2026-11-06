"""Clinical analysis uses the same metric definitions and candidate partitions."""

import numpy as np
import pytest

from evidrug.analysis import micro_scores, paired_bootstrap, patient_terms, summarize_terms
from evidrug.metrics import metrics


def test_patient_terms_and_bootstrap():
    y = np.array([[1, 0, 1], [0, 1, 0]], np.float64)
    p = np.array([[0.6, 0.6, 0.1], [0.2, 0.8, 0.9]], np.float64)
    graph = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    terms = patient_terms(y, p, graph)
    summarized = summarize_terms(terms, (p >= 0.3).sum(1), y.sum(1))
    for key, value in metrics(y, p, graph).items():
        assert summarized[key] == pytest.approx(value)
    paired = paired_bootstrap(terms[None], terms[None], repetitions=32)
    assert all(value == {"difference": 0.0, "low": 0.0, "high": 0.0} for value in paired.values())
    mask = np.array([[True, False, False], [False, True, False]])
    continued = micro_scores(y, p >= 0.3, mask)
    new = micro_scores(y, p >= 0.3, ~mask)
    assert continued["f1"] == 1.0
    assert new["true_positive"] == 0 and new["false_negative"] == 1

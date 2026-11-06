"""Model size, tensor flow and query-label independence."""

import json
from pathlib import Path

import pytest
import torch

from evidrug import EviDrug
from evidrug.config import ARMS
from evidrug.training import association_table, feature_batch, load_resources, loader_for


@pytest.mark.parametrize("arm", ARMS)
def test_scoring_does_not_read_query_labels(content, arm):
    records, _, codes, drugs, *_ = content
    pairs = [(12, 2), (13, 2)]
    memory = load_resources(content, pairs, arm, "cpu")
    batch = next(iter(loader_for(arm)(records, len(drugs), memory, pairs, 2)[0]))
    feature_batch(batch, association_table(content), arm, "cpu")
    model = EviDrug(len(codes), len(drugs), arm).eval()
    with torch.no_grad():
        original, evidence = model.distribution(batch)
        rows = torch.arange(len(batch["labels"]))
        batch["labels"][rows, batch["visit_mask"].sum(1) - 1] = float("nan")
        changed, _ = model.distribution(batch)
    assert torch.equal(original, changed)
    assert torch.isfinite(changed).all()
    assert changed.shape == (2, len(drugs))
    assert evidence["drug_features"].shape == (2, len(drugs), 32)
    if arm == "no_history":
        assert batch["visit_mask"].sum(1).tolist() == [1, 1]
    if arm == "no_memory":
        assert not hasattr(model.backbone, "query_projection")
        assert not batch["transition_mask"].any()
    if arm == "no_set":
        assert not hasattr(model.backbone, "set_encoder")
    if arm == "no_clinical":
        assert not hasattr(model.refinement, "clinical_projection")


@pytest.mark.parametrize(
    "dataset,expected",
    [
        ("mimic_iv_full", 7766432),
        ("nwicu", 2142330),
        ("sicdb", 1068728),
    ],
)
def test_published_parameter_counts(dataset, expected):
    root = Path(__file__).resolve().parents[1]
    cohort = json.loads((root / "configs/cohorts.json").read_text())[dataset]
    model = EviDrug(cohort["codes"], cohort["medications"])
    assert sum(p.numel() for p in model.parameters()) == expected
    assert sum(p.numel() for p in model.refinement.parameters()) == 112514
    assert not any("graph" in name for name, _ in model.named_parameters())
    assert sum(isinstance(module, torch.nn.TransformerEncoder) for module in model.modules()) == 1

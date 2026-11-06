"""Synthetic training, matched controls, test freezing and complete inference."""

import json
import pickle
from argparse import Namespace

import numpy as np
import pytest
import torch

from evidrug import training
from evidrug.config import ARMS
from evidrug.inference import Predictor


@pytest.mark.parametrize("arm", ARMS)
def test_three_stage_pipeline_and_freeze(tmp_path, content, arm):
    encoded = tmp_path / "synthetic.pkl"
    with encoded.open("wb") as stream:
        pickle.dump(content, stream)
    runs = tmp_path / "runs"
    output = runs / arm / "seed0"
    args = Namespace(
        encoded=str(encoded),
        data=None,
        resources="unused",
        device="cpu",
        dataset="synthetic",
        output=str(output),
        seed=0,
        arm=arm,
        continuation="dynamic",
        initial_refinement=None,
        epochs=1,
        batch_size=4,
    )
    training.train(args)
    selected = json.loads((output / "selected.json").read_text())
    assert selected["epochs_per_stage"] == 1
    if arm == "full":
        control = Namespace(**vars(args))
        control.output = str(runs / "continue_zero" / "seed0")
        control.continuation = "continue_zero"
        control.initial_refinement = str(output / "refinement/best.pt")
        training.train(control)
        history = json.loads((runs / "continue_zero/seed0/continuation/history.json").read_text())
        assert all(row["weight_used"] == 0 for row in history)
        assert not (runs / "continue_zero/seed0/backbone").exists()
    training.freeze(Namespace(runs=str(runs)))
    evaluate = Namespace(
        encoded=str(encoded),
        data=None,
        resources="unused",
        device="cpu",
        run=str(output),
        freeze=str(runs / "TEST_FREEZE.json"),
    )
    training.evaluate(evaluate)
    with np.load(output / "test_predictions.npz") as predictions:
        probabilities = predictions["probabilities"].copy()
    assert probabilities.shape == (2, 6)
    predictor = Predictor(content, output / selected["checkpoint"], arm)
    _, online = predictor.predict([(14, 2), (15, 2)], batch_size=2)
    np.testing.assert_array_equal(online, probabilities)
    assert torch.isfinite(torch.tensor(online)).all()
    selected["seed"] = 99
    (output / "selected.json").write_text(json.dumps(selected))
    with pytest.raises(AssertionError):
        training.evaluate(evaluate)

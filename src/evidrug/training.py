"""Three-stage training, validation selection, freezing and separate test evaluation."""

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .config import MODEL
from .config import TRAINING as CONFIG
from .data.batching import loader_for
from .data.records import load_split
from .evidence.association import ClinicalAssociation
from .evidence.retrieval import build_memory
from .metrics import metrics
from .models import ClinicalMemoryBackbone, EviDrug
from .objectives import next_weight, quality_ok, selection_key, soft_ddi
from .utils import device_batch, seed_all, sha, target, write

CONTROLS = CONFIG["fixed_coefficient_controls"]


def load_resources(content, queries, arm, device):
    records, splits = content[:2]
    if arm == "no_memory":
        return {
            "refs": [],
            "retrieved": {f"{p}:{t}": [] for p, t in queries},
            "scores": {f"{p}:{t}": [] for p, t in queries},
        }
    return build_memory(
        records,
        splits["train"],
        queries,
        device=device,
        state_only=arm == "no_history",
        neighbors=8,
    )


def freeze_backbone(model):
    model.backbone.requires_grad_(False)
    model.refinement.requires_grad_(True)
    return list(model.refinement.parameters())


def load_content(args):
    if args.encoded:
        # Author-generated/prepared local data; never included in the public artifact.
        with Path(args.encoded).open("rb") as stream:
            return pickle.load(stream)
    return load_split(args.data, resources=args.resources)


def association_table(content):
    return ClinicalAssociation(content[0], content[1]["train"], len(content[2]), len(content[3]))


def feature_batch(batch, association, arm, device):
    patients = batch["index"].tolist()
    visits = batch["target_visit"].tolist()
    features = [
        association.features(
            p,
            t,
            beta=MODEL["association_smoothing"],
            exclude_patient=p in association.train_indices,
        )
        for p, t in zip(patients, visits)
    ]
    batch["association_features"] = torch.from_numpy(np.stack(features)).to(device)
    return batch


def frozen_eval(module):
    if not any(parameter.requires_grad for parameter in module.parameters()):
        module.eval()
    else:
        for child in module.children():
            frozen_eval(child)


def parameter_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


@torch.no_grad()
def predict(model, loader, association, arm, device, refinement):
    model.eval()
    labels, probability, indices = [], [], []
    for raw in loader:
        batch = device_batch(raw, device)
        if refinement:
            feature_batch(batch, association, arm, device)
        logits, _ = model(batch)
        labels.append(target(batch).cpu().numpy())
        probability.append(logits.sigmoid().cpu().numpy())
        indices.extend(raw["index"].tolist())
    return np.concatenate(labels), np.concatenate(probability), np.asarray(indices)


def stage(
    model,
    parameters,
    optimizer,
    train_loader,
    sampler,
    validation,
    association,
    arm,
    device,
    folder,
    epochs,
    ddi,
    baseline=None,
    monitor=None,
    control="dynamic",
):
    refinement = association is not None
    continuation = baseline is not None
    coefficient = 1.0 if control == "dynamic" else CONTROLS[control]
    graph = torch.as_tensor(ddi, dtype=torch.float32, device=device)
    threshold = CONFIG["recommendation_threshold"]
    target_rate = None
    if continuation:
        y, probability, _ = predict(model, monitor, association, arm, device, True)
        target_rate = 0.9 * metrics(y, probability, ddi, threshold)["ddi"]
        if control == "dynamic" and target_rate <= 0:
            coefficient = 0.0
    best_key, best_state, history = None, None, []
    selected_epoch, selected_score = None, None
    folder.mkdir(parents=True, exist_ok=True)

    def validate(epoch, used=None, feedback=None, loss=None):
        nonlocal best_key, best_state, selected_epoch, selected_score
        y, p, _ = predict(model, validation, association, arm, device, refinement)
        score = metrics(y, p, ddi, threshold)
        key = selection_key(score, baseline, epoch) if continuation else (-score["prauc"], epoch)
        if best_key is None or key < best_key:
            best_key, best_state = key, parameter_state(model)
            selected_epoch, selected_score = epoch, score
        row = {"epoch": epoch, "validation": score, "loss": loss}
        if continuation:
            row.update(
                weight_used=used,
                weight_next=coefficient,
                training_hard_ddi=feedback,
                quality_eligible=quality_ok(score, baseline),
            )
        history.append(row)
        write(folder / "history.json", history)

    if continuation:
        validate(0, coefficient, target_rate / 0.9 if target_rate else 0.0)
    for epoch in range(1, epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        if refinement:
            frozen_eval(model)
        total, count = 0.0, 0
        used = coefficient
        for raw in train_loader:
            batch = device_batch(raw, device)
            if refinement:
                feature_batch(batch, association, arm, device)
            optimizer.zero_grad(set_to_none=True)
            logits, extra = model(batch)
            y = target(batch)
            loss = F.binary_cross_entropy_with_logits(logits, y) + extra
            if continuation and used:
                loss = loss + used * (soft_ddi(logits, graph) - target_rate)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(y)
            count += len(y)
        assert count == len(sampler.dataset.by_patient)
        feedback = None
        if continuation:
            y, probability, _ = predict(model, monitor, association, arm, device, True)
            feedback = metrics(y, probability, ddi, threshold)["ddi"]
            if control == "dynamic":
                coefficient = (
                    next_weight(coefficient, feedback, target_rate) if target_rate > 0 else 0.0
                )
        validate(epoch, used, feedback, total / max(count, 1))
        print(
            f"{folder.name}: epoch {epoch}/{epochs}, AP={history[-1]['validation']['prauc']:.4f}",
            flush=True,
        )
    model.load_state_dict(best_state, strict=True)
    model.eval()
    torch.save(
        {"state_dict": best_state, "epoch": selected_epoch, "validation": selected_score},
        folder / "best.pt",
    )
    summary = {
        "epoch": selected_epoch,
        "validation": selected_score,
        "checkpoint_sha256": sha(folder / "best.pt"),
        "epochs": epochs,
        "target_rate": target_rate,
    }
    write(folder / "selection.json", summary)
    return summary


def train(args):
    output = Path(args.output)
    if (output / "selected.json").exists():
        raise FileExistsError("Completed run already exists; use a new output directory.")
    if args.continuation != "dynamic" and not args.initial_refinement:
        raise ValueError("Matched controls require --initial-refinement from the paired main run.")
    output.mkdir(parents=True, exist_ok=True)
    content = load_content(args)
    records, splits, codes, drugs, ddi, ehr, metadata = content
    train_pairs = [
        (int(p), t) for p in splits["train"] for t in range(1, len(records[int(p)]["visits"]))
    ]
    validation_pairs = [(int(p), len(records[int(p)]["visits"]) - 1) for p in splits["validation"]]
    queries = train_pairs + validation_pairs
    memory = load_resources(content, queries, args.arm, args.device)
    association = association_table(content)
    loader = loader_for(args.arm)
    training, sampler = loader(
        records, len(drugs), memory, train_pairs, args.batch_size, True, args.seed
    )
    validation, _ = loader(records, len(drugs), memory, validation_pairs, args.batch_size)
    sampler.set_epoch(1)
    monitoring = [train_pairs[i] for i in sorted(i for group in sampler for i in group)]
    monitor, _ = loader(records, len(drugs), memory, monitoring, args.batch_size)
    backbone_arm = args.arm if args.arm in ("no_history", "no_memory", "no_set") else "full"
    if args.initial_refinement:
        if args.arm != "full":
            raise ValueError("Matched fixed-weight controls use --arm full.")
        saved = torch.load(args.initial_refinement, map_location="cpu", weights_only=True)
        seed_all(args.seed + 4100)
        model = EviDrug(len(codes), len(drugs), "full").to(args.device)
        model.load_state_dict(saved["state_dict"], strict=True)
        parameters = freeze_backbone(model)
        initial = {"validation": saved["validation"]}
    else:
        seed_all(args.seed)
        backbone = ClinicalMemoryBackbone(len(codes), len(drugs), backbone_arm).to(args.device)
        parameters = list(backbone.parameters())
        encoder = list(backbone.code_encoder.parameters())
        ids = {id(p) for p in encoder}
        optimizer = torch.optim.AdamW(
            [
                {"params": encoder, "lr": 5e-5},
                {"params": [p for p in parameters if id(p) not in ids], "lr": 7e-4},
            ],
            weight_decay=0.001,
        )
        stage(
            backbone,
            parameters,
            optimizer,
            training,
            sampler,
            validation,
            None,
            args.arm,
            args.device,
            output / "backbone",
            args.epochs,
            ddi,
        )
        backbone_state = parameter_state(backbone)
        del backbone, optimizer
        seed_all(args.seed + 4100)
        model = EviDrug(len(codes), len(drugs), args.arm).to(args.device)
        model.backbone.load_state_dict(backbone_state, strict=True)
        parameters = freeze_backbone(model)
        optimizer = torch.optim.AdamW(parameters, lr=0.0005, weight_decay=0.001)
        initial = stage(
            model,
            parameters,
            optimizer,
            training,
            sampler,
            validation,
            association,
            args.arm,
            args.device,
            output / "refinement",
            args.epochs,
            ddi,
        )
        del optimizer
    seed_all(args.seed + 7100)
    optimizer = torch.optim.AdamW(parameters, lr=0.0005, weight_decay=0.001)
    continuation = stage(
        model,
        parameters,
        optimizer,
        training,
        sampler,
        validation,
        association,
        args.arm,
        args.device,
        output / "continuation",
        args.epochs,
        ddi,
        initial["validation"],
        monitor,
        args.continuation,
    )
    write(
        output / "selected.json",
        {
            "model_version": "clinical_memory_dynamic_20261104",
            "dataset": args.dataset,
            "seed": args.seed,
            "arm": args.arm,
            "continuation": args.continuation,
            "split_sha256": metadata["split_sha256"],
            "parameters": sum(p.numel() for p in model.parameters()),
            "head_parameters": sum(p.numel() for p in parameters),
            "checkpoint": "continuation/best.pt",
            "checkpoint_sha256": continuation["checkpoint_sha256"],
            "validation": continuation["validation"],
            "epochs_per_stage": args.epochs,
            "release_execution": "Online evidence; cleaned package with verified checkpoint migration.",
        },
    )


def freeze(args):
    root = Path(args.runs)
    if (root / "TEST_FREEZE.json").exists():
        raise FileExistsError("Run group already frozen; use a new run root for a new experiment.")
    selected = sorted(root.rglob("selected.json"))
    if not selected:
        raise FileNotFoundError("No completed runs found")
    rows = {}
    for path in selected:
        meta = json.loads(path.read_text())
        checkpoint = path.parent / meta["checkpoint"]
        assert sha(checkpoint) == meta["checkpoint_sha256"]
        rows[str(path.relative_to(root))] = {
            "selection_sha256": sha(path),
            "checkpoint_sha256": sha(checkpoint),
        }
    write(root / "TEST_FREEZE.json", {"runs": rows, "declared_runs": len(rows)})
    print(f"Frozen {len(rows)} completed runs; test evaluation is now a separate command.")


def evaluate(args):
    run = Path(args.run).resolve()
    frozen_path = Path(args.freeze).resolve()
    frozen = json.loads(frozen_path.read_text())
    relative = str((run / "selected.json").relative_to(frozen_path.parent))
    assert frozen["runs"][relative]["selection_sha256"] == sha(run / "selected.json")
    selected = json.loads((run / "selected.json").read_text())
    checkpoint = run / selected["checkpoint"]
    assert selected["checkpoint_sha256"] == sha(checkpoint)
    content = load_content(args)
    assert selected["split_sha256"] == content[-1]["split_sha256"]
    records, splits, codes, drugs, ddi, ehr, metadata = content
    arm = selected["arm"]
    queries = [(int(p), len(records[int(p)]["visits"]) - 1) for p in splits["test"]]
    memory = load_resources(content, queries, arm, args.device)
    loader, _ = loader_for(arm)(records, len(drugs), memory, queries, 16)
    association = association_table(content)
    model = EviDrug(len(codes), len(drugs), arm).to(args.device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=args.device, weights_only=True)["state_dict"],
        strict=True,
    )
    y, p, indices = predict(model, loader, association, arm, args.device, True)
    np.savez_compressed(run / "test_predictions.npz", labels=y, probabilities=p, indices=indices)
    write(
        run / "test_metrics.json",
        {
            **metrics(y, p, ddi, 0.3),
            "checkpoint_sha256": sha(checkpoint),
            "predictions_sha256": sha(run / "test_predictions.npz"),
            "freeze_sha256": sha(frozen_path),
        },
    )
    print(json.dumps(metrics(y, p, ddi, 0.3), indent=2))

"""Measure complete batched inference, including online retrieval and associations."""

import json
import statistics
import time
from pathlib import Path

import torch

from .inference import Predictor
from .training import load_content
from .utils import sha, write


def benchmark(args):
    torch.set_num_threads(args.threads)
    content = load_content(args)
    run = Path(args.run)
    selected = json.loads((run / "selected.json").read_text())
    checkpoint = run / selected["checkpoint"]
    if sha(checkpoint) != selected["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from the run selection.")
    predictor = Predictor(content, checkpoint, arm=selected["arm"], device=args.device)
    records, splits = content[:2]
    queries = [
        (int(p), len(records[int(p)]["visits"]) - 1) for p in splits["validation"][: args.patients]
    ]
    predictor.predict(queries, args.batch_size)
    durations = []
    for _ in range(args.passes):
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(args.device)
        start = time.perf_counter()
        predictor.predict(queries, args.batch_size)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(args.device)
        durations.append(time.perf_counter() - start)
    report = {
        "parameters": sum(p.numel() for p in predictor.model.parameters()),
        "patients": len(queries),
        "batch_size": args.batch_size,
        "threads": args.threads,
        "device": args.device,
        "dtype": "float32",
        "seconds": durations,
        "ms_per_patient": 1000 * statistics.median(durations) / len(queries),
        "online_retrieval_and_associations_included": True,
        "query_cache": False,
        "fixed_training_index_built_before_timing": True,
        "checkpoint_sha256": sha(checkpoint),
    }
    write(args.output, report)
    print(json.dumps(report, indent=2))

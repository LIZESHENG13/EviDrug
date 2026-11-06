"""Patient-balanced prefix sampling and structured code serialization."""

from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from ..evidence.retrieval import query_key
from .records import collate

MAX_SEQUENCE_LENGTH, MAX_VISITS, MAX_SERIALIZATION = 512, 64, 128


class VisitQueryDataset(Dataset):
    def __init__(self, records, query_pairs):
        self.records = records
        self.queries = [(int(i), int(t)) for i, t in query_pairs]
        self.lengths = np.asarray(
            [
                sum(len(visit["codes"]) for visit in records[i]["visits"][: t + 1])
                for i, t in self.queries
            ],
            dtype=np.int64,
        )
        self.by_patient = {}
        for sample, (patient, _) in enumerate(self.queries):
            self.by_patient.setdefault(patient, []).append(sample)

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        return self.queries[index]


class PatientVisitSampler(Sampler):
    """One target per patient per epoch, cycling through that patient's visits."""

    def __init__(self, dataset, batch_size, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 1
        self.cycles = {}
        for patient, samples in dataset.by_patient.items():
            rng = np.random.default_rng(seed + 104729 * (patient + 1))
            self.cycles[patient] = rng.permutation(samples).tolist()

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return (len(self.cycles) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        selected = [cycle[(self.epoch - 1) % len(cycle)] for cycle in self.cycles.values()]
        rng = np.random.default_rng(self.seed + self.epoch)
        selected = np.asarray(selected, dtype=np.int64)
        selected = selected[rng.permutation(len(selected))]
        batches = []
        pool_size = self.batch_size * 20
        for start in range(0, len(selected), pool_size):
            pool = selected[start : start + pool_size]
            pool = pool[np.argsort(-self.dataset.lengths[pool], kind="stable")]
            batches.extend(
                [
                    pool[j : j + self.batch_size].tolist()
                    for j in range(0, len(pool), self.batch_size)
                ]
            )
        yield from (batches[j] for j in rng.permutation(len(batches)))


def visit_collate(items, records, medications, memory):
    queries = [(int(patient), int(step)) for patient, step in items]
    prefixes = []
    for patient, step in queries:
        record = records[patient]
        visits = record["visits"][: step + 1]
        prefixes.append(
            {
                "patient_id": record["patient_id"],
                "visits": visits,
                "n_codes": sum(len(visit["codes"]) for visit in visits),
            }
        )
    base_items = [(row, prefix, []) for row, prefix in enumerate(prefixes)]
    batch = collate(base_items, records, medications)
    batch["index"] = torch.as_tensor([patient for patient, _ in queries], dtype=torch.long)
    batch["target_visit"] = torch.as_tensor([step for _, step in queries], dtype=torch.long)

    selected = [memory["retrieved"][query_key(patient, step)] for patient, step in queries]
    k = max(1, max(map(len, selected)))
    refs = memory["refs"]
    width = max(
        [1]
        + [
            len(records[refs[j][0]]["visits"][refs[j][1] - offset]["codes"])
            for ids in selected
            for j in ids
            for offset in (0, 1)
        ]
    )
    before = np.zeros((len(items), k, width), np.int64)
    after = np.zeros_like(before)
    y0 = np.zeros((len(items), k, medications), np.float32)
    y1 = np.zeros_like(y0)
    mask = np.zeros((len(items), k), bool)
    similarity = np.zeros((len(items), k, 2), np.float32)
    for row, ids in enumerate(selected):
        patient, step = queries[row]
        key = query_key(patient, step)
        for col, j in enumerate(ids):
            owner, transition = refs[j]
            previous, current = records[owner]["visits"][transition - 1 : transition + 1]
            before[row, col, : len(previous["codes"])] = previous["codes"]
            after[row, col, : len(current["codes"])] = current["codes"]
            y0[row, col, previous["drugs"]] = 1
            y1[row, col, current["drugs"]] = 1
            mask[row, col] = True
            similarity[row, col] = memory["scores"][key][col]

    values = [
        ("transition_before_codes", before),
        ("transition_after_codes", after),
        ("transition_before_labels", y0),
        ("transition_after_labels", y1),
        ("transition_mask", mask),
        ("transition_similarity", similarity),
    ]
    for key, value in values:
        batch[key] = torch.from_numpy(value)
    return batch


def structured_collate(items, records, medications, memory, max_length=MAX_SEQUENCE_LENGTH):
    batch = visit_collate(items, records, medications, memory)
    code_array = batch["codes"].numpy()
    visit_array = batch["visit_mask"].numpy()
    input_rows, visit_rows, serial_rows, mapping_rows = [], [], [], []
    for patient_codes, valid_visits in zip(code_array, visit_array):
        tokens, visit_ids, serial_ids, visit_map = [], [], [], []
        for visit_index, (code_ids, valid) in enumerate(zip(patient_codes, valid_visits)):
            if not valid:
                continue
            codes = code_ids[code_ids != 0].tolist()
            for order, token in enumerate(codes):
                tokens.append(token)
                visit_ids.append(min(visit_index, MAX_VISITS - 1))
                serial_ids.append(min(order, MAX_SERIALIZATION - 1))
                visit_map.append(visit_index)
        if len(tokens) > max_length:
            tokens = tokens[-max_length:]
            visit_ids = visit_ids[-max_length:]
            serial_ids = serial_ids[-max_length:]
            visit_map = visit_map[-max_length:]
        input_rows.append(tokens)
        visit_rows.append(visit_ids)
        serial_rows.append(serial_ids)
        mapping_rows.append(visit_map)

    length = max(1, max(map(len, input_rows)))
    batch_size = len(items)
    input_ids = torch.zeros((batch_size, length), dtype=torch.long)
    visit_ids = torch.zeros_like(input_ids)
    serial_ids = torch.zeros_like(input_ids)
    visit_map = torch.full_like(input_ids, -1)
    for row, (tokens, v_ids, s_ids, v_map) in enumerate(
        zip(input_rows, visit_rows, serial_rows, mapping_rows)
    ):
        n = len(tokens)
        if n:
            input_ids[row, :n] = torch.as_tensor(tokens, dtype=torch.long)
            visit_ids[row, :n] = torch.as_tensor(v_ids, dtype=torch.long)
            serial_ids[row, :n] = torch.as_tensor(s_ids, dtype=torch.long)
            visit_map[row, :n] = torch.as_tensor(v_map, dtype=torch.long)
    batch["ehr_input_ids"] = input_ids
    batch["ehr_visit_ids"] = visit_ids
    batch["ehr_serial_ids"] = serial_ids
    batch["ehr_visit_map"] = visit_map
    return batch


def current_only_collate(items, records, medications, memory):
    """Original target identity and donors; no query history enters the batch."""
    queries = [(int(p), int(t)) for p, t in items]
    current = [
        {
            "patient_id": records[p]["patient_id"],
            "visits": [records[p]["visits"][t]],
            "n_codes": len(records[p]["visits"][t]["codes"]),
        }
        for p, t in queries
    ]
    batch = collate(
        [(p, record, []) for (p, _), record in zip(queries, current)], records, medications
    )
    batch["target_visit"] = torch.tensor([t for _, t in queries], dtype=torch.long)
    # Recreate code sequence from the singleton current visit, before the structured encoder. Padding, visit/order/global positions start afresh.
    sequences = [r["visits"][0]["codes"][-512:] for r in current]
    length = max(1, max(map(len, sequences)))
    ids = torch.zeros((len(items), length), dtype=torch.long)
    mapping = torch.full_like(ids, -1)
    serial = torch.zeros_like(ids)
    for row, sequence in enumerate(sequences):
        n = len(sequence)
        ids[row, :n] = torch.tensor(sequence, dtype=torch.long)
        mapping[row, :n] = 0
        # Original singleton sequence truncation retains within-visit order.
        offset = max(0, len(current[row]["visits"][0]["codes"]) - 512)
        serial[row, :n] = torch.arange(offset, offset + n).clamp_max(127)
    batch.update(
        ehr_input_ids=ids,
        ehr_visit_ids=torch.zeros_like(ids),
        ehr_serial_ids=serial,
        ehr_visit_map=mapping,
    )

    selected = [memory["retrieved"][query_key(p, t)] for p, t in queries]
    refs = memory["refs"]
    k = max(1, max(map(len, selected)))
    width = max(
        [1]
        + [
            len(records[refs[j][0]]["visits"][refs[j][1] - offset]["codes"])
            for indices in selected
            for j in indices
            for offset in (0, 1)
        ]
    )
    before = np.zeros((len(items), k, width), np.int64)
    after = np.zeros_like(before)
    y0 = np.zeros((len(items), k, medications), np.float32)
    y1 = np.zeros_like(y0)
    mask = np.zeros((len(items), k), bool)
    similarity = np.zeros((len(items), k, 2), np.float32)
    for row, indices in enumerate(selected):
        key = query_key(*queries[row])
        for column, j in enumerate(indices):
            owner, step = refs[j]
            assert owner != queries[row][0]
            previous, target = records[owner]["visits"][step - 1 : step + 1]
            before[row, column, : len(previous["codes"])] = previous["codes"]
            after[row, column, : len(target["codes"])] = target["codes"]
            y0[row, column, previous["drugs"]] = 1
            y1[row, column, target["drugs"]] = 1
            mask[row, column] = True
            similarity[row, column] = memory["scores"][key][column]
    batch.update(
        transition_before_codes=torch.from_numpy(before),
        transition_after_codes=torch.from_numpy(after),
        transition_before_labels=torch.from_numpy(y0),
        transition_after_labels=torch.from_numpy(y1),
        transition_mask=torch.from_numpy(mask),
        transition_similarity=torch.from_numpy(similarity),
    )
    return batch


def make_loader(
    records, medications, memory, query_pairs, batch_size=64, training=False, seed=0, arm="full"
):
    dataset = VisitQueryDataset(records, query_pairs)
    function = current_only_collate if arm == "no_history" else structured_collate
    kwargs = {
        "collate_fn": partial(function, records=records, medications=medications, memory=memory),
        "num_workers": 0,
    }
    if training:
        sampler = PatientVisitSampler(dataset, batch_size, seed)
        return DataLoader(dataset, batch_sampler=sampler, **kwargs), sampler
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, **kwargs), None


def loader_for(arm):
    return partial(make_loader, arm=arm)

"""Fixed training-case index; exact clinical Jaccard retrieval without drug inputs."""

import numpy as np
import torch
from scipy import sparse


def query_key(patient, step):
    return f"{int(patient)}:{int(step)}"


class CaseMemoryIndex:
    """Build training references once, then retrieve each request online.

    ``state_only=True`` implements the history ablation. The full model
    interleaves state and clinical-change rankings. Score ties use ascending
    reference indices, and both original same-patient exclusions are retained.
    """

    def __init__(self, records, train_indices, device="cpu", state_only=False, seed=20261026):
        self.records, self.device, self.state_only, self.seed = records, device, state_only, seed
        self.train_indices = list(map(int, train_indices))
        self.refs = [
            (p, t) for p in self.train_indices for t in range(1, len(records[p]["visits"]))
        ]
        if not self.refs:
            raise ValueError("Case memory requires training patients with at least two visits.")
        self.vocabulary = {
            c: i
            for i, c in enumerate(
                sorted(
                    {
                        c
                        for p in self.train_indices
                        for visit in records[p]["visits"]
                        for c in visit["retrieval_codes"]
                    }
                )
            )
        }
        owners = np.asarray([p for p, _ in self.refs], np.int64)
        self.permutation = np.random.default_rng(seed).permutation(len(self.refs))
        self.owners = torch.from_numpy(owners).to(device)
        self.shuffled_owners = torch.from_numpy(owners[self.permutation]).to(device)
        self.reference_ids = torch.arange(len(self.refs), device=device, dtype=torch.int32)
        self.banks, self.sizes = [], []
        for kind in range(1 if state_only else 2):
            rows, columns, sizes = [], [], []
            for row, (p, t) in enumerate(self.refs):
                state, change, counts = self._sets(p, t)
                values = (state, change)[kind]
                rows.extend([row] * len(values))
                columns.extend(values)
                sizes.append(counts[kind])
            matrix = sparse.csr_matrix(
                (np.ones(len(columns), np.float32), (rows, columns)),
                shape=(len(self.refs), len(self.vocabulary) * (1 if kind == 0 else 4)),
            )
            bank = torch.sparse_csr_tensor(
                torch.from_numpy(matrix.indptr.astype(np.int32)).to(device),
                torch.from_numpy(matrix.indices.astype(np.int32)).to(device),
                torch.from_numpy(matrix.data).to(device),
                size=matrix.shape,
                device=device,
            )
            self.banks.append(bank)
            self.sizes.append(torch.tensor(sizes, dtype=torch.float32, device=device))

    def _sets(self, patient, step):
        after = self.records[int(patient)]["visits"][int(step)]["retrieval_codes"]
        before = (
            self.records[int(patient)]["visits"][int(step) - 1]["retrieval_codes"]
            if step
            else frozenset()
        )
        width = len(self.vocabulary)
        state = [self.vocabulary[c] for c in after if c in self.vocabulary]
        change = [
            self.vocabulary[c] + offset * width
            for offset, values in enumerate((before, after, after - before, before - after))
            for c in values
            if c in self.vocabulary
        ]
        return state, change, (len(after), len(before) + len(after) + len(after ^ before))

    @torch.no_grad()
    def retrieve(self, query_pairs, neighbors=8, batch_size=128, progress=False):
        retrieved, scores = {}, {}
        for start in range(0, len(query_pairs), batch_size):
            queries = query_pairs[start : start + batch_size]
            encoded = [self._sets(int(p), int(t)) for p, t in queries]
            owners = torch.tensor([int(p) for p, _ in queries], device=self.device)
            valid = (self.owners[None] != owners[:, None]) & (
                self.shuffled_owners[None] != owners[:, None]
            )
            if not bool((valid.sum(1) >= neighbors).all()):
                raise ValueError(f"Each query needs at least {neighbors} eligible training donors.")
            rankings, similarities = [], []
            for kind, bank in enumerate(self.banks):
                indicator = torch.zeros((bank.shape[1], len(queries)), device=self.device)
                for column, values in enumerate(encoded):
                    indicator[values[kind], column] = 1
                intersection = torch.sparse.mm(bank, indicator).T.contiguous()
                query_sizes = torch.tensor(
                    [v[2][kind] for v in encoded], dtype=torch.float32, device=self.device
                )
                similarity = intersection / (
                    query_sizes[:, None] + self.sizes[kind][None] - intersection
                ).clamp_min(1)
                similarity.masked_fill_(~valid, -1)
                cutoff = torch.topk(similarity, neighbors, dim=1).values[:, -1:]
                tie_keys = torch.where(
                    similarity > cutoff,
                    -1,
                    torch.where(similarity == cutoff, self.reference_ids[None], len(self.refs)),
                )
                chosen = torch.topk(tie_keys, neighbors, dim=1, largest=False).indices.cpu().numpy()
                values = (
                    similarity.gather(1, torch.from_numpy(chosen).to(self.device)).cpu().numpy()
                )
                rankings.append(
                    [ids[np.lexsort((ids, -value))].tolist() for ids, value in zip(chosen, values)]
                )
                similarities.append(similarity)
            for row, (patient, step) in enumerate(queries):
                selected = []
                if self.state_only:
                    selected = rankings[0][row]
                else:
                    for state_rank, change_rank in zip(rankings[0][row], rankings[1][row]):
                        for candidate in (state_rank, change_rank):
                            if candidate not in selected and len(selected) < neighbors:
                                selected.append(candidate)
                        if len(selected) == neighbors:
                            break
                key = query_key(patient, step)
                retrieved[key] = selected
                if self.state_only:
                    values = similarities[0][row, selected].cpu().numpy()
                    scores[key] = np.stack([values, np.zeros_like(values)], 1).tolist()
                else:
                    scores[key] = np.stack(
                        [s[row, selected].cpu().numpy() for s in similarities], 1
                    ).tolist()
            if progress:
                print(
                    f"Retrieved {min(start + batch_size, len(query_pairs))}/{len(query_pairs)} queries",
                    flush=True,
                )
        return {
            "refs": self.refs,
            "permutation": self.permutation.tolist(),
            "retrieved": retrieved,
            "scores": scores,
            "train_indices": self.train_indices,
            "query_pairs": [[int(p), int(t)] for p, t in query_pairs],
            "neighbors": neighbors,
            "permutation_seed": self.seed,
            "retrieval_uses_medications": False,
            "same_patient_excluded": True,
        }


def build_memory(
    records,
    train_indices,
    query_pairs,
    device="cpu",
    state_only=False,
    neighbors=8,
    seed=20261026,
    progress=False,
):
    index = CaseMemoryIndex(records, train_indices, device, state_only, seed)
    return index.retrieve(query_pairs, neighbors=neighbors, progress=progress)

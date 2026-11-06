"""CPU inverted case index and immutable training association table for serving."""

import numpy as np
from scipy import sparse


def clinical_sets(record, step):
    after = set(record["visits"][step]["retrieval_codes"])
    before = set(record["visits"][step - 1]["retrieval_codes"]) if step else set()
    transition = {("before", c) for c in before} | {("after", c) for c in after}
    transition |= {("added", c) for c in after - before}
    transition |= {("removed", c) for c in before - after}
    return after, transition


def _matrix(sets, vocabulary):
    rows, cols = [], []
    for i, values in enumerate(sets):
        ids = [vocabulary[v] for v in values if v in vocabulary]
        rows.extend([i] * len(ids))
        cols.extend(ids)
    return sparse.csr_matrix(
        (np.ones(len(rows), np.float32), (rows, cols)), shape=(len(sets), len(vocabulary))
    )


def _top_references(scores, valid, count):
    candidates = np.flatnonzero(valid)
    if not len(candidates):
        return np.empty(0, np.int64)
    count = min(count, len(candidates))
    values = scores[candidates]
    if len(candidates) > count:
        cutoff = np.partition(values, len(values) - count)[-count]
        better = candidates[values > cutoff]
        tied = candidates[values == cutoff][: count - len(better)]
        candidates = np.concatenate((better, tied))
    return candidates[np.lexsort((candidates, -scores[candidates]))][:count]


class InferenceCaseIndex:
    """Resident inverted index; original float32 Jaccard and reference tie rule."""

    def __init__(self, records, training_patients, neighbors=8, seed=20261026):
        self.records = records
        training_patients = list(map(int, training_patients))
        self.refs = [(p, t) for p in training_patients for t in range(1, len(records[p]["visits"]))]
        self.neighbors = neighbors
        self.owners = np.asarray([p for p, _ in self.refs], np.int64)
        self.permutation = np.random.default_rng(seed).permutation(len(self.refs))
        self.shuffled_owners = self.owners[self.permutation]
        self.tables = []
        # Build one modality at a time to bound startup temporary memory.
        for kind in range(2):
            bank = [clinical_sets(records[p], t)[kind] for p, t in self.refs]
            vocabulary = {value: i for i, value in enumerate(sorted(set().union(*bank)))}
            matrix = _matrix(bank, vocabulary).T.tocsr()
            sizes = np.asarray([len(value) for value in bank], np.float32)
            self.tables.append((vocabulary, matrix, sizes))

    def query(self, patient, step):
        query = clinical_sets(self.records[int(patient)], int(step))
        eligible = (self.owners != int(patient)) & (self.shuffled_owners != int(patient))
        rankings, similarities = [], []
        for kind, (vocabulary, bank, sizes) in enumerate(self.tables):
            hit = (_matrix([query[kind]], vocabulary) @ bank).tocoo()
            scores = np.zeros(len(self.refs), np.float32)
            scores[hit.col] = hit.data / np.maximum(len(query[kind]) + sizes[hit.col] - hit.data, 1)
            rankings.append(_top_references(scores, eligible, self.neighbors))
            similarities.append(scores)
        selected = []
        for pair in zip(*rankings):
            for reference in pair:
                if int(reference) not in selected and len(selected) < self.neighbors:
                    selected.append(int(reference))
            if len(selected) == self.neighbors:
                break
        scores = np.asarray(
            [[float(values[j]) for values in similarities] for j in selected], np.float32
        )
        return selected, scores.reshape(len(selected), 2)

    @property
    def bytes(self):
        size = self.owners.nbytes + self.permutation.nbytes + self.shuffled_owners.nbytes
        for _, matrix, sizes in self.tables:
            size += sum(a.nbytes for a in (matrix.data, matrix.indices, matrix.indptr, sizes))
        return size


class ClinicalFeatureTable:
    """Training-only float64 log ratios; identical normalization to training."""

    def __init__(self, records, training_patients, vocabulary, medications):
        visits = [visit for p in training_patients for visit in records[int(p)]["visits"]]
        cr, cc, dr, dc = [], [], [], []
        for row, visit in enumerate(visits):
            codes = sorted({int(c) for c in visit["codes"] if c >= 3})
            drugs = sorted(set(visit["drugs"]))
            cr.extend([row] * len(codes))
            cc.extend(codes)
            dr.extend([row] * len(drugs))
            dc.extend(drugs)
        x = sparse.csr_matrix((np.ones(len(cr)), (cr, cc)), shape=(len(visits), vocabulary))
        y = sparse.csr_matrix((np.ones(len(dr)), (dr, dc)), shape=(len(visits), medications))
        joint = (y.T @ x).toarray()
        drug_count = np.asarray(y.sum(0)).ravel()
        code_count = np.asarray(x.sum(0)).ravel()
        beta = 0.1
        positive = (joint + beta) / (drug_count[:, None] + 2 * beta)
        negative = (code_count[None] - joint + beta) / (
            len(visits) - drug_count[:, None] + 2 * beta
        )
        self.table = np.log(positive) - np.log(negative)
        self.table[:, :3] = 0

    def features(self, visit):
        values = []
        for name in ("codes", "diagnoses", "procedures"):
            codes = np.asarray(sorted({int(c) for c in visit.get(name, []) if c >= 3}), np.int64)
            summed = self.table[:, codes].sum(-1)
            centered = (summed - summed.mean()) / max(summed.std(ddof=1), 1e-3)
            values.extend((centered, summed / max(len(codes), 1)))
        return np.stack(values, axis=-1).astype(np.float32)

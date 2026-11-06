"""Train-patient code/medication association evidence, with patient exclusion."""

import numpy as np
import torch
from scipy import sparse


class ClinicalAssociation:
    """Visit-level sufficient statistics; query targets are never scoring inputs.

    Training features can exclude all visits of their query patient. Validation
    and deployment use the complete training statistics, without new outcomes.
    """

    def __init__(self, records, train_indices, vocabulary, medications):
        self.records = records
        self.train_indices = set(map(int, train_indices))
        self.vocabulary = vocabulary
        self.medications = medications
        self.patients = {}
        self.n = 0
        self.drug_count = np.zeros(medications, np.float64)
        self.code_count = np.zeros(vocabulary, np.float64)
        self.joint_count = np.zeros((medications, vocabulary), np.float64)
        for patient in sorted(self.train_indices):
            visits = records[patient]["visits"]
            rows_c, cols_c, rows_d, cols_d = [], [], [], []
            for step, visit in enumerate(visits):
                codes = sorted({int(c) for c in visit["codes"] if c >= 3})
                drugs = sorted(set(visit["drugs"]))
                rows_c.extend([step] * len(codes))
                cols_c.extend(codes)
                rows_d.extend([step] * len(drugs))
                cols_d.extend(drugs)
            x = sparse.csr_matrix(
                (np.ones(len(rows_c)), (rows_c, cols_c)), shape=(len(visits), vocabulary)
            )
            y = sparse.csr_matrix(
                (np.ones(len(rows_d)), (rows_d, cols_d)), shape=(len(visits), medications)
            )
            joint = (y.T @ x).tocsr()
            dc = np.asarray(y.sum(0)).ravel()
            cc = np.asarray(x.sum(0)).ravel()
            self.patients[patient] = (len(visits), dc, cc, joint)
            self.n += len(visits)
            self.drug_count += dc
            self.code_count += cc
            self.joint_count += joint.toarray()

    def likelihood_ratio(self, codes, beta=0.1, exclude_patient=None):
        codes = np.asarray(sorted({int(c) for c in codes if c >= 3}), np.int64)
        if len(codes) == 0:
            return np.zeros((self.medications, 0), np.float64)
        n, dc, cc = self.n, self.drug_count, self.code_count[codes]
        joint = self.joint_count[:, codes]
        if exclude_patient is not None:
            if int(exclude_patient) not in self.train_indices:
                raise ValueError("Only a training patient can be excluded")
            own_n, own_dc, own_cc, own_joint = self.patients[int(exclude_patient)]
            n, dc, cc = n - own_n, dc - own_dc, cc - own_cc[codes]
            joint = joint - own_joint[:, codes].toarray()
        positive = (joint + beta) / (dc[:, None] + 2 * beta)
        negative = (cc[None] - joint + beta) / (n - dc[:, None] + 2 * beta)
        return np.log(positive) - np.log(negative)

    def features(self, patient, step, beta=0.1, exclude_patient=False):
        visit = self.records[int(patient)]["visits"][int(step)]
        excluded = int(patient) if exclude_patient else None
        values = []
        for name in ("codes", "diagnoses", "procedures"):
            codes = visit.get(name, visit["codes"] if name == "codes" else [])
            lr = self.likelihood_ratio(codes, beta, excluded)
            summed = lr.sum(-1)
            centered = (summed - summed.mean()) / max(summed.std(ddof=1), 1e-3)
            mean = summed / max(lr.shape[-1], 1)
            values.extend([centered, mean])
        return np.stack(values, axis=-1).astype(np.float32)

    def deployment_table(self, beta=0.1):
        value = np.zeros((self.vocabulary, self.medications), np.float32)
        value[3:] = self.likelihood_ratio(range(3, self.vocabulary), beta).T
        return torch.from_numpy(value)


def association_features(tokens, table):
    """Normalized summed and mean log-ratio from a fixed train-only table."""
    valid = tokens >= 3
    score = (table[tokens] * valid[..., None]).sum(-2)
    centered = (score - score.mean(-1, keepdim=True)) / score.std(-1, keepdim=True).clamp_min(1e-3)
    mean = score / valid.sum(-1, keepdim=True).clamp_min(1)
    return torch.stack([centered, mean], dim=-1)

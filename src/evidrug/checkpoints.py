"""Explicit migration of trusted historical study checkpoints to the clean package."""

from collections import OrderedDict

HEAD_NAMES = {
    "ssp_local_projection": "local_projection",
    "ssp_score_projection": "score_projection",
    "ssp_context_projection": "context_projection",
    "ssp_fusion_gate": "fusion_gate",
    "ssp_fusion_norm": "fusion_norm",
    "ssp_garm": "identity_scale",
    "ssp_score": "score",
    "clinical_projection": "clinical_projection",
}


def migrate_study_state(state, backbone_only=False):
    """Rename retained parameters; no tensor values, shapes or equations change."""
    migrated = OrderedDict()
    for name, tensor in state.items():
        prefix, _, suffix = name.partition(".")
        if prefix in HEAD_NAMES:
            target = "refinement." + HEAD_NAMES[prefix] + ("." + suffix if suffix else "")
        else:
            target = name.replace("code_encoder.bert.embeddings.", "code_encoder.embeddings.")
            if not backbone_only:
                target = "backbone." + target
        if target in migrated:
            raise ValueError(f"Duplicate migrated parameter: {target}")
        migrated[target] = tensor
    return migrated

"""The structured encoder, clinical-memory backbone and residual evidence head."""

from .backbone import ClinicalMemoryBackbone
from .model import EviDrug

__all__ = ["ClinicalMemoryBackbone", "EviDrug"]

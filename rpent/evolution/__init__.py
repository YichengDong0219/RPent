"""Baseline-compatible skill evolution utilities.

The online agent remains the reviewed RPent baseline.  This package only adds
an immutable memory view, passive tracing, and offline patch admission.
"""

from rpent.evolution.admission import AdmissionDecision, decide_admission
from rpent.evolution.library import apply_patch, create_snapshot
from rpent.evolution.schemas import SkillOptimizationDecision, SkillPatch

__all__ = [
    "AdmissionDecision",
    "SkillPatch",
    "SkillOptimizationDecision",
    "apply_patch",
    "create_snapshot",
    "decide_admission",
]

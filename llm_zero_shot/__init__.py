"""Zero-shot LLM benchmark for UML-10, ECORE-10, and V12."""

from .metrics import FIVE_METRICS, multilabel_metrics

__all__ = ["FIVE_METRICS", "multilabel_metrics"]

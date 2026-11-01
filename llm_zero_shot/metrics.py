"""Five-metric reporting and graph-level bootstrap for multilabel predictions."""

from __future__ import annotations

from typing import Sequence

import numpy as np


FIVE_METRICS = (
    "sample_f1",
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "macro_f1",
)


def _divide(numerator: np.ndarray | float, denominator: np.ndarray | float) -> np.ndarray:
    numerator_array = np.asarray(numerator, dtype=np.float64)
    denominator_array = np.asarray(denominator, dtype=np.float64)
    return np.divide(
        numerator_array,
        denominator_array,
        out=np.zeros(np.broadcast_shapes(numerator_array.shape, denominator_array.shape)),
        where=denominator_array != 0,
    )


def validate_binary_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    if true.ndim != 2 or true.shape != pred.shape or true.shape[0] == 0:
        raise ValueError("Expected nonempty sample-by-label arrays with equal shapes")
    if not np.isin(true, (0, 1)).all() or not np.isin(pred, (0, 1)).all():
        raise ValueError("Gold labels and predictions must be binary")
    return true, pred


def multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return Sample-F1 and four pooled/label-averaged metrics.

    All zero denominators follow the project's existing ``zero_division=0`` rule.
    Sample-F1 is the same quantity called Example-F1 by ModelSet.
    """

    true, pred = validate_binary_arrays(y_true, y_pred)
    tp = ((true == 1) & (pred == 1)).astype(np.int64)
    fp = ((true == 0) & (pred == 1)).astype(np.int64)
    fn = ((true == 1) & (pred == 0)).astype(np.int64)

    sample_tp = tp.sum(axis=1)
    sample_f1 = _divide(2 * sample_tp, true.sum(axis=1) + pred.sum(axis=1)).mean()

    total_tp = tp.sum()
    total_fp = fp.sum()
    total_fn = fn.sum()
    micro_precision = _divide(total_tp, total_tp + total_fp).item()
    micro_recall = _divide(total_tp, total_tp + total_fn).item()
    micro_f1 = _divide(2 * total_tp, 2 * total_tp + total_fp + total_fn).item()

    label_tp = tp.sum(axis=0)
    label_fp = fp.sum(axis=0)
    label_fn = fn.sum(axis=0)
    macro_f1 = _divide(
        2 * label_tp, 2 * label_tp + label_fp + label_fn
    ).mean()

    return {
        "sample_f1": float(sample_f1),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
    }


def per_label_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, label_ids: Sequence[str]
) -> list[dict[str, float | int | str]]:
    true, pred = validate_binary_arrays(y_true, y_pred)
    if true.shape[1] != len(label_ids):
        raise ValueError("Label ID count does not match prediction width")
    tp = ((true == 1) & (pred == 1)).sum(axis=0)
    fp = ((true == 0) & (pred == 1)).sum(axis=0)
    fn = ((true == 1) & (pred == 0)).sum(axis=0)
    precision = _divide(tp, tp + fp)
    recall = _divide(tp, tp + fn)
    f1 = _divide(2 * tp, 2 * tp + fp + fn)
    return [
        {
            "label_id": str(label_id),
            "support": int(true[:, index].sum()),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
        }
        for index, label_id in enumerate(label_ids)
    ]


def bootstrap_mean_confidence_intervals(
    repeat_arrays: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    samples: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    """Graph-level bootstrap CIs for means across complete API repeats."""

    if not repeat_arrays:
        raise ValueError("At least one complete repeat is required")
    validated = [validate_binary_arrays(true, pred) for true, pred in repeat_arrays]
    shape = validated[0][0].shape
    if any(true.shape != shape for true, _ in validated):
        raise ValueError("All repeats must use the same paired graph-label matrix")

    rng = np.random.default_rng(seed)
    draw_indices = rng.integers(0, shape[0], size=(samples, shape[0]))
    weights = np.zeros((samples, shape[0]), dtype=np.int32)
    row_indices = np.repeat(np.arange(samples), shape[0])
    np.add.at(weights, (row_indices, draw_indices.ravel()), 1)

    values = {metric: [] for metric in FIVE_METRICS}
    for true, pred in validated:
        tp = ((true == 1) & (pred == 1)).astype(np.int64)
        fp = ((true == 0) & (pred == 1)).astype(np.int64)
        fn = ((true == 1) & (pred == 0)).astype(np.int64)

        sample_scores = _divide(
            2 * tp.sum(axis=1), true.sum(axis=1) + pred.sum(axis=1)
        )
        values["sample_f1"].append((weights @ sample_scores) / shape[0])

        tp_global = weights @ tp.sum(axis=1)
        fp_global = weights @ fp.sum(axis=1)
        fn_global = weights @ fn.sum(axis=1)
        values["micro_precision"].append(_divide(tp_global, tp_global + fp_global))
        values["micro_recall"].append(_divide(tp_global, tp_global + fn_global))
        values["micro_f1"].append(
            _divide(2 * tp_global, 2 * tp_global + fp_global + fn_global)
        )

        tp_labels = weights @ tp
        fp_labels = weights @ fp
        fn_labels = weights @ fn
        values["macro_f1"].append(
            _divide(
                2 * tp_labels,
                2 * tp_labels + fp_labels + fn_labels,
            ).mean(axis=1)
        )

    intervals: dict[str, tuple[float, float]] = {}
    for metric, repeat_values in values.items():
        averaged = np.mean(np.stack(repeat_values, axis=0), axis=0)
        low, high = np.percentile(averaged, [2.5, 97.5])
        intervals[metric] = (float(low), float(high))
    return intervals

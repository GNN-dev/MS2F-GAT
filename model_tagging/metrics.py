"""Multilabel metrics on one run; scores use the 0--1 scale."""
import numpy as np


def scores(truth, predicted):
    y, p = np.asarray(truth, dtype=bool), np.asarray(predicted, dtype=bool)
    if y.shape != p.shape or y.ndim != 2 or min(y.shape) == 0:
        raise ValueError('Expected equally shaped, nonempty [graphs, labels] arrays.')
    tp, fp, fn = y & p, ~y & p, y & ~p

    def f1(axis):
        a, b, c = tp.sum(axis), fp.sum(axis), fn.sum(axis)
        denominator = 2*a+b+c
        return np.divide(2*a, denominator, out=np.zeros_like(a, dtype=float), where=denominator != 0)

    a, b, c = int(tp.sum()), int(fp.sum()), int(fn.sum())
    return {'sample_f1': float(f1(1).mean()), 'micro_precision': a/(a+b) if a+b else 0.,
            'micro_recall': a/(a+c) if a+c else 0., 'micro_f1': 2*a/(2*a+b+c) if 2*a+b+c else 0.,
            'macro_f1': float(f1(0).mean())}

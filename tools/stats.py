"""Rank statistics shared by the report scripts and the manuscript generator.

Why this file exists. Several reported correlations range over quantities with
exact ties: knob gains that are identically zero on an arm the knob cannot move,
constructed control arms that all sit at the same value, per-class IoU deltas
that round to nothing. An earlier version of these scripts ranked with
`np.argsort(np.argsort(x))`, which assigns *ordinal* ranks and breaks ties by
whatever order the input happened to arrive in. That is not Spearman's rho, and
it is not reproducible: permuting the class order alone moved one reported
segmentation correlation across [+0.598, +0.710] without touching a single
measurement. Ties are averaged here, which is the textbook definition and what
`scipy.stats.spearmanr` computes, so the value depends on the data alone.

Kept dependency-free apart from numpy: the manuscript generator has to run on
numpy and matplotlib alone, without scipy or torch.
"""
import numpy as np


def average_ranks(a):
    """Ranks of `a` with tied values sharing their mean rank."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="stable")
    s = a[order]
    r = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and s[j + 1] == s[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return r


def spearman(x, y):
    """Spearman rank correlation with tied ranks averaged.

    Returns nan when either input is constant, rather than raising, so a
    degenerate cell reports as missing instead of aborting a whole table.
    """
    rx, ry = average_ranks(x), average_ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])

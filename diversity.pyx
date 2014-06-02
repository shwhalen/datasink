#cython: boundscheck = False
#cython: wraparound = False

"""
    datasink: A Pipeline for Large-Scale Heterogeneous Ensemble Learning
    Copyright (C) 2013 Sean Whalen

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see [http://www.gnu.org/licenses/].
"""

cimport numpy
import cython
import itertools
import numpy

ctypedef numpy.float64_t DOUBLE
ctypedef numpy.uint64_t INT

cpdef crosstab(numpy.ndarray[INT, ndim = 1] a, numpy.ndarray[INT, ndim = 1] b):
    cdef:
        INT tp = 0, tn = 0, fn = 0, fp = 0, i
    for i in range(a.shape[0]):
        if a[i] & b[i]:
            tp += 1
        elif a[i] > b[i]:
            fn += 1
        elif a[i] < b[i]:
            fp += 1
    tn = a.shape[0] - tp - fn - fp
    return tp, fn, fp, tn


cpdef DOUBLE q_score(numpy.ndarray[INT, ndim = 1] a, numpy.ndarray[INT, ndim = 1] b):
    cdef:
        DOUBLE tp, fn, fp, tn, score
    tp, fn, fp, tn = crosstab(a, b)
    if fp + fn == 0: # 0s on off-diagonal
        score = 1
    elif tn + tp == 0: # 0s on diagonal
        score = -1
    elif fp + tn == 0: # a is all 1s
        score = tp / (tp + fn)
    elif fn + tn == 0: # b is all 1s
        score = tp / (tp + fp)
    elif tp + fp == 0: # b is all 0s
        score = tn / (fn + tn)
    elif tp + fn == 0: # a is all 0s
        score = tn / (fp + tn)
    else:
        score = (tp * tn - fp * fn) / (tp * tn + fp * fn)
    assert -1 <= score <= 1
    return score


cpdef DOUBLE average_diversity_score(numpy.ndarray[DOUBLE, ndim = 2] _x, DOUBLE threshold = 0.5, pairwise_diversity_score = q_score):
    if _x.ndim == 1 or _x.shape[1] == 1:
        return 1.0
    x = (_x >= threshold).astype(numpy.uint64)
    pairwise_scores = [pairwise_diversity_score(x[:, a], x[:, b]) for a, b in itertools.combinations(range(x.shape[1]), 2)]
    return numpy.mean(pairwise_scores)

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

from numpy import argmax, argmin, argsort, corrcoef, mean, nanmax, sqrt, triu_indices_from, where
from pandas import DataFrame, concat, read_csv
from scipy.io.arff import loadarff
import sklearn.metrics

def argsortbest(x):
    return argsort(x) if greater_is_better else argsort(x)[::-1]


def average_pearson_score(x):
    if isinstance(x, DataFrame):
        x = x.values
    rho = corrcoef(x, rowvar = 0)
    return mean(abs(rho[triu_indices_from(rho, 1)]))


def get_best_performer(df, one_se = False):
    if not one_se:
        return df[df.score == best(df.score)].head(1)
    se = df.score.std() / sqrt(df.shape[0] - 1)
    if greater_is_better:
        return df[df.score >= best(df.score) - se].head(1)
    return df[df.score <= best(df.score) + se].head(1)


def confusion_matrix_fpr(labels, predictions, false_discovery_rate = 0.1):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, predictions)
    max_fpr_index = where(fpr >= false_discovery_rate)[0][0]
    print sklearn.metrics.confusion_matrix(labels, predictions > thresholds[max_fpr_index])


def fmax_score(labels, predictions, beta = 1.0, pos_label = 1):
    """
        Radivojac, P. et al. (2013). A Large-Scale Evaluation of Computational Protein Function Prediction. Nature Methods, 10(3), 221-227.
        Manning, C. D. et al. (2008). Evaluation in Information Retrieval. In Introduction to Information Retrieval. Cambridge University Press.
    """
    precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, predictions, pos_label)
    f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    return nanmax(f1)


def load_arff(filename):
    return DataFrame.from_records(loadarff(filename)[0])


def load_arff_headers(filename):
    dtypes = {}
    for line in open(filename):
        if line.startswith('@data'):
            break
        if line.startswith('@attribute'):
            _, name, dtype = line.split()
            if dtype.startswith('{'):
                dtype = dtype[1:-1]
            dtypes[name] = set(dtype.split(','))
    return dtypes


def load_properties(dirname):
    properties = [_.split('=') for _ in open(dirname + '/weka.properties').readlines()]
    d = {}
    for key, value in properties:
        d[key.strip()] = value.strip()
    return d


def read_fold(path, fold):
    train_df        = read_csv('%s/validation-%i.csv.gz' % (path, fold), index_col = [0, 1], compression = 'gzip')
    test_df         = read_csv('%s/predictions-%i.csv.gz' % (path, fold), index_col = [0, 1], compression = 'gzip')
    train_labels    = train_df.index.get_level_values('label').values
    test_labels     = test_df.index.get_level_values('label').values
    return train_df, train_labels, test_df, test_labels


def rmse_score(a, b):
    return sqrt(mean((a - b)**2))


def unbag(df, bag_count):
    cols = []
    bag_start_indices = range(0, df.shape[1], bag_count)
    names = [_.split('.')[0] for _ in df.columns.values[bag_start_indices]]
    for i in bag_start_indices:
        cols.append(df.ix[:, i:i+bag_count].mean(axis = 1))
    df = concat(cols, axis = 1)
    df.columns = names
    return df


diversity_score = average_pearson_score
score = sklearn.metrics.roc_auc_score
greater_is_better = True
best = max if greater_is_better else min
argbest = argmax if greater_is_better else argmin
fmax_scorer = sklearn.metrics.make_scorer(fmax_score, greater_is_better = True, needs_threshold = True)

#!/usr/bin/env python

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

from os import mkdir
from os.path import abspath, exists
from sys import argv

from numpy import array, column_stack
from numpy.random import choice, seed
from pandas import DataFrame, concat
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals.joblib import Parallel, delayed
from sklearn.linear_model import SGDClassifier
import common

def get_cluster_performance(labels, predictions, n_clusters, fold, seedval):
    return {'fold': fold, 'seed': seedval, 'score': common.score(labels, predictions), 'n_clusters': n_clusters}


def get_performance(df, ensemble, fold, seedval):
    labels          = df.index.get_level_values('label').values
    predictions     = df[ensemble].mean(axis = 1)
    return {'fold': fold, 'seed': seedval, 'score': common.score(labels, predictions), 'ensemble': ensemble[-1], 'ensemble_size': len(ensemble)}


def get_predictions(df, ensemble, fold, seedval):
    ids             = df.index.get_level_values('id')
    labels          = df.index.get_level_values('label')
    predictions     = df[ensemble].mean(axis = 1)
    diversity       = common.diversity_score(df[ensemble].values)
    return DataFrame({'fold': fold, 'seed': seedval, 'id': ids, 'label': labels, 'prediction': predictions, 'diversity': diversity, 'ensemble_size': len(ensemble)})


def select_candidate_greedy(train_df, train_labels, best_classifiers, ensemble, i):
    return best_classifiers.index.values[i]


def select_candidate_enhanced(train_df, train_labels, best_classifiers, ensemble, i):
    if len(ensemble) >= initial_ensemble_size:
        candidates = choice(best_classifiers.index.values, min(max_candidates, len(best_classifiers)), replace = False)
        candidate_scores = [common.score(train_labels, train_df[ensemble + [candidate]].mean(axis = 1)) for candidate in candidates]
        best_candidate = candidates[common.argbest(candidate_scores)]
    else:
        best_candidate = best_classifiers.index.values[i]
    return best_candidate


def select_candidate_drep(train_df, train_labels, best_classifiers, ensemble, i):
    if len(ensemble) >= initial_ensemble_size:
        candidates = choice(best_classifiers.index.values, min(max_candidates, len(best_classifiers)), replace = False)
        candidate_diversity_scores = [abs(common.diversity_score(train_df[ensemble + [candidate]].values)) for candidate in candidates]
        candidate_diversity_ranks = array(candidate_diversity_scores).argsort()
        diversity_candidates = candidates[candidate_diversity_ranks[:max_diversity_candidates]]
        candidate_accuracy_scores = [common.score(train_labels, train_df[ensemble + [candidate]].mean(axis = 1)) for candidate in diversity_candidates]
        best_candidate = candidates[common.argbest(candidate_accuracy_scores)]
    else:
        best_candidate = best_classifiers.index.values[i]
    return best_candidate


def select_candidate_sdi(train_df, train_labels, best_classifiers, ensemble, i):
    if len(ensemble) >= initial_ensemble_size:
        candidates = choice(best_classifiers.index.values, min(max_candidates, len(best_classifiers)), replace = False)
        candidate_diversity_scores = [1 - abs(common.diversity_score(train_df[ensemble + [candidate]].values)) for candidate in candidates] # 1 - kappa so larger = more diverse
        candidate_scores = [accuracy_weight * best_classifiers.ix[candidate] + (1 - accuracy_weight) * candidate_diversity_scores[candidate_i] for candidate_i, candidate in enumerate(candidates)]
        best_candidate = candidates[common.argbest(candidate_scores)]
    else:
        best_candidate = best_classifiers.index.values[i]
    return best_candidate


def stack_intra(n_clusters, distances, fit_df, fit_labels, predict_df):
    cluster_labels = MiniBatchKMeans(n_clusters).fit_predict(distances)
    cols = []
    for label in set(cluster_labels):
        mask = cluster_labels == label
        model = stacker.fit(fit_df.ix[:, mask], fit_labels)
        predictions = model.predict_proba(predict_df.ix[:, mask])[:, 1]
        cols.append(predictions)
    values = column_stack(cols)
    predictions = values.mean(axis = 1)
    return values, predictions


def stack_inter(n_clusters, distances, fit_df, fit_labels, predict_df):
    cluster_labels = MiniBatchKMeans(n_clusters).fit_predict(distances)
    cols = []
    for label in set(cluster_labels):
        mask = cluster_labels == label
        predictions = fit_df.ix[:, mask].mean(axis = 1)
        cols.append(predictions)
    model = stacker.fit(column_stack(cols), fit_labels)
    cols = []
    for label in set(cluster_labels):
        mask = cluster_labels == label
        predictions = predict_df.ix[:, mask].mean(axis = 1)
        cols.append(predictions)
    values = column_stack(cols)
    predictions = model.predict_proba(values)[:, 1]
    return values, predictions


def stacked_selection(fold):
    seed(seedval)
    train_df, train_labels, test_df, test_labels = common.read_fold(path, fold)
    train_distances = 1 - train_df.corr().abs()
    train_performance = []
    test_performance = []
    for n_clusters in range(1, max_clusters + 1):
        train_values, train_predictions = stack_function(n_clusters, train_distances, train_df, train_labels, predict_df = train_df)
        test_values, test_predictions = stack_function(n_clusters, train_distances, train_df, train_labels, predict_df = test_df)
        train_performance.append(get_cluster_performance(train_labels, train_predictions, n_clusters, fold, seedval))
        test_performance.append(get_cluster_performance(test_labels, test_predictions, n_clusters, fold, seedval))
    best_cluster_size = common.get_best_performer(DataFrame.from_records(train_performance)).n_clusters.values
    test_values, test_predictions = stack_function(best_cluster_size, train_distances, train_df, train_labels, predict_df = test_df)
    return DataFrame({'fold': fold, 'seed': seedval, 'id': test_df.index.get_level_values('id'), 'label': test_labels, 'prediction': test_predictions, 'diversity': common.diversity_score(test_values), 'metric': common.score.__name__}), DataFrame.from_records(test_performance)


def selection(fold):
    seed(seedval)
    train_df, train_labels, test_df, test_labels = common.read_fold(path, fold)
    best_classifiers = train_df.apply(lambda x: common.score(train_labels, x)).order(ascending = not common.greater_is_better)
    train_performance = []
    test_performance = []
    ensemble = []
    for i in range(min(max_ensemble_size, len(best_classifiers))):
        best_candidate = select_candidate(train_df, train_labels, best_classifiers, ensemble, i)
        ensemble.append(best_candidate)
        train_performance.append(get_performance(train_df, ensemble, fold, seedval))
        test_performance.append(get_performance(test_df, ensemble, fold, seedval))
    train_performance_df = DataFrame.from_records(train_performance)
    best_ensemble_size = common.get_best_performer(train_performance_df).ensemble_size.values
    best_ensemble = train_performance_df.ensemble[:best_ensemble_size + 1]
    return get_predictions(test_df, best_ensemble, fold, seedval), DataFrame.from_records(test_performance)


path = abspath(argv[1])
assert exists(path)
if not exists('%s/analysis' % path):
    mkdir('%s/analysis' % path)
method = argv[2]
assert method in ['greedy', 'enhanced', 'drep', 'sdi', 'inter', 'intra']
if method in ['inter', 'intra']:
    stack_function = eval('stack_' + method)
    method_function = stacked_selection
else:
    select_candidate = eval('select_candidate_' + method)
    method_function = selection
p = common.load_properties(path)
fold_count = int(p['foldCount'])
initial_ensemble_size = 2
max_ensemble_size = 50
max_candidates = 50
max_diversity_candidates = 5
accuracy_weight = 0.5
max_clusters = 20

# use shallow non-linear stacker by default
stacker = RandomForestClassifier(n_estimators = 200, max_depth = 2, bootstrap = False, random_state = 0)
if len(argv) > 3 and argv[3] == 'linear':
    stacker = SGDClassifier(loss = 'log', n_iter = 50, random_state = 0)

predictions_dfs = []
performance_dfs = []
seeds = [0] if method == 'greedy' else range(10)
for seedval in seeds:
    results = Parallel(n_jobs = -1, verbose = 1)(delayed(method_function)(fold) for fold in range(fold_count))
    for predictions_df, performance_df in results:
        predictions_dfs.append(predictions_df)
        performance_dfs.append(performance_df)
performance_df = concat(performance_dfs)
performance_df.to_csv('%s/analysis/selection-%s-%s-iterations.csv' % (path, method, common.score.__name__), index = False)
predictions_df = concat(predictions_dfs)
predictions_df['method'] = method
predictions_df['metric'] = common.score.__name__
predictions_df.to_csv('%s/analysis/selection-%s-%s.csv' % (path, method, common.score.__name__), index = False)
print '%.3f %i' % (predictions_df.groupby(['fold', 'seed']).apply(lambda x: common.score(x.label, x.prediction)).mean(), predictions_df.ensemble_size.mean())

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

from pandas import DataFrame, concat
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals.joblib import Parallel, delayed
from sklearn.linear_model import SGDClassifier
from nnls import NNLS
import common

def stacked_generalization(fold):
    train_df, train_labels, test_df, test_labels = common.read_fold(path, fold)
    if method == 'aggregate':
        train_df = common.unbag(train_df, bag_count)
        test_df = common.unbag(test_df, bag_count)
    test_predictions = stacker.fit(train_df, train_labels).predict_proba(test_df)[:, 1]
    return DataFrame({'fold': fold, 'id': test_df.index.get_level_values('id'), 'label': test_labels, 'prediction': test_predictions, 'diversity': common.diversity_score(test_df.values)})


path = abspath(argv[1])
assert exists(path)
if not exists('%s/analysis' % path):
    mkdir('%s/analysis' % path)
method = argv[2]
assert method in ['aggregate', 'standard']
p = common.load_properties(path)
fold_count = int(p['foldCount'])
bag_count = int(p['bagCount'])

# use non-negative least squares for regression
if 'predictClassValue' not in p:
    stacker = NNLS()
else:
    # use linear stacker if requested, else use shallow non-linear stacker
    if len(argv) > 3 and argv[3] == 'linear':
        stacker = SGDClassifier(loss = 'log', n_iter = 50, random_state = 0)
    else:
        stacker = RandomForestClassifier(n_estimators = 200, max_depth = 2, bootstrap = False, random_state = 0)

predictions_dfs = Parallel(n_jobs = -1, verbose = 1)(delayed(stacked_generalization)(fold) for fold in range(fold_count))
predictions_df = concat(predictions_dfs)
predictions_df['method'] = method
predictions_df.to_csv('%s/analysis/stacking-%s.csv' % (path, method), index = False)
print '%.3f' % predictions_df.groupby('fold').apply(lambda x: common.score(x.label, x.prediction)).mean()

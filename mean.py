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
from sklearn.externals.joblib import Parallel, delayed
import common

def mean_aggregation(fold):
    train_df, train_labels, test_df, test_labels = common.read_fold(path, fold)
    ids         = test_df.index.get_level_values('id')
    labels      = test_df.index.get_level_values('label')
    predictions = test_df.mean(axis = 1)
    diversity   = common.diversity_score(test_df.values)
    return DataFrame({'id': ids, 'label': labels, 'fold': fold, 'prediction': predictions, 'diversity': diversity})


path = abspath(argv[1])
assert exists(path)
if not exists('%s/analysis' % path):
    mkdir('%s/analysis' % path)
p = common.load_properties(path)
fold_count = int(p['foldCount'])

predictions = Parallel(n_jobs = -1, verbose = 0)(delayed(mean_aggregation)(fold) for fold in range(fold_count))
predictions_df = concat(predictions)
predictions_df['method']  = 'mean'
predictions_df.to_csv('%s/analysis/mean.csv' % path, index = False)
print '%.3f' % predictions_df.groupby('fold').apply(lambda x: common.score(x.label, x.prediction)).mean()

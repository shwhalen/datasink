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

from glob import glob
from os.path import abspath, exists
from sys import argv

from pandas import DataFrame, concat, read_csv
import common

from warnings import filterwarnings
filterwarnings('ignore', category = DeprecationWarning)

path = abspath(argv[1])
assert exists(path)

scores = []
for dirname in glob('%s/weka.classifiers.*' % path):
    filenames = glob('%s/predictions-*.csv.gz' % dirname)
    predictions_df = concat([read_csv(filename, index_col = [0, 1], skiprows = 1, compression = 'gzip') for filename in filenames])
    score = common.score(predictions_df.index.get_level_values('label').values, predictions_df.prediction)
    scores.append([dirname.split('/')[-1], score])
print DataFrame(scores, columns = ['classifier', 'score']).set_index('classifier').sort('score', ascending = not common.greater_is_better)

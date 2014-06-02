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

from itertools import product
from os import environ, system
from os.path import abspath, dirname, exists
from sys import argv

from common import load_arff_headers, load_properties
from sklearn.externals.joblib import Parallel, delayed

def classify(parameters):
    working_dir, project_path, classifier, fold, bag = parameters
    expected_filenames = ['%s/%s/predictions-%s-%02i.csv.gz' % (project_path, classifier.split()[0], fold, bag)] + ['%s/%s/validation-%s-%02i-%02i.csv.gz' % (project_path, classifier.split()[0], fold, nested_fold, bag) for nested_fold in nested_fold_values]
    if sum(map(exists, expected_filenames)) == len(expected_filenames):
        return
    cmd = 'groovy -cp %s %s/Pipeline.groovy %s %s %s %s' % (classpath, working_dir, project_path, fold, bag, classifier)
    if use_cluster:
        cmd = '%s \"%s\"' % (cluster_cmd, cmd)
    system(cmd)


# ensure project directory exists
project_path = abspath(argv[1])
assert exists(project_path)

# load and parse project properties
p = load_properties(project_path)
classifiers_fn = '%s/%s' % (project_path, p['classifiersFilename'])
input_fn = '%s/%s' % (project_path, p['inputFilename'])
assert exists(input_fn)

# generate cross validation values for leave-one-value-out or k-fold
assert ('foldAttribute' in p) or ('foldCount' in p)
if 'foldAttribute' in p:
    headers = load_arff_headers(input_fn)
    fold_values = headers[p['foldAttribute']]
else:
    fold_values = range(int(p['foldCount']))
nested_fold_values = range(int(p['nestedFoldCount']))
bag_count = int(p['bagCount'])
bag_values = range(bag_count) if bag_count > 1 else [0]

# ensure java's classpath is set
classpath = environ['CLASSPATH']

# command for cluster execution if enabled
use_cluster = False if 'useCluster' not in p else p['useCluster'] == 'true'
cluster_cmd = 'rc.py --cores 1 --walltime 06:00:00 --queue small --allocation acc_9'

# load classifiers from file, skip commented lines
classifiers = filter(lambda x: not x.startswith('#'), open(classifiers_fn).readlines())
classifiers = [_.strip() for _ in classifiers]

working_dir = dirname(abspath(argv[0]))
n_jobs = 1 if use_cluster else -1#3
all_parameters = list(product([working_dir], [project_path], classifiers, fold_values, bag_values))
Parallel(n_jobs = n_jobs, verbose = 50)(delayed(classify)(parameters) for parameters in all_parameters)

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

from numpy import dot, zeros
from scipy.optimize import nnls
from sklearn.base import BaseEstimator

class NNLS(BaseEstimator):
    def __init__(self):
        pass


    def fit(self, X, y):
        self.coef_, self.residuals_ = nnls(X, y)
        return self


    def predict(self, X):
        return dot(X, self.coef_)


    # hack so we don't have to change classification stacking code
    def predict_proba(self, X):
        predictions = zeros([X.shape[0], 2])
        predictions[:, 1] = self.predict(X)
        return predictions

# -*- coding: utf-8 -*-
#
# Copyright 2015-2022 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions to model the idle engine speed.
"""
import numpy as np
from sklearn.cluster import DBSCAN


# noinspection PyPep8Naming,PyMissingOrEmptyDocstring
class _IdleDetector(DBSCAN):
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean',
                 algorithm='auto', leaf_size=30, p=None):
        super(_IdleDetector, self).__init__(
            eps=eps, min_samples=min_samples, metric=metric,
            algorithm=algorithm, leaf_size=leaf_size, p=p
        )
        self.cluster_centers_ = None
        self.min, self.max = None, None

    def fit(self, X, y=None, sample_weight=None):
        super(_IdleDetector, self).fit(X, y=y, sample_weight=sample_weight)

        c, lb = self.components_, self.labels_[self.core_sample_indices_]
        self.cluster_centers_ = np.array(
            [np.mean(c[lb == i]) for i in range(lb.max() + 1)]
        )
        self.min, self.max = c.min(), c.max()
        return self

    def predict(self, X, set_outliers=True):
        import sklearn.metrics as sk_met
        y = sk_met.pairwise_distances_argmin(X, self.cluster_centers_[:, None])
        if set_outliers:
            y[((X > self.max) | (X < self.min))[:, 0]] = -1
        return y

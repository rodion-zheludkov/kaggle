# coding=utf-8

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.cross_validation import StratifiedKFold
from sklearn.utils.validation import assert_all_finite


class Stacking(BaseEstimator):

    def __init__(self, meta_estimator, estimators, cvK):
        self.estimators = estimators
        self.meta_estimator = meta_estimator
        self.cvK = cvK

    def _base_estimator_predict_proba(self, e, X):
        pred = e.predict_proba(X)
        assert_all_finite(pred)
        return pred

    def _make_meta(self, X_b):
        rows = []
        for j, e in enumerate(self.estimators):
            X = X_b[j]
            pred = self._base_estimator_predict_proba(e, X)
            rows.append(pred)

        return np.hstack(rows)

    def fit(self, Xs, y):
        X_meta = []  # meta-level features
        y_meta = []  # meta-level labels

        print 'Training and validating the base (level-0) estimator(s)...'
        print

        cv = StratifiedKFold(y, self.cvK, indices=True)
        for i, (a, b) in enumerate(cv):
            print 'Fold [%s]' % i
            y_a, y_b = y[a], y[b]  # training and validation labels

            rows = []
            for j, e in enumerate(self.estimators):
                X = Xs[j]

                X_a, X_b = X[a], X[b]  # training and validation features
                print '  Training base (level-0) estimator %d...' % j
                print '  X_a' + str(X_a.shape)
                print '  y_a' + str(y_a.shape)

                e.fit(X_a, y_a)
                print '  done.'

                pred = self._base_estimator_predict_proba(e, X_b)
                rows.append(pred)

            X_fold_meta = np.hstack(rows)
            print '  X_fold_meta ', X_fold_meta.shape

            X_meta.append(X_fold_meta)
            y_meta.append(y_b)

        print

        X_meta = np.vstack(X_meta)
        y_meta = np.hstack(y_meta)

        # Train meta estimator.
        print 'Training meta (level-1) estimator...'
        print '  X_meta' + str(X_meta.shape)
        print '  y_meta' + str(y_meta.shape)
        self.meta_estimator.fit(X_meta, y_meta)
        print 'done.'

        # Re-train base estimators on full data.
        # for j, e in enumerate(self.estimators):
        #     print 'Re-training base (level-0) estimator %d on full data...' % (j),
        #     e.fit(X, y)
        #     print 'done.'

    def predict(self, I):
        raise NotImplementedError('Use predict_proba')

    def predict_proba(self, Xs):
        rows = []
        for j, e in enumerate(self.estimators):
            X = Xs[j]
            if self.proba:
                # Predict label probabilities
                pred = self._base_estimator_predict_proba(e, X)
            else:
                # Predict label values
                pred = self._base_estimator_predict(e, X)

            rows.append(pred)

        X_meta = np.hstack(rows)

        return self.meta_estimator.predict_proba(X_meta)



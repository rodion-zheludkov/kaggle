# coding=utf-8

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.cross_validation import StratifiedKFold
from sklearn.utils.validation import assert_all_finite


class Stacking(BaseEstimator):
    """
    Implements stacking/blending.

    David H. Wolpert (1992). Stacked generalization. Neural Networks,
    5:241-259, Pergamon Press.

    Parameters
    ----------
    meta_estimator_cls : string or callable
        May be one of "best", "vote", "average", or any
        classifier or regressor constructor

    estimators : iterator
        An iterable of estimators; each must support predict_proba()

    cv : iterator
        A cross validation object. Base (level-0) estimators are
        trained on the training folds, then the meta (level-1) estimator
        is trained on the testing folds.

    stackingc : bool
        Whether to use StackingC or not. For more information, refer to
        the following paper:
          Seewald A.K.: How to Make Stacking Better and Faster While
          Also Taking Care of an Unknown Weakness, in Sammut C.,
          Hoffmann A. (eds.), Proceedings of the Nineteenth
          International Conference on Machine Learning (ICML 2002),
          Morgan Kaufmann Publishers, pp.554-561, 2002.

    kwargs :
        Arguments passed to instantiate meta_estimator.
    """

    def __init__(self, meta_estimator, estimators, cvK, proba=True, **kwargs):
        self.estimators = estimators
        self.meta_estimator = meta_estimator
        self.cvK = cvK
        self.proba = proba

    def _base_estimator_predict(self, e, X):
        """ Predict label values with the specified estimator on
        predictor(s) X.

        Parameters
        ----------
        e : int
            The estimator object.

        X : np.ndarray, shape=(n, m)
            The feature data for which to compute the predicted outputs.

        Returns
        -------
        pred : np.ndarray, shape=(len(X), 1)
            The mean of the label probabilities predicted by the
            specified estimator for each fold for each instance X.
        """
        # Generate array for the base-level testing set, which is n x n_folds.
        pred = e.predict(X)
        assert_all_finite(pred)
        return pred

    def _base_estimator_predict_proba(self, e, X):
        """ Predict label probabilities with the specified estimator
        on predictor(s) X.

        Parameters
        ----------
        e : int
            The estimator object.

        X : np.ndarray, shape=(n, m)
            The feature data for which to compute the predicted outputs.

        Returns
        -------
        pred : np.ndarray, shape=(len(X), 1)
            The mean of the label probabilities predicted by the
            specified estimator for each fold for each instance X.
        """
        # Generate array for the base-level testing set, which is n x n_folds.
        pred = e.predict_proba(X)
        assert_all_finite(pred)
        return pred

    def _make_meta(self, X_b):
        """ Make the feature set for the meta (level-1) estimator.

        Parameters
        ----------
        X : np.ndarray, shape=(n, m)
            The feature data.

        Returns
        -------
        An n x len(self.estimators_) array of meta-level features.
        """
        rows = []
        for j, e in enumerate(self.estimators):
            X = X_b[j]
            if self.proba:
                # Predict label probabilities
                pred = self._base_estimator_predict_proba(e, X)
            else:
                # Predict label values
                pred = self._base_estimator_predict(e, X)
            rows.append(pred)
        return np.hstack(rows)

    def fit(self, Xs, y):
        """ Fit the estimator given predictor(s) X and target y.

        Parameters
        ----------
        X : np.ndarray, shape=(n, m)
            The feature data for which to compute the predicted output.

        y : array of shape = [n_samples]
            The actual outputs (class data).
        """
        # Build meta data.
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

                if self.proba:
                    # Predict label probabilities
                    pred = self._base_estimator_predict_proba(e, X_b)
                else:
                    # Predict label values
                    pred = self._base_estimator_predict(e, X_b)
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



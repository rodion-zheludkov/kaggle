#!/usr/bin/python
# coding=utf-8
import codecs
import numpy as np
import scipy.sparse as sp
import sys
from sklearn.externals.joblib import Parallel, delayed
from sklearn.base import clone
import metrics
import numbers
from sklearn.utils import validation, safe_mask
from sklearn.cross_validation import StratifiedKFold
from sklearn import cross_validation


def log_progress(n=10000):
    if not hasattr(log_progress, 'c'):
        log_progress.c = 0
    log_progress.c += 1
    if log_progress.c % n == 0:
        print 'Processed ' + str(log_progress.c)
        sys.stdout.flush()


def reset_progress():
    log_progress.c = 0


def meta_cross_val_score(estimator, Xs, y, scorer, train, test):
    Xs_train = []
    Xs_test = []
    for X in Xs:
        X_train = X[safe_mask(X, train)]
        X_test = X[safe_mask(X, test)]
        Xs_train.append(X_train)
        Xs_test.append(X_test)

    y_train = y[train]
    y_test = y[test]

    estimator.fit(Xs_train, y_train)

    if scorer is None:
        score = estimator.score(Xs_test, y_test)
    else:
        score = scorer(estimator, Xs_test, y_test)

    return score


def meta_xvalidation(X, y, clf, k=5):
    skf = StratifiedKFold(y, k, indices=True)

    scorer = cross_validation._deprecate_loss_and_score_funcs(
        loss_func=None,
        score_func=None,
        scoring=metrics.mean_avg_at_k
    )
    scores = []
    for train, test in skf:
        scores.append(meta_cross_val_score(clf, X, y, scorer, train, test))

    return scores


def xvalidation(X, y, clf, k=5):
    #skf = StratifiedKFold(y, k, indices=True)
    skf = StratifiedKFold(y, k)
    return cross_validation.cross_val_score(clf, X, y, scoring=metrics.mean_avg_at_k, cv=skf)



def _check_prediction(estimator, X, y, ids, train, test):
    if not hasattr(X, "shape"):
        if getattr(estimator, "_pairwise", False):
            raise ValueError("Precomputed kernels or affinity matrices have "
                             "to be passed as arrays or sparse matrices.")
        X_train = [X[idx] for idx in train]
        X_test = [X[idx] for idx in test]
    else:
        if getattr(estimator, "_pairwise", False):
            # X is a precomputed square kernel matrix
            if X.shape[0] != X.shape[1]:
                raise ValueError("X should be a square kernel matrix")
            X_train = X[np.ix_(train, train)]
            X_test = X[np.ix_(test, train)]
        else:
            X_train = X[safe_mask(X, train)]
            X_test = X[safe_mask(X, test)]

    y_train = y[train]
    y_test = y[test]
    ids_test = [ids[idx] for idx in test]

    y_test_n = sum(y_test)
    print 'ids_test', len(ids_test)
    print 'y_test_n', y_test_n

    estimator.fit(X_train, y_train)
    predicted_scores = estimator.predict_proba(X_test).T[1]

    errors = []
    processed = 0
    # for pred_score, id_test, y_test_i in sorted(zip(predicted_scores, ids_test, y_test), reverse=True)[:y_test_n]:
    for pred_score, id_test, y_test_i in zip(predicted_scores, ids_test, y_test):
        processed += 1
        errors.append((pred_score, id_test))
        #if y_test_i == 0:
        #    errors.append((pred_score, id_test))

    print 'errors', len(errors)
    print 'processed', processed

    return errors


def xvalidation_result(train_file, error_file, X, y, ids, clf, k=5):
    skf = StratifiedKFold(y, k)
    n_jobs = 1
    verbose = 0
    pre_dispatch = '2*n_jobs'
    estimator = clf

    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
    errors_list = parallel(delayed(_check_prediction)(clone(estimator), X, y, ids, train, test)
                      for train, test in skf)

    all_errors = sorted([item for errors in errors_list for item in errors], reverse=True)
    all_errors = dict(map(lambda (a, b): (b, a), all_errors))

    with codecs.open('errors/' + error_file, 'w', 'utf-8') as f:
        for parts in read_train(train_file):
            item_id = parts[0]
            score = all_errors.get(item_id, '')
            if score:
                is_blocked = parts[8]
                line = "\t".join(parts)
                f.write('{0:.20f}'.format(score) + '\t' + is_blocked + '\t' + line)



def read_train(filename, skipfirst=False):
    prev_l = ''
    reset_progress()
    with codecs.open(filename, 'r', 'utf-8') as f:
        if skipfirst:
            f.readline()
        for l in f:
            l = prev_l + l
            parts = l.split('\t')
            if len(parts) >= 13:
                prev_l = ''
                is_blocked = parts[8]
                if is_blocked == '0' or is_blocked == '1':
                    log_progress()
                    yield parts
            else:
                prev_l = l


def read_test(filename, skipfirst=False):
    prev_l = ''
    reset_progress()
    with codecs.open(filename, 'r', 'utf-8') as f:
        if skipfirst:
            f.readline()
        for l in f:
            l = prev_l + l
            parts = l.split('\t')
            if len(parts) >= 10:
                prev_l = ''
                log_progress()
                yield parts
            else:
                prev_l = l
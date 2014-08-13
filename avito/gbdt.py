# coding=utf-8
import codecs
import numpy
import cleantext
import sys
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
import utils


def read_ensemble(**clf_files):
    clfs = []
    for clf_file in clf_files:
        clfs.append(joblib.load(clf_file))

    return clfs


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Usage " + sys.argv[0] + " file"
        sys.exit(1)

    filename = sys.argv[1]

    params = {'max_depth': 8, 'subsample': 0.5, 'verbose': 2, 'random_state': 0,
              'min_samples_split': 20, 'min_samples_leaf': 20, 'max_features': 100,
              'n_estimators': 500, 'learning_rate': 0.05}

    clf = GradientBoostingClassifier(**params)
    #print utils.xvalidation(X, y, clf)

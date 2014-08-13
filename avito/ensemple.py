# coding=utf-8

import sys
import numpy

from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import binarize
from sklearn.linear_model import LogisticRegression

import stacking

import utils
import cleantext


def read_train(train_file):
    lines = []
    y = []
    vectorizer = CountVectorizer(min_df=3)
    tf_idf = TfidfTransformer()

    for parts in utils.read_train(train_file):
        is_blocked = parts[8]
        desc = cleantext.clean(parts[4], False)
        lines.append(desc)
        y.append(int(is_blocked))

    vectorizer = vectorizer.fit_transform(lines)
    X_nb = tf_idf.fit_transform(vectorizer)
    X_log = binarize(vectorizer)

    return X_nb, X_log, numpy.asarray(y)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Usage " + sys.argv[0] + " file_train model_file"
        sys.exit(1)

    train_filename = sys.argv[1]

    nb = MultinomialNB()
    logreg = SGDClassifier(loss="log", penalty="l2", alpha=1e-4, class_weight="auto")

    clfs = [nb, logreg]
    meta_classifier = LogisticRegression()
    clf = stacking.Stacking(meta_classifier, clfs, 5, stackingc=False)

    X_nb, X_log, y = read_train(train_filename)
    X = [X_nb, X_log]
    print 'X_nb ' + str(X_nb.shape)
    print 'X_log ' + str(X_log.shape)

    print utils.meta_xvalidation(X, y, clf)


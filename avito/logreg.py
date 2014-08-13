# coding=utf-8

import sys
import numpy

from sklearn.preprocessing import binarize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier

import utils
import cleantext


def read_train_nb(train_file):
    lines = []
    y = []
    vectorizer = CountVectorizer(min_df=3)
    tf_idf = TfidfTransformer()

    for parts in utils.read_train(train_file):
        is_blocked = parts[8]
        # desc = cleantext.clean(parts[4], False)
        desc = parts[4]
        lines.append(desc)
        y.append(int(is_blocked))

    #nb_features = vectorizer.fit_transform(lines)
    #log_features = binarize(nb_features)

    log_features = tf_idf.fit_transform(vectorizer.fit_transform(lines))

    return log_features, numpy.asarray(y), vectorizer



if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Usage " + sys.argv[0] + " file_train model_file"
        sys.exit(1)

    train_filename = sys.argv[1]
    model_filename = sys.argv[2]

    X, y, vectorizer = read_train_nb(train_filename)
    logreg = SGDClassifier(loss="log", penalty="l2", alpha=1e-4, class_weight="auto")

    print utils.xvalidation(X, y, logreg)

    # nb.fit(X, y)
    # joblib.dump((nb, vectorizer), model_filename)


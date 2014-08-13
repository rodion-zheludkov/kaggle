# coding=utf-8

import sys
import numpy

from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

import utils
import cleantext


def test(X, y, vectorizer):
    nb = MultinomialNB()
    nb.fit(X, y)
    print nb.predict_proba(vectorizer.transform([u'куплю автомобиль', u'массажа таможню']))


def read_train_nb(train_file):
    lines = []
    y = []
    ids = []
    vectorizer = TfidfVectorizer(min_df=3, ngram_range=(1, 2))

    for parts in utils.read_train(train_file):
        ids.append(parts[0])
        is_blocked = parts[8]
        # desc = cleantext.clean(parts[4], False)
        desc = parts[3] + ' ' + parts[4]
        lines.append(desc)
        y.append(int(is_blocked))

    print 'Generate features'
    features = vectorizer.fit_transform(lines)
    print 'Generate features. Done'

    return features, numpy.asarray(y), ids, [vectorizer]


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Usage " + sys.argv[0] + " file_train model_file"
        sys.exit(1)

    train_filename = sys.argv[1]
    model_filename = sys.argv[2]

    X, y, ids, transformers = read_train_nb(train_filename)
    nb = MultinomialNB()

    #print utils.xvalidation(X, y, nb)
    utils.xvalidation_result(train_filename, 'nb.ng2.txt', X, y, ids, nb)

    # nb.fit(X, y)
    # joblib.dump((nb, transformers), model_filename)



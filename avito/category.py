# coding=utf-8

import sys
import numpy
import itertools
from sklearn.base import BaseEstimator
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

import utils
import cleantext

category_dict = {
    u'Бытовая электроника': 0,
    u'Для бизнеса': 1,
    u'Для дома и дачи': 2,
    u'Животные': 3,
    u'Личные вещи': 4,
    u'Недвижимость': 5,
    u'Работа': 6,
    u'Транспорт': 7,
    u'Услуги': 8,
    u'Хобби и отдых': 9
}

category_dict_r = dict(map(reversed, category_dict.items()))


class CategoryNb(BaseEstimator):
    def __init__(self):
        self.n = len(category_dict.keys())
        self.nbs = [MultinomialNB() for i in range(self.n)]

    def fit(self, Xs, ys):
        for i in range(self.n):
            X = [x for (cat, x) in Xs if cat == i]
            y = [y for (cat, y) in ys if cat == i]
            self.nbs[i].fit(Xs[i], y[i])

    def predict(self, I):
        raise NotImplementedError('Use predict_proba')

    def predict_proba(self, Xs):
        result = []
        for (cat, X) in Xs:
            result.append(self.nbs[cat].predict_proba(X)[0])

        return result


def read_train_nb(train_file):
    lines = [[] for _ in range(10)]
    y = [[] for _ in range(10)]
    ids = []

    vectorizer = TfidfVectorizer(min_df=3, ngram_range=(1, 2))

    for parts in utils.read_train(train_file):
        i = category_dict[parts[1]]
        ids.append(parts[0])
        is_blocked = parts[8]
        # desc = cleantext.clean(parts[4], False)
        desc = parts[3] + ' ' + parts[4]
        lines[i].append(desc)
        y[i].append(int(is_blocked))

    print 'Generate features'
    merged = itertools.chain.from_iterable(lines)
    vectorizer.fit(merged)
    print 'Generate features. 1 done'

    y = map(lambda a: numpy.asarray(a), y)
    c_features = []
    for i in range(10):
        c_features.append(vectorizer.transform(lines[i]))

    print 'Generate features. 2 Done'
    sys.stdout.flush()

    return c_features, y#, ids, [vectorizer]


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Usage " + sys.argv[0] + " file_train model_file"
        sys.exit(1)

    train_filename = sys.argv[1]
    model_filename = sys.argv[2]

    X, y = read_train_nb(train_filename)
    xv = []
    sz = []
    for i in range(10):
        sz.append(X[i].shape[0])
        nb = MultinomialNB()
        xv.append(utils.xvalidation(X[i], y[i], nb))

    total_sz = float(sum(sz))
    for i in range(10):
        print category_dict_r[i], sz[i]/total_sz, xv[i]

    #utils.xvalidation_result(train_filename, 'nb.ng2.txt', X, y, ids, nb)

    # nb.fit(X, y)
    # joblib.dump((nb, transformers), model_filename)



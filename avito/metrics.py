import numpy as np
from sklearn.metrics import precision_recall_curve, make_scorer
from itertools import ifilter
import sys


def mean_avg_at_k_score(y_true, y_scores):
    #k = sum(ifilter(lambda y: y == 1, y_true))
    #print 'K is: ' + str(k)
    #sys.stdout.flush()

    k = 55000

    precision, _, _ = precision_recall_curve(y_true, y_scores)
    precision = precision[-k:]
    return sum(precision) / len(precision)

mean_avg_at_k = make_scorer(mean_avg_at_k_score, needs_threshold=True)

if __name__ == '__main__':
    y_true = np.array([0, 1, 0, 1])
    y_scores = np.array([0.1, 0.35, 0.4, 0.8])
    print mean_avg_at_k_score(y_true, y_scores)
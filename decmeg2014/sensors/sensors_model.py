import os

os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
import sensors_data
import config
from sklearn import linear_model
from sklearn.externals import joblib
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV


def report(grid_search):
    print grid_search.best_score_
    print grid_search.best_params_


def save_model(clf, run_id, sendsor_j):
    model_folder = config.model_logreg_folder + '/' + str(run_id)
    if not os.path.isdir(model_folder):
        os.mkdir(model_folder)

    joblib.dump(clf, model_folder + '/' + str(sendsor_j) + '.pkl')


def load_model(i):
    joblib.load(config.model_folder + '/' + str(i) + '.pkl')


def run():
    if not os.path.isdir(config.model_logreg_folder):
        os.mkdir(config.model_logreg_folder)

    for i in range(120):
        XX, Y = sensors_data.read_data(i)
        clf = linear_model.LogisticRegression()
        scores = cross_validation.cross_val_score(clf, XX, Y, cv=16)

        save_model(clf, 'c1', i)

        print str(i) + '\t' + str(scores.mean()) + '\t' + str(scores.std())
        sys.stdout.flush()


def opt_run():
    param_grid = [{'C': [1, 10, 100, 1000]}]
    clf = linear_model.LogisticRegression()
    grid_search = GridSearchCV(clf, cv=16, param_grid=param_grid)
    XX, Y = sensors_data.read_data(0)
    grid_search.fit(XX, Y)
    report(grid_search)

def signle_run():
    XX, Y = sensors_data.read_data_spectr(0)
    clf = linear_model.LogisticRegression()
    scores = cross_validation.cross_val_score(clf, XX, Y, cv=16)
    print str(0) + '\t' + str(scores.mean()) + '\t' + str(scores.std())

if __name__ == '__main__':
    signle_run()
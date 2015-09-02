from __future__ import print_function
import os
from vwsklearn import VW
import numpy as np
import pandas as pd
from hyperopt import fmin, tpe, hp, STATUS_OK
from scipy.stats import zscore
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier


def train_vw(filename):
    vw = VW('/home/' + os.getenv('USER') + '/bin/vw', '/home/rodion/facebids/tmp/',
            {'loss_function': 'hinge', 'passes': '10', 'holdout_off': False,
             'learning_rate': '0.9961190941783473', 'b': '24'})

    with open(filename, 'r') as f:
        xx = f.readlines()
        y = [int(x[0]) - 1 for x in xx]

    y = np.asarray(y)
    print('Crossvalidation:', cross_val_score(vw, xx, y, scoring='roc_auc', cv=StratifiedKFold(y, 4), n_jobs=-1))


def train_vw_hyperopt(filename):
    with open(filename, 'r') as f:
        xx = f.readlines()
        y = [int(x[0]) - 1 for x in xx]

    y = np.asarray(y)

    space = hp.choice('vw', [
        {
            'learning_rate': hp.uniform('learning_rate', 0, 1),
            'passes': hp.quniform('passes', 1, 30, 1),
            'holdout_off': hp.choice('holdout_off', [True, False]),
            'loss_function': hp.choice('loss_function', ['hinge']),
            'b': hp.choice('b', ['24']),
        }
    ])

    def objective(params):
        params['learning_rate'] = str(params['learning_rate'])
        params['passes'] = str(int(params['passes']))

        print(params)

        os.popen('rm -f /home/rodion/facebids/tmp/*')
        vw = VW('/home/' + os.getenv('USER') + '/bin/vw', '/home/rodion/facebids/tmp/', params)
        score = cross_val_score(vw, xx, y, scoring='roc_auc', cv=StratifiedKFold(y, 4), n_jobs=-1)

        var = np.var(score)
        mean = np.mean(score)

        return {'loss': -mean, 'status': STATUS_OK, 'loss_variance': var}

    best = fmin(objective,
                space=space,
                algo=tpe.suggest,
                max_evals=30)

    print(best)


def train_gbdt(filename):
    df = pd.read_csv(filename)
    columns = 'b_total,b_d_total,b_i_total,b_c_total,ai_dt_max,ai_dt_min,ai_dt_avg,ai_dt_std,ai_dt_std_min,' \
              'ai_dt_std_max,a_dt_max,a_dt_min,a_dt_avg,a_dt_std,a_dt_std_min,a_dt_std_max,dt_max,dt_min,dt_avg,dt_std,' \
              'ai_total,ai_d_total,a_total,a_d_total,a_i_total,a_c_total,proba,hist_0,hist_1,hist_2'.split(',')
    for column in columns:
        df[column] = zscore(df[column])

    xx = df[columns]
    y = df['outcome'].values

    ps = {'n_estimators': 155, 'learning_rate': 0.01673821514381137, 'max_depth': 4}
    gbdt = GradientBoostingClassifier(**ps)
    print('Crossvalidation:', cross_val_score(gbdt, xx, y, scoring='roc_auc', cv=StratifiedKFold(y, 4), n_jobs=-1))


def traing_gbdt_hyperopt(filename):
    df = pd.read_csv(filename)
    columns = 'b_total,b_d_total,b_i_total,b_c_total,ai_dt_max,ai_dt_min,ai_dt_avg,ai_dt_std,ai_dt_std_min,' \
              'ai_dt_std_max,a_dt_max,a_dt_min,a_dt_avg,a_dt_std,a_dt_std_min,a_dt_std_max,dt_max,dt_min,dt_avg,dt_std,' \
              'ai_total,ai_d_total,a_total,a_d_total,a_i_total,a_c_total,proba,hist_0,hist_1,hist_2'.split(',')
    for column in columns:
        df[column] = zscore(df[column])

    xx = df[columns]
    y = df['outcome'].values

    space = hp.choice('gbdt', [
        {
            'learning_rate': hp.uniform('learning_rate', 0, 1),
            'n_estimators': hp.quniform('n_estimators', 10, 300, 1),
            'max_depth': hp.quniform('max_depth', 2, 10, 1),
        }
    ])

    def objective(params):
        params['n_estimators'] = int(params['n_estimators'])
        params['max_depth'] = int(params['max_depth'])

        print(params)
        gbdt = GradientBoostingClassifier(**params)
        score = cross_val_score(gbdt, xx, y, scoring='roc_auc', cv=StratifiedKFold(y, 5), n_jobs=-1)

        var = np.var(score)
        mean = np.mean(score)

        return {'loss': -mean, 'status': STATUS_OK, 'loss_variance': var}

    best = fmin(objective,
                space=space,
                algo=tpe.suggest,
                max_evals=100)

    best.pop('gbdt', None)
    best['n_estimators'] = int(best['n_estimators'])
    best['max_depth'] = int(best['max_depth'])

    print('Best', best)
    best_gbdt = GradientBoostingClassifier(**best)
    print('Crossvalidation:', cross_val_score(best_gbdt, xx, y, scoring='roc_auc', cv=StratifiedKFold(y, 4), n_jobs=-1))



if __name__ == '__main__':
    # train_vw('/home/rodion/facebids/train/join.str.csv.vw')
    # train_vw_hyperopt('/home/rodion/facebids/train/join.str.csv.vw')
    train_gbdt('/home/rodion/facebids/train/join.count.proba.time.aff.csv')
    # traing_gbdt_hyperopt('/home/rodion/facebids/train/join.count.proba.time.csv')
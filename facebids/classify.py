from __future__ import print_function
import os
import itertools
from vwsklearn import VW
import numpy as np
import pandas as pd
from hyperopt import fmin, tpe, hp, STATUS_OK
from scipy.stats import zscore
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier


def get_data(filename):
    df = pd.read_csv(filename)
    # columns = ['auctions', 'countries', 'devices', 'ips',
    #            'total', 'proba',
    #            'ip_devices_min', 'ip_devices_max', 'ip_devices_mean', 'ip_devices_std',
    #            'ip_total',
    #            'auc_devices_min', 'auc_devices_max', 'auc_devices_mean', 'auc_devices_std',
    #            'auc_countries_min', 'auc_countries_max', 'auc_countries_mean', 'auc_countries_std',
    #            'auc_ips_min', 'auc_ips_max', 'auc_ips_mean', 'auc_ips_std',
    #            'auc_total',
    #            'dt_max', 'dt_mean', 'dt_min',  'dt_std',
    #            'auc_dt_max', 'auc_dt_mean', 'auc_dt_min', 'auc_dt_std_min', 'auc_dt_std_max', 'auc_dt_std_mean',
    # ]
    columns = 'b_total,b_d_total,b_i_total,b_c_total,ai_dt_max,ai_dt_min,ai_dt_avg,ai_dt_std,ai_dt_std_min,' \
              'ai_dt_std_max,a_dt_max,a_dt_min,a_dt_avg,a_dt_std,a_dt_std_min,a_dt_std_max,dt_max,dt_min,dt_avg,dt_std,' \
              'ai_total,ai_d_total,a_total,a_d_total,a_i_total,a_c_total,proba,hist_0,hist_1,hist_2,' \
              'aff_d_avg,aff_d_max,aff_d_min,aff_d_std,aff_i_avg,aff_i_max,aff_i_min,aff_i_std,aff_u_avg,' \
              'aff_u_max,aff_u_min,aff_u_std'.split(',')
    print()
    for column in columns:
        df[column] = zscore(df[column])
        print(column, df[column].isnull().any())

    xx = df[columns].as_matrix()
    if 'outcome' in df.columns:
        y = df['outcome'].values
    else:
        y = []
    tags = df['bidder_id'].values

    return xx, y, tags, columns


def classify():
    ps = {'n_estimators': 155, 'learning_rate': 0.01673821514381137, 'max_depth': 4}
    xx, y, tags, columns = get_data('/home/rodion/facebids/train/join.count.proba.time.csv')

    gbdt = GradientBoostingClassifier(**ps)
    cv = StratifiedKFold(y, 4)

    for a, b in cv:
        y_a, y_b = y[a], y[b]
        xx_a, xx_b = xx[a], xx[b]
        tags_b = tags[b]

        gbdt.fit(xx_a, y_a)

        sort_indices = np.argsort(np.array(gbdt.feature_importances_))[::-1]
        print(np.asarray(gbdt.feature_importances_)[sort_indices])
        print(np.asarray(columns)[sort_indices])

        proba = gbdt.predict_proba(xx_b)
        proba = proba[:, 1]

        sort_indices = np.argsort(proba)
        a = np.array([tags_b[sort_indices], y_b[sort_indices], proba[sort_indices]]).T
        np.savetxt("foo.csv", a, delimiter=",", fmt="%s")

        break


def classify_test_vw(train_filename, test_filename, proba_filename):
    vw = VW('/home/' + os.getenv('USER') + '/bin/vw', '/home/rodion/facebids/tmp/',
            {'loss_function': 'hinge', 'passes': '10', 'holdout_off': False,
             'learning_rate': '0.9961190941783473', 'b': '24'})

    with open(train_filename, 'r') as f:
        xx = f.readlines()
        y = [int(x[0]) - 1 for x in xx]
        vw.fit(xx, y)

    with open(test_filename, 'r') as f, open(proba_filename, 'w') as fw:
        xx = f.readlines()
        tags, probas = vw.predict_tag_proba(xx)

        fw.write('bidder_id,proba\n')
        for tag, proba in itertools.izip(tags, probas):
            fw.write(tag + ',' + str(proba[1]) + '\n')


def classif_test_gbdt(train_filename, test_filename, proba_filename):
    ps = {'n_estimators': 155, 'learning_rate': 0.01673821514381137, 'max_depth': 4}
    gbdt = GradientBoostingClassifier(**ps)

    xx, y, tags, columns = get_data(train_filename)
    gbdt.fit(xx, y)

    xx, _, tags, columns = get_data(test_filename)
    probas = gbdt.predict_proba(xx)

    with open(proba_filename, 'w') as fw:
        fw.write('bidder_id,proba\n')
        for tag, proba in itertools.izip(tags, probas):
            fw.write(tag + ',' + str(proba[1]) + '\n')


if __name__ == '__main__':
    # classify()
    # classify_test_vw('/home/rodion/facebids/train/join.str.csv.vw',
    #                  '/home/rodion/facebids/test/join.str.csv.vw',
    #                  '/home/rodion/facebids/test/vw.proba.csv')
    classif_test_gbdt('/home/rodion/facebids/train/join.count.proba.time.aff.csv',
                      '/home/rodion/facebids/test/join.count.proba.time.aff.csv',
                      '/home/rodion/facebids/test/gbdt.proba.csv')
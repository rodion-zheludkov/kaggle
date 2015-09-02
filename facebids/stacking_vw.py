from __future__ import print_function
import os
import itertools

import numpy as np
from sklearn.cross_validation import StratifiedKFold

from vwsklearn import VW


def train_vw(filename, output):

    with open(filename, 'r') as f:
        xx = f.readlines()
        y = [int(x[0]) - 1 for x in xx]

    y = np.asarray(y)
    xx = np.asarray(xx)
    cv = StratifiedKFold(y, 4)

    os.popen('rm -f /home/rodion/facebids/tmp/*')

    with open(output, 'w') as fw:
        fw.write('bidder_id,proba\n')
        for a, b in cv:
            y_a, y_b = y[a], y[b]
            xx_a, xx_b = xx[a], xx[b]

            vw = VW('/home/' + os.getenv('USER') + '/bin/vw', '/home/rodion/facebids/tmp/',
                    {'loss_function': 'hinge', 'passes': '10', 'holdout_off': False,
                     'learning_rate': '0.9961190941783473', 'b': '24'})

            vw.fit(xx_a, y_a)
            tags, probas = vw.predict_tag_proba(xx_b)
            for tag, proba in itertools.izip(tags, probas):
                fw.write(tag + ',' + str(proba[1]) + '\n')


if __name__ == '__main__':
    train_vw('/home/rodion/facebids/train/join.str.csv.vw', '/home/rodion/facebids/train/vw.proba.csv')

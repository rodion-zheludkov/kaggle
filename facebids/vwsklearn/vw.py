from __future__ import print_function
import subprocess
import shlex
import math
import random

import string

import numpy as np
import sklearn.base


def _randomword(length):
    return ''.join(random.choice(string.lowercase) for _ in range(length))


def _sigmoid(x):
    return 1 / (1 + math.exp(-x))

class VW(sklearn.base.BaseEstimator):
    def __init__(self, vw_bin, tmp_folder, params):
        self.vw_bin = vw_bin
        self.tmp_folder = tmp_folder.rstrip('/')
        self.model = self.tmp_folder + '/' + 'model.' + _randomword(8)
        self.cache = self.tmp_folder + '/' + 'cache.' + _randomword(8)
        self.params = params

        # print(self.model, self.cache, self.params)

    def _build_train_command(self):
        command = self.vw_bin
        command += ' --loss_function ' + self.params['loss_function']
        if 'l2' in self.params:
            command += ' --l2 ' + self.params['l2']

        if 'oaa' in self.params:
            command += ' --oaa ' + self.params['oaa']
        else:
            command += ' --oaa 2'

        if 'holdout_off' in self.params:
            command += ' --holdout_off'

        if 'bfgs' in self.params:
            command += ' --bfgs'

        if 'passes' in self.params:
            command += ' --passes ' + self.params['passes']

        if 'learning_rate' in self.params:
            command += ' --learning_rate ' + self.params['learning_rate']

        if 'b' in self.params:
            command += ' -b ' + self.params['b']

        if 'q' in self.params:
            command += ' -q ' + self.params['q']

        command += ' -f ' + self.model
        command += ' --cache_file ' + self.cache

        return command

    def _build_test_command(self):
        command = self.vw_bin

        if 'b' in self.params:
            command += ' -b ' + self.params['b']

        if 'q' in self.params:
            command += ' -q ' + self.params['q']

        command += ' -t -i %s -r /dev/stdout' % self.model

        return command

    def fit(self, xx, _):
        stderr = open('/dev/null', 'w')
        command = self._build_train_command()

        print(command)

        vw_process = subprocess.Popen(shlex.split(command), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=stderr, close_fds=True, universal_newlines=True)

        for x in xx:
            x = x.strip()
            vw_process.stdin.write(('%s\n' % x))

        vw_process.stdin.close()
        if vw_process.wait() != 0:
            raise Exception("vw_process %d (%s) exited abnormally with return code %d" %
                            (vw_process.pid, command, vw_process.returncode))

    def predict(self, xx):
        probas = self._predict_proba(xx)
        return np.asarray([proba.index(max(proba)) for proba in probas])

    def _convert_proba(self, prediction):
        if ':' not in prediction[-1]:
            tag = prediction[-1]
            prediction = prediction[:-1]
        else:
            tag = ''

        labels, vs = list(zip(*[[float(x) for x in l.split(':')] for l in prediction]))
        probs = [_sigmoid(v) for v in vs]
        sum_probs = sum(probs)
        if sum_probs != 0.:
            probs = [prob / sum_probs for prob in probs]

        return tag, probs


    def _predict_proba(self, xx, need_tag=False):
        stderr = open('/dev/null', 'w')
        command = self._build_test_command()

        print(command)

        vw_process = subprocess.Popen(shlex.split(command), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=stderr, close_fds=True, universal_newlines=True)
        probas = []
        for x in xx:
            x = x.strip()
            vw_process.stdin.write(('%s\n' % x))
            l = vw_process.stdout.readline().strip()
            proba = self._convert_proba(l.split())
            probas.append(proba)

        vw_process.stdin.close()
        if vw_process.wait() != 0:
            raise Exception("vw_process %d (%s) exited abnormally with return code %d" %
                            (vw_process.pid, command, vw_process.returncode))

        return probas

    def predict_proba(self, xx):
        tag, probas = zip(*self._predict_proba(xx, True))
        return np.asarray(probas)

    def predict_tag_proba(self, xx):
        tag, probas = zip(*self._predict_proba(xx, True))
        return tag, np.asarray(probas)

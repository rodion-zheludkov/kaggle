import numpy as np
import sklearn.base

from vowpal_porpoise import vw


class VW_Classifier(sklearn.base.BaseEstimator):

    """scikit-learn interface for Vowpal Wabbit

    Only works for regression and binary classification.
    """

    def __init__(self,
                 logger=None,
                 vw='vw',
                 moniker='moniker',
                 name=None,
                 bits=None,
                 loss=None,
                 passes=10,
                 log_stderr_to_file=False,
                 silent=False,
                 l1=None,
                 l2=None,
                 learning_rate=None,
                 quadratic=None,
                 audit=None,
                 power_t=None,
                 adaptive=False,
                 working_dir=None,
                 decay_learning_rate=None,
                 initial_t=None,
                 minibatch=None,
                 total=None,
                 node=None,
                 unique_id=None,
                 span_server=None,
                 bfgs=None,
                 oaa=None,
                 old_model=None,
                 incremental=False,
                 mem=None,
                 nn=None,
                 holdoff=False,
                 weights=None
                 ):
        self.logger = logger
        self.vw = vw
        self.moniker = moniker
        self.name = name
        self.bits = bits
        self.loss = loss
        self.passes = passes
        self.log_stderr_to_file = log_stderr_to_file
        self.silent = silent
        self.l1 = l1
        self.l2 = l2
        self.learning_rate = learning_rate
        self.quadratic = quadratic
        self.audit = audit
        self.power_t = power_t
        self.adaptive = adaptive
        self.working_dir = working_dir
        self.decay_learning_rate = decay_learning_rate
        self.initial_t = initial_t
        self.minibatch = minibatch
        self.total = total
        self.node = node
        self.unique_id = unique_id
        self.span_server = span_server
        self.bfgs = bfgs
        self.oaa = oaa
        self.old_model = old_model
        self.incremental = incremental
        self.mem = mem
        self.nn = nn
        self.holdoff = holdoff
        self.weights = weights

    def init_wv(self):
        self.vw_ = vw.VW(
            logger=self.logger,
            vw=self.vw,
            moniker=self.moniker,
            name=self.name,
            bits=self.bits,
            loss=self.loss,
            passes=self.passes,
            log_stderr_to_file=self.log_stderr_to_file,
            silent=self.silent,
            l1=self.l1,
            l2=self.l2,
            learning_rate=self.learning_rate,
            quadratic=self.quadratic,
            audit=self.audit,
            power_t=self.power_t,
            adaptive=self.adaptive,
            working_dir=self.working_dir,
            decay_learning_rate=self.decay_learning_rate,
            initial_t=self.initial_t,
            minibatch=self.minibatch,
            total=self.total,
            node=self.node,
            unique_id=self.unique_id,
            span_server=self.span_server,
            bfgs=self.bfgs,
            oaa=self.oaa,
            old_model=self.old_model,
            incremental=self.incremental,
            mem=self.mem,
            nn=self.nn,
            holdoff=self.holdoff
        )

    def fit(self, X, y):
        self.init_wv()

        examples = [_as_vw_string(X[i], self.weights, y[i]) for i in range(np.shape(X)[0])]

        # add examples to model
        with self.vw_.training():
            for instance in examples:
                self.vw_.push_instance(instance)

        # learning done after "with" statement
        return self

    def predict(self, X):
        examples = [_as_vw_string(X[i], self.weights, 1) for i in range(np.shape(X)[0])]

        # add test examples to model
        with self.vw_.predicting():
            for instance in examples:
                self.vw_.push_instance(instance)

        # read out predictions
        predictions = np.asarray(list(self.vw_.read_predictions_()))

        return predictions

    def predict_proba(self, X):
        predictions = self.predict(X)
        return np.asarray([convert_proba(prediction.split()) for prediction in predictions])

    def predict_ranks(self, X):
        predictions = self.predict(X)
        return np.asarray([convert_ranks(prediction.split()) for prediction in predictions])

    def predict_proba_iter(self, X_iter):
        # initialize model
        self.init_wv()

        # add test examples to model
        with self.vw_.predicting():
            for key, sample in X_iter:
                instance = _as_vw_string(sample, None, None, key)
                self.vw_.push_instance(instance)

        for prediction in self.vw_.read_predictions_():
            yield convert_prediction_proba_tag(prediction)

    def predict_ranks_iter(self, X_iter):
        # initialize model
        self.init_wv()

        # add test examples to model
        with self.vw_.predicting():
            for key, sample in X_iter:
                instance = _as_vw_string(sample, None, None, key)
                self.vw_.push_instance(instance)

        for prediction in self.vw_.read_predictions_():
            yield convert_prediction_rank_tag(prediction)

sigmoid = lambda x: 1/(1+np.exp(-x))


def convert_proba(prediction):
    probs = sigmoid(convert_ranks(prediction))
    if probs.sum() != 0.:
        probs = probs / probs.sum()
    return probs


def convert_ranks(prediction):
    labels, vs = list(zip(*[[float(x) for x in l.split(':')] for l in prediction[:]]))
    return np.asarray(vs)


def convert_prediction_proba_tag(prediction):
    prediction = prediction.split()
    key = prediction[-1]
    probs = convert_proba(prediction[:-1])
    return key, probs

def convert_prediction_rank_tag(prediction):
    prediction = prediction.split()
    key = prediction[-1]
    probs = convert_ranks(prediction[:-1])
    return key, probs


def _as_vw_string(x, weights, y=None, tag=None):
    if y:
        result = str(y) + ' '
    else:
        result = "1 "

    if weights:
        result += str(weights[y-1]) + ' '

    if tag:
        result += "'" + tag

    for namespace_name, namespace in x.items():
        namespace_str = " ".join(["%s:%f" % (key, value) for (key, value) in namespace.items()])
        result += '|' + namespace_name + ' ' + namespace_str + ' '

    return result

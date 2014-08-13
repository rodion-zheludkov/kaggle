import numpy
import sys

import theano
import theano.tensor as T

from logreg import LogisticRegression


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        self.input = input

        if W is None:
            W_values = numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6. / (n_in + n_out)),
                high=numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]


class MLP(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        self.hiddenLayer = HiddenLayer(rng=rng, input=input,
                                       n_in=n_in, n_out=n_hidden,
                                       activation=T.tanh)

        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out)

        self.L1 = abs(self.hiddenLayer.W).sum() + abs(self.logRegressionLayer.W).sum()

        self.L2_sqr = (self.hiddenLayer.W ** 2).sum() + (self.logRegressionLayer.W ** 2).sum()

        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        self.errors = self.logRegressionLayer.errors

        self.params = self.hiddenLayer.params + self.logRegressionLayer.params


def build_model(x, y):
    learning_rate = 0.01
    L1_reg = 0.00
    L2_reg = 0.0001

    print '... building the model'
    sys.stdout.flush()

    rng = numpy.random.RandomState(1234)
    #classifier = MLP(rng=rng, input=x, n_in=76500, n_hidden=100, n_out=2)
    classifier = MLP(rng=rng, input=x, n_in=25500, n_hidden=100, n_out=2)
    cost = classifier.negative_log_likelihood(y) + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr

    gparams = []
    for param in classifier.params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)

    updates = []
    for param, gparam in zip(classifier.params, gparams):
        updates.append((param, param - learning_rate * gparam))

    train_model = theano.function(inputs=[x, y], outputs=cost, updates=updates)
    validate_model = theano.function(inputs=[x, y], outputs=classifier.errors(y))
    test_model = theano.function(inputs=[x, y], outputs=classifier.errors(y))

    return train_model, validate_model, test_model


def train_batch(x, y, x_train, y_train, x_valid, y_valid, x_test, y_test):
    batch_size = 10

    train_model, validate_model, test_model = build_model(x, y)

    print '... training the model'
    sys.stdout.flush()

    n_train_batches = x_train.shape[0] / batch_size
    n_valid_batches = x_valid.shape[0] / batch_size
    n_test_batches = x_test.shape[0] / batch_size

    patience = 1000
    patience_increase = 2
    improvement_threshold = 0.995
    n_epochs = 100

    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = numpy.inf
    test_score = 0.

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in xrange(n_train_batches):
            train_model(x_train[minibatch_index * batch_size: (minibatch_index + 1) * batch_size],
                        y_train[minibatch_index * batch_size: (minibatch_index + 1) * batch_size])

            iter_num = (epoch - 1) * n_train_batches + minibatch_index

            if (iter_num + 1) % validation_frequency == 0:
                validation_losses = [validate_model(x_valid[i * batch_size: (i + 1) * batch_size],
                                                    y_valid[i * batch_size: (i + 1) * batch_size])
                                     for i in xrange(n_valid_batches)]

                this_validation_loss = numpy.mean(validation_losses)

                # print('epoch %i, minibatch %i/%i, validation error %f %%' %
                #       (epoch, minibatch_index + 1, n_train_batches,
                #        this_validation_loss * 100.))

                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter_num * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(x_test[i * batch_size: (i + 1) * batch_size],
                                              y_test[i * batch_size: (i + 1) * batch_size])
                                   for i in xrange(n_test_batches)]

                    test_score = numpy.mean(test_losses)

                    # print('epoch %i, minibatch %i/%i, test error of best model %f %%' %
                    #       (epoch, minibatch_index + 1, n_train_batches, test_score * 100.))

            if patience <= iter_num:
                done_looping = True
                break

    print '... training done'
    sys.stdout.flush()

    return 1 - test_score


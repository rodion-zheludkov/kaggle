import numpy
from glob import glob

import theano
import theano.tensor as T
import theano.printing as printing

import config
import gen_vw
import utils


class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        #self.W = T.dmatrix()
        self.W = theano.shared(
            value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='W', borrow=True)
        #self.b = T.dvector()
        self.b = theano.shared(
            value=numpy.zeros((n_out,), dtype=theano.config.floatX),
            name='b', borrow=True)
        self.dot = T.dot(input, self.W) + self.b
        self.p_y_given_x = T.nnet.softmax(self.dot)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                            ('y', y.type, 'y_pred', self.y_pred.type))

        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def printW(self):
        printW = printing.Print("W: ")
        theano.function([], printW(self.W))()

    def printDot(self, x, x_value):
        printDot = printing.Print("x*W + b: ")
        theano.function([x], printDot(self.dot))(x_value)

    def printY(self, x, x_value):
        printY = printing.Print("Y: ")
        theano.function([x], printY(self.p_y_given_x))(x_value)


def build_model(x, y, learning_rate=0.13):
    print '... building the model'
    classifier = LogisticRegression(input=x, n_in=76500, n_out=2)
    #classifier = LogisticRegression(input=x, n_in=25500, n_out=2)
    cost = classifier.negative_log_likelihood(y)

    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    train_model = theano.function(inputs=[x, y], outputs=cost, updates=updates)
    validate_model = theano.function(inputs=[x, y], outputs=classifier.errors(y))
    test_model = theano.function(inputs=[x, y], outputs=classifier.errors(y))

    return train_model, validate_model, test_model


def train_batch(x, y, x_train, y_train, x_valid, y_valid, x_test, y_test):
    batch_size = 10

    train_model, validate_model, test_model = build_model(x, y)

    print '... training the model'

    n_train_batches = x_train.shape[0] / batch_size
    n_valid_batches = x_valid.shape[0] / batch_size
    n_test_batches = x_test.shape[0] / batch_size

    patience = 5000
    patience_increase = 2
    improvement_threshold = 0.995
    n_epochs = 1000

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

                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter_num * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(x_test[i * batch_size: (i + 1) * batch_size],
                                              y_test[i * batch_size: (i + 1) * batch_size])
                                   for i in xrange(n_test_batches)]

                    test_score = numpy.mean(test_losses)

                    print('epoch %i, minibatch %i/%i, test error of best model %f %%' %
                          (epoch, minibatch_index + 1, n_train_batches, test_score * 100.))

            if patience <= iter_num:
                done_looping = True
                break

    print '... training done'

    return test_score


def run_single(train_files_glob, i):
    x_train, y_train = gen_vw.read_train_data(train_files_glob, i)
    x_test, y_test = gen_vw.read_test_data(train_files_glob, i)
    x_valid, y_valid = gen_vw.read_valid_data(train_files_glob, i)

    x = T.matrix('x')
    y = T.ivector('y')

    error = train_batch(x, y, x_train, y_train, x_valid, y_valid, x_test, y_test)
    return 1 - error


if __name__ == '__main__':
    tfile = config.train_folder + "train_subject*.mat"
    print run_single(tfile, 0)

    #tfile = config.train_folder + "*.mat"
    #precisions = utils.parmap(lambda i: run_single(tfile, i), len(glob(tfile)) - 1)
    #print sum(precisions) / len(precisions)

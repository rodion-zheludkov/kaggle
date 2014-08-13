import sys
import numpy
import operator
import cPickle
import theano
import theano.tensor as T

import config
from logreg import LogisticRegression
from mlp import HiddenLayer
import spatial_read
import convnet3d


def get_hyper_param():
    n_input = []
    n_output = []
    kernels = []
    videos = []
    pools = []

    # time, height, width
    kernels.append((5, 2, 2))
    videos.append((config.time_slice, config.height, config.width))
    pools.append((5, 1, 2))

    n_input.append(3)
    n_output.append(18)

    videos.append(map(lambda (a, b, c): (a - b + 1) / c, zip(videos[0], kernels[0], pools[0])))
    kernels.append((3, 2, 2))
    pools.append((4, 1, 1))

    n_input.append(n_output[0])
    n_output.append(27)

    videos.append(map(lambda (a, b, c): (a - b + 1) / c, zip(videos[1], kernels[1], pools[1])))

    hidden_input = reduce(operator.mul, videos[2]) * n_output[1]

    print 'video_shape', videos
    print 'kernel_shape', kernels
    print 'pools shape', pools
    print 'inputs feauture maps', n_input
    print 'output feature maps', n_output
    print 'hidden input', hidden_input
    print 'batch size', config.batch_size
    sys.stdout.flush()

    return videos, kernels, pools, n_input, n_output, hidden_input



def dump_model(params):
    with open(config.model_spatial_folder + 'cnn.save', 'wb') as fw:
        cPickle.dump(params, fw, protocol=cPickle.HIGHEST_PROTOCOL)


def train_batch(x, y, x_train, y_train, x_valid, y_valid):
    batch_size = config.batch_size
    videos, kernels, pools, n_input, n_output, hidden_input = get_hyper_param()
    model = convnet3d.ConvNet(x, y, config.batch_size, videos, kernels, pools, n_input, n_output, hidden_input)
    train_model, validate_model, params = model.train_model, model.validate_model, model.params

    print '... training the model'
    sys.stdout.flush()

    n_train_batches = x_train.shape[0] / batch_size
    n_valid_batches = x_valid.shape[0] / batch_size

    patience = 10000
    patience_increase = 2
    improvement_threshold = 0.995
    n_epochs = 100

    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = 1.
    test_score = 0.

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in xrange(n_train_batches):
            x_train_batch = x_train[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            y_train_batch = y_train[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]

            train_model(x_train_batch, y_train_batch)

            iter_num = (epoch - 1) * n_train_batches + minibatch_index

            if (iter_num + 1) % validation_frequency == 0:
                validation_losses = [validate_model(x_valid[i * batch_size: (i + 1) * batch_size],
                                                    y_valid[i * batch_size: (i + 1) * batch_size])
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                #train_losses = [validate_model(x_train[i * batch_size: (i + 1) * batch_size],
                #                               y_train[i * batch_size: (i + 1) * batch_size])
                #                for i in xrange(n_valid_batches)]
                #this_train_loss = numpy.mean(train_losses)

                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter_num * patience_increase)

                    best_validation_loss = this_validation_loss
                    dump_model(params)

                print('epoch %i, minibatch %i/%i, train precision %f, precision %f %%, best precision %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches, 1. - this_validation_loss,
                       1. - this_validation_loss, 1. - best_validation_loss))
                sys.stdout.flush()

            if patience <= iter_num:
                done_looping = True
                break

    print '... training done'
    sys.stdout.flush()

    return test_score

def run_all():
    print '... reading'
    sys.stdout.flush()

    x_train, y_train, x_test, y_test = spatial_read.read_all(config.trainfiles)
    print '... reading done ', x_train.shape, x_test.shape
    print '... reading done ', y_train.shape, y_test.shape

    #print x_train
    #print y_train

    sys.stdout.flush()

    x = T.matrix('x')
    x = x.reshape((config.batch_size, 3, config.time_slice, config.height, config.width))
    y = T.ivector('y')

    error = train_batch(x, y, x_train, y_train, x_test, y_test)
    return 1 - error


def run_single(i):
    print '... reading'
    sys.stdout.flush()

    x_train, y_train, x_test, y_test = spatial_read.read_tr_te(config.trainfiles, i)
    print '... reading done ', x_train.shape, x_test.shape
    print '... reading done ', y_train.shape, y_test.shape

    #print x_train
    #print y_train

    sys.stdout.flush()

    x = T.matrix('x')
    x = x.reshape((config.batch_size, 3, config.time_slice, config.height, config.width))
    y = T.ivector('y')

    error = train_batch(x, y, x_train, y_train, x_test, y_test)
    return 1 - error


if __name__ == '__main__':
    #print run_single(0)
    print run_all()

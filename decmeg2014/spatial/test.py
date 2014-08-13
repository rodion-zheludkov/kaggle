import cPickle
import numpy
import sys
import theano

import theano.tensor as T

import cnn
import config
import convnet3d
import spatial_read

#theano.config.compute_test_value = 'warn'

def read_model(filename, x, y):
    with open(config.model_spatial_folder + filename, 'rb') as fr:
        params = cPickle.load(fr)
        videos, kernels, pools, n_input, n_output, hidden_input = cnn.get_hyper_param()
        model = convnet3d.ConvNet(x, y, test_batch_size, videos, kernels, pools, n_input, n_output, hidden_input,
                                  params)
        return model


def test(filename, x, y, x_test):
    model = read_model(filename, x, y)
    return model.predict(x_test)


def print_result(fw, result, result_i):
    for i in range(len(result)):
        index = str(17 + result_i) + "%03d" % i
        fw.write(index + ',' + str(result[i]) + '\n')


if __name__ == '__main__':

    # model = read_model(x, y)
    # theano.printing.debugprint(model.predict)

    with open('2.4.csv', 'w') as fw:
        fw.write('Id,Prediction\n')
        for i, testfile in enumerate(config.testfiles):
            x_test = spatial_read.read_te([testfile])

            test_batch_size = x_test.shape[0]
            x = T.matrix('x')
            x = x.reshape((test_batch_size, 3, config.time_slice, config.height, config.width))
            y = T.ivector('y')

            result = test('2.4.save', x, y, x_test)
            print '.. done', result.shape
            sys.stdout.flush()

            print_result(fw, result, i)
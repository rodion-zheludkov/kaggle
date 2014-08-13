import os
os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import pandas
import numpy
from matplotlib import mlab

def read_data(sensor_j):
    samples = [config.train_logreg_folder + '/' + str(i) + '/' for i in range(16)]

    XX, Y = None, None
    for sample_folder in samples:

        records = pandas.read_csv(sample_folder + str(sensor_j) + '.csv').as_matrix()
        if XX is None:
            XX = records
        else:
            XX = numpy.concatenate((XX, records))

        records = pandas.read_csv(sample_folder + 'Y' + '.csv').as_matrix()
        if Y is None:
            Y = records
        else:
            Y = numpy.concatenate((Y, records))

    return XX, Y.reshape(Y.shape[0])


def read_data_spectr(sensor_j):
    print 'reading..'
    XX, Y = read_data(sensor_j)
    XX = XX[:, 500:]

    X_result = None
    for X in XX:
        P, freqs, bins = mlab.specgram(X, NFFT=20, noverlap=10)
        P = P.reshape((1, P.shape[0] * P.shape[1]))
        if X_result is None:
            X_result = P
        else:
            X_result = numpy.concatenate((X_result, P))

    print 'reading done..'
    return X_result, Y

if __name__ == '__main__':
    XX, Y = read_data_spectr(0)
    print XX.shape
    print Y.shape

    print Y
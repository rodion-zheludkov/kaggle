import numpy
from scipy.io import loadmat, savemat
import sklearn
from sklearn.decomposition import RandomizedPCA
from multiprocessing import Process, Pipe
from sklearn.utils import extmath

import config
import utils

train = False


def get_matrix(X, whiten=False):
    U, S, V = sklearn.utils.extmath.randomized_svd(X, X.shape[0],
                                                   n_iter=3,
                                                   random_state=sklearn.utils.validation.check_random_state(None))
    if whiten:
        U = U / S[:, numpy.newaxis] * numpy.sqrt(X.shape[1])
        print 'U,S,V', U.shape, S.shape, V.shape

    return U


def debug_plot():
    file = config.trainfiles[0]
    print 'Reading ' + file
    data = loadmat(config.train_folder + file, squeeze_me=True)
    XX = data['X']

    XX -= XX.mean(0)
    XX = numpy.nan_to_num(XX / XX.std(0))

    # for j in [0, 1, 2]:
    for j in [2]:
        #m = XX[:, j::3].mean(0)
        trial_noise = XX[0, j::3]
        trial_signal = XX[0, j::3]

        noise = trial_noise[:, :125]
        signal = trial_signal
        print 'noise', noise.shape
        print 'signal', signal.shape

        W = get_matrix(noise, True)
        print 'W,signal', W.shape, signal.shape
        s = numpy.dot(W.T, signal)

        # E = get_matrix(s)
        # Ep = E.copy()
        # Ep[:, 10:] = 0
        #
        # print 'E,s', E.shape, s.shape
        # s = numpy.dot(Ep.T, s)
        #
        # print 'E,s', E.shape, s.shape
        # s = numpy.dot(E, s)
        #
        # WtI = numpy.linalg.inv(W.T)
        # print 'WtI,s', WtI.shape, s.shape
        #
        # s = numpy.dot(WtI, s)


        '''
        noise = trial_noise[:, :125].T
        signal = trial_signal.T

        pca_noise = RandomizedPCA(whiten=True).fit(noise)
        Wt = pca_noise.components_
        s = numpy.dot(Wt, signal.T)

        pca_signal = RandomizedPCA().fit(s)
        Et = pca_signal.components_.copy()
        Ett = pca_signal.components_.copy()
        Ett[10:, :] = 0

        s = numpy.dot(Ett, s.T)
        s = numpy.dot(Et.T, s)
        WtI = numpy.linalg.inv(Wt)
        s = numpy.dot(WtI, s.T)
        '''


        print 's', s.shape

        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)


        for i in range(trial_noise.shape[0]):
            ax1.plot([f for f in range(375)], trial_signal[i])

        for i in range(len(s)):
            ax2.plot([f for f in range(375)], s[i])

    plt.show()


def plot():
    data = loadmat(config.train_folder + 'train_subject01.mat', squeeze_me=True)
    napca_data = loadmat(config.train_napca_folder + 'train_subject01.mat', squeeze_me=True)
    XX = data['X'][0]
    s = napca_data['X'][0]

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    for i in range(40):
        ax1.plot([f for f in range(250)], XX[i][125:])

    for i in range(40):
        ax2.plot([f for f in range(250)], s[i])

    plt.show()


def do_fit(X):
    n_samples = X.shape[0]

    n_components = X.shape[1]

    U, S, V = extmath.randomized_svd(X, n_components, n_iter=3)

    return U, S, V


def convert_train_file(file):
    print 'Reading ' + file
    if train:
        data = loadmat(config.train_folder + file, squeeze_me=True)
        XX = data['X']
        y = data['y']
    else:
        data = loadmat(config.test_folder + file, squeeze_me=True)
        XX = data['X']

    # XX -= XX.mean(0)
    # XX = numpy.nan_to_num(XX / XX.std(0))

    result_list = []
    for i in range(XX.shape[0]):
        trial_result = []
        for j in [0, 1, 2]:
            trial = XX[i][j::3]

            '''
            noise = numpy.transpose(trial[:, :125])
            signal = numpy.transpose(trial[:, 125:])


            pca_noise = RandomizedPCA(whiten=True).fit(noise)
            Wt = pca_noise.components_
            s = numpy.dot(Wt, signal.T)

            pca_signal = RandomizedPCA().fit(s)
            Et = pca_signal.components_.copy()
            Ett = pca_signal.components_.copy()
            Ett[10:, :] = 0

            s = numpy.dot(Ett, s.T)
            s = numpy.dot(Et.T, s)
            WtI = numpy.linalg.inv(Wt)
            s = numpy.dot(WtI, s.T)
            '''

            noise = trial[:, :125]
            signal = trial[:, 125:]

            W = get_matrix(noise, True)
            s = numpy.dot(W.T, signal)

            E = get_matrix(s)
            Ep = E.copy()
            Ep[:, 10:] = 0

            s = numpy.dot(Ep.T, s)
            s = numpy.dot(E, s)

            WtI = numpy.linalg.inv(W.T)

            s = numpy.dot(WtI, s)

            trial_result.append(s)

        result_list.append(numpy.vstack((trial_result[0], trial_result[1], trial_result[2])))

    new_data = {}
    new_data['X'] = numpy.asarray(result_list)
    if train:
        new_data['y'] = y

    print 'New X shape', new_data['X'].shape
    if train:
        savemat(config.train_napca_folder + file, new_data)
    else:
        savemat(config.test_napca_folder + file, new_data)


def convert_data():
    if train:
        files = config.trainfiles
        num = 16
    else:
        files = config.testfiles
        num = 7

    utils.parmap(convert_train_file, files, num)


if __name__ == '__main__':
    # convert_data()
    debug_plot()
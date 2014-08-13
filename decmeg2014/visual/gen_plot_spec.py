from scipy import ndimage
from matplotlib import mlab
import numpy
import config
import matplotlib.pyplot as plt
from scipy.io import loadmat


def plot_specgram(P, freqs, bins):
    """Spectrogram"""
    Z = numpy.flipud(P)

    xmin, xmax = 0, numpy.amax(bins)
    extent = xmin, xmax, freqs[0], freqs[-1]

    plt.figure()
    plt.imshow(Z, extent=extent)
    plt.axis('auto')
    #plt.xlim([0.0, bins[-1]])
    #plt.ylim([0, 1])


def plot_no_sub(XX, index_1, index_2):
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.pcolormesh(XX[index_1])
    ax2.pcolormesh(XX[index_2])

    plt.show()


def plot(XX, index_1, index_2):
    #XX = ndimage.median_filter(XX, 3)
    XX0 = XX[:, 0::3, :]
    XX1 = XX[:, 1::3, :]
    XX2 = XX[:, 2::3, :]

    fig = plt.figure()
    ax1 = fig.add_subplot(611)
    ax2 = fig.add_subplot(612)
    ax3 = fig.add_subplot(613)
    ax4 = fig.add_subplot(614)
    ax5 = fig.add_subplot(615)
    ax6 = fig.add_subplot(616)

    x_sample21 = XX2[index_1]
    x_sample22 = XX2[index_2]
    x_sample01 = XX0[index_1]
    x_sample02 = XX0[index_2]
    x_sample11 = XX1[index_1]
    x_sample12 = XX1[index_2]

    ax1.pcolormesh(x_sample01)
    ax2.pcolormesh(x_sample11)
    ax3.pcolormesh(x_sample21)

    ax4.pcolormesh(x_sample02)
    ax5.pcolormesh(x_sample12)
    ax6.pcolormesh(x_sample22)

    plt.show()


def read_XX_0():
    data = loadmat(config.train_folder + 'train_subject01.mat', squeeze_me=True)
    XX = data['X']
    Y = data['y']

    XX_0 = XX[Y == 0]
    XX_1 = XX[Y == 1]

    XX_0_0 = XX_0[:, 0::3, :].mean(0)
    XX_0_1 = XX_0[:, 1::3, :].mean(0)
    XX_0_2 = XX_0[:, 2::3, :].mean(0)

    XX_1_0 = XX_1[:, 0::3, :].mean(0)
    XX_1_1 = XX_1[:, 1::3, :].mean(0)
    XX_1_2 = XX_1[:, 2::3, :].mean(0)


    return XX_0_0, XX_0_1, XX_0_2, XX_1_0, XX_1_1, XX_1_2


if __name__ == '__main__':
    XX_0_0, XX_0_1, XX_0_2, XX_1_0, XX_1_1, XX_1_2 = read_XX_0()

    P, freqs, bins = mlab.specgram(XX_0_2[0], NFFT=16, noverlap=8, Fs=1)
    plot_specgram(P, freqs, bins)

    P, freqs, bins = mlab.specgram(XX_1_2[0], NFFT=16, noverlap=8, Fs=1)
    plot_specgram(P, freqs, bins)

    plt.show()

    '''
    data = loadmat(config.train_folder + 'train_subject01.mat', squeeze_me=True)
    XX = data['X']
    plot(XX, 187, 188)
    '''

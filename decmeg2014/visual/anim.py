import matplotlib
#matplotlib.use('qt4agg')

import matplotlib.animation as animation
import numpy
from scipy.io import loadmat
import matplotlib.pyplot as plt
import config
from scipy.interpolate import griddata
from numpy import linspace


def read_positions(channel_type):
    positions = []

    with open('0.txt') as f:
        dots = []
        for l in f:
            n, x, y = map(lambda x: x.strip(), l.split(' '))
            dots.append((float(x), float(y)))
        positions.append(dots)

    with open('1.txt') as f:
        dots = []
        for l in f:
            n, x, y = map(lambda x: x.strip(), l.split(' '))
            dots.append((float(x), float(y)))
        positions.append(dots)

    with open('2.txt') as f:
        dots = []
        for l in f:
            n, x, y = map(lambda x: x.strip(), l.split(' '))
            dots.append((float(x), float(y)))
        positions.append(dots)

    x, y = [], []
    for i in positions[channel_type]:
        x.append(i[0])
        y.append(i[1])

    return x, y


def read_XX(channel_type, sample_0, sample_1):
    data = loadmat(config.train_folder + 'train_subject01.mat', squeeze_me=True)
    XX = data['X']
    #print list(enumerate(data['y']))

    XX -= XX.mean(0)
    XX = numpy.nan_to_num(XX / XX.std(0))

    #XX = numpy.nan_to_num(XX / XX.max(0))
    XX_0 = XX[sample_0, channel_type::3, :]
    XX_1 = XX[sample_1, channel_type::3, :]

    return XX_0, XX_1


def read_XX_clean(channel_type, sample_0, sample_1):
    data = loadmat(config.train_folder + 'train_subject01.mat', squeeze_me=True)
    XX = data['X']
    Y = data['y']
    #print list(enumerate(data['y']))

    #XX_0 = XX[sample_0, channel_type::3, :]
    #XX_1 = XX[sample_1, channel_type::3, :]

    # mean by samples, then mean by time
    baseline = XX.mean(0)[channel_type::3, :125].mean(1)[:, numpy.newaxis]
    XX_0 = XX[Y == 0]
    XX_1 = XX[Y == 1]

    XX_0 = XX_0.mean(0)[channel_type::3, :]
    XX_1 = XX_1.mean(0)[channel_type::3, :]

    data_0 = XX_0[:, 125:] - baseline

    data_1 = XX_1[:, 125:] - baseline
    #data_1 = data_1 - data_0

    #XX -= XX.mean(0)
    #XX = numpy.nan_to_num(XX / XX.std(0))

    return data_0, data_1

def read_XX_mean(channel_type):
    data = loadmat(config.train_folder + 'train_subject01.mat', squeeze_me=True)
    XX = data['X']
    Y = data['y']

    #XX -= XX.mean(0)
    #XX = numpy.nan_to_num(XX / XX.std(0))

    XX_0 = XX[Y == 0]
    XX_1 = XX[Y == 1]

    XX_0 = XX_0.mean(0)[channel_type::3, :]
    XX_1 = XX_1.mean(0)[channel_type::3, :]

    return XX_0, XX_1

def update_contour_plot(i, zi_0, zi_1, ax0, ax1, fig, xi, yi):
    ax0.cla()
    im0 = ax0.contourf(xi, yi, zi_0[i], 30, cmap=plt.cm.Spectral)

    ax1.cla()
    im1 = ax1.contourf(xi, yi, zi_1[i], 30, cmap=plt.cm.Spectral)

    plt.title(str(i))

    return im0, im1


if __name__ == '__main__':
    channel_type = 2
    sample_0 = 14
    sample_1 = 13

    #XX_0, XX_1 = read_XX(channel_type, sample_0, sample_1)
    XX_0, XX_1 = read_XX_mean(channel_type)
    XX_0, XX_1 = read_XX_clean(channel_type, 0, 2)

    print XX_0.shape, XX_1.shape
    print XX_0
    print XX_1

    x, y = read_positions(channel_type)

    fig = plt.figure(figsize=(16, 9))
    ax0 = fig.add_subplot(121)
    ax1 = fig.add_subplot(122)

    plt.xlim(min(x) - 5, max(x) + 5)
    plt.ylim(min(y) - 5, max(y) + 5)

    xi = linspace(-90, 90, 180)
    yi = linspace(-90, 90, 180)

    zi_0 = []
    zi_1 = []
    for i in range(XX_1.shape[1]):
        zi_0.append(griddata((x, y), XX_0[:, i], (xi[None, :], yi[:, None]), method='cubic'))
        zi_1.append(griddata((x, y), XX_1[:, i], (xi[None, :], yi[:, None]), method='cubic'))

    #ax0.contourf(xi, yi, zi_0[0], 30, cmap=plt.cm.Spectral)
    #ax1.contourf(xi, yi, zi_1[0], 30, cmap=plt.cm.Spectral)

    ani = animation.FuncAnimation(fig, update_contour_plot, frames=xrange(XX_1.shape[1]),
                                  fargs=(zi_0, zi_1, ax0, ax1, fig, xi, yi),
                                  interval=300)

    plt.show()
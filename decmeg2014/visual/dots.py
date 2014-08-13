from matplotlib import pyplot
import numpy
from scipy.io import loadmat
from math import sqrt, atan2, pi
import math
import matplotlib.pyplot as plt
import config


def scale_gcf(scale_factor, debugText=True):
    figure = pyplot.gcf()
    inches_size = map(lambda x: float(x), figure.get_size_inches())
    new_inches_size = map(lambda x: x * scale_factor, inches_size)
    dpi = [figure.get_dpi() for i in range(len(inches_size))]
    figure.set_size_inches(new_inches_size)
    if debugText:
        print "Scale factor: " + str(scale_factor)
        print "Old inches size: " + str(inches_size)
        print "New inches size: " + str(new_inches_size)
        print "Old pixel size: " + str(map(lambda x, y: x * y, dpi, inches_size))
        print "New pixel size: " + str(map(lambda x, y: x * y, dpi, new_inches_size))


def read_positions():
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

    return positions


def get_center(positions):
    min_x = min(map(lambda (x, y): x, positions))
    max_x = max(map(lambda (x, y): x, positions))
    min_y = min(map(lambda (x, y): y, positions))
    max_y = max(map(lambda (x, y): y, positions))
    center_x = min_x + ((max_x - min_x) / 2)
    center_y = min_y + ((max_y - min_y) / 2)
    return center_x, center_y


def draw_dots():
    pyplot.close('all')
    ax = pyplot.axes(xlim=(-90, 90), ylim=(-90, 90))

    positions = read_positions()
    positions = positions[2]

    center_x, center_y = get_center(positions)

    # for dots, color in zip(positions, ['ro', 'ko', 'go']):
    #     for index, (x, y) in enumerate(dots):
    #         pyplot.plot(x, y, color)
    #         ax.annotate(index, xy=(x, y), xytext=(-5, 5), ha='right', textcoords='offset points')

    pyplot.plot(center_x, center_y, 'ro')
    for index, (x, y) in enumerate(positions):
        pyplot.plot(x, y, 'ko')
        ax.annotate(index, xy=(x, y), xytext=(-5, 5), ha='right', textcoords='offset points')

    scale_gcf(0.75)
    pyplot.show()


def draw_time(XX, positions, sample, time):
    XX = XX[sample, :, time]

    from scipy.interpolate import griddata
    from numpy import linspace

    x, y = [], []
    for i in positions:
        x.append(i[0])
        y.append(i[1])
    print x
    print y
    z = XX
    xi = linspace(-90, 90, 180)
    yi = linspace(-90, 90, 180)
    zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='cubic')
    plt.contourf(xi, yi, zi, 30, cmap=plt.cm.jet)
    #plt.scatter(x, y, marker='o', c='b', s=15)
    plt.xlim(min(x) - 5, max(x) + 5)
    plt.ylim(min(y) - 5, max(y) + 5)
    plt.show()


def read_XX(channel_type):
    data = loadmat(config.train_folder + 'train_subject01.mat', squeeze_me=True)
    XX = data['X']

    XX -= XX.mean(0)
    XX = numpy.nan_to_num(XX / XX.std(0))

    #XX = numpy.nan_to_num(XX / XX.max(0))
    XX = XX[:, channel_type::3, :]

    return XX


if __name__ == '__main__':
    channel = 2
    positions = read_positions()

    draw_dots()
    #XX = read_XX(channel)
    #draw_time(XX, positions[channel], 2, 100)

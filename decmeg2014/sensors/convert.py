import os

os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import config
from scipy.io import loadmat
from itertools import izip
from glob import glob


def open_x_files(file_nr, folder):
    file_nr = str(file_nr)
    sample_folder = folder + '/' + file_nr

    return [open(sample_folder + '/' + str(i) + ".csv", 'w') for i in range(102)]


def open_y_file(file_nr, folder):
    file_nr = str(file_nr)
    sample_folder = folder + '/' + file_nr
    return open(sample_folder + '/' + 'Y.csv', 'w')


def close_files(ff):
    map(lambda f: f.close(), ff)


def read_data(files, train):
    for file_nr, glob_file in enumerate(files):
        print "\n\nLoading:", glob_file, file_nr + 1, "/", len(files)
        if train:
            xf = open_x_files(file_nr, config.train_logreg_folder)
        else:
            xf = open_x_files(file_nr, config.test_logreg_folder)

        data = loadmat(glob_file, squeeze_me=True)
        data_x = data['X']

        data_x = data_x[:, :, 125:]
        data_x -= data_x.mean(0)
        data_x = np.nan_to_num(data_x / data_x.std(0))

        for XX in data_x:
            for i in range(102):
                values1 = ",".join(map(str, XX[i]))
                values2 = ",".join(map(str, XX[i + 1]))
                values3 = ",".join(map(str, XX[i + 2]))

                xf[i].write(values1 + ',' + values2 + ',' + values3 + '\n')

        if train:
            data_y = data['y']
            yf = open_y_file(file_nr, config.train_logreg_folder)
            for y in data_y:
                yf.write(str(y) + '\n')

            yf.close()

        close_files(xf)


def prepare():
    if not os.path.isdir(config.train_logreg_folder):
        os.mkdir(config.train_logreg_folder)

    if not os.path.isdir(config.test_logreg_folder):
        os.mkdir(config.test_logreg_folder)

    for i in range(102):
        sample_folder = config.train_logreg_folder + '/' + str(i)
        if not os.path.isdir(sample_folder):
            os.mkdir(sample_folder)
        sample_folder = config.test_logreg_folder + '/' + str(i)
        if not os.path.isdir(sample_folder):
            os.mkdir(sample_folder)


if __name__ == '__main__':
    prepare()
    #read_data(glob(config.train_folder + "*.mat"), True)
    read_data(glob(config.test_folder + "*.mat"), False)
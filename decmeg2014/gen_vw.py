import numpy as np
from scipy import ndimage
import theano
import config
from scipy.io import loadmat
from glob import glob
from datetime import datetime

train_files = config.train_folder + "train_subject01.mat"


def cast_dataset((data_x, data_y)):
    cast_x = data_x.astype(theano.config.floatX)
    cast_y = data_y.astype('int32')

    return cast_x, cast_y


def get_train_files(train_files_glob, index):
    all_files = glob(train_files_glob)
    return all_files[:index] + all_files[index + 2:]


def get_test_files(train_files_glob, index):
    all_files = glob(train_files_glob)
    return [all_files[index]]


def get_valid_files(train_files_glob, index):
    all_files = glob(train_files_glob)
    return [all_files[index + 1]]


def read_test_data(train_files_glob, index):
    files = get_test_files(train_files_glob, index)
    return cast_dataset(_read_signal_data(files))


def read_valid_data(train_files_glob, index):
    files = get_valid_files(train_files_glob, index)
    return cast_dataset(_read_signal_data(files))


def read_train_data(train_files_glob, index):
    files = get_train_files(train_files_glob, index)
    return cast_dataset(_read_signal_data(files))


def _read_signal_data(files):
    XX, y = read_data(files)

    XX = XX[:, 2::3, 125:]
    XX = XX.reshape(XX.shape[0], XX.shape[1] * XX.shape[2])

    XX -= XX.mean(0)
    XX = np.nan_to_num(XX / XX.std(0))

    #baseline = XX[:, :, :125]
    #baseline_mean = baseline.mean(2)
    #XX = XX[:, :, 125:]
    #XX = XX - baseline_mean[:, :, np.newaxis]

    return XX, y


def read_data(files):
    XX, Y = None, None
    for file_nr, glob_file in enumerate(files):
        print "\n\nLoading:", glob_file, file_nr + 1, "/", len(files)
        data = loadmat(glob_file, squeeze_me=True)
        data_x = data['X']
        #data_x = data_x[:, 2::3, :]
        if XX is not None:
            XX = np.concatenate((XX, data_x))
        else:
            XX = data_x

        data_y = data['y']
        if Y is not None:
            Y = np.concatenate((Y, data_y))
        else:
            Y = data_y

        #print "\nDataset summary:"
        print "XX: ", XX.shape
        print "Y: ", Y.shape

    return (XX, Y)




if __name__ == '__main__':
    XX, y = read_data(get_test_files(config.train_folder + 'train_subject01.mat', 0))

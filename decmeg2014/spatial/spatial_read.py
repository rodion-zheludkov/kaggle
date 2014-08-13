import os
import sys
import theano
import numpy as np
import config
from scipy.io import loadmat
from sklearn.utils import shuffle


def get_x_name(folder, filename):
    return folder + '/' + filename + "_X.csv"


def get_y_name(folder, filename):
    return folder + '/' + filename + "_Y.csv"


def open_x_file(orig_filename, folder):
    return open(get_x_name(folder, orig_filename), 'w')


def open_y_file(orig_filename, folder):
    return open(get_y_name(folder, orig_filename), 'w')


def read_spatial_map():
    index_dict = {}
    with open(config.smap_file, 'r') as f:
        for line in f:
            index, x, y = map(lambda x: int(x.strip()), line.split(' '))
            index_dict[index] = (x, y)

    return index_dict


def convert_spatial(XX, smap):
    converted = np.zeros(shape=(config.width, config.height, XX.shape[1]), dtype=XX.dtype)
    for index, XX_channel in enumerate(XX):
        (x, y) = smap[index]
        converted[x, y] = XX_channel

    return converted


def test_mapping(XX_sample):
    smap = read_spatial_map()
    # Channel we want to check
    channel = 15
    # Channel index in channel type
    channel_map_index = channel / 3
    channel_type = channel % 3
    time_start = 10
    time_end = 20

    XX = XX_sample[:, time_start:time_end]

    XX_0 = convert_spatial(XX[0::3, :], smap)
    XX_1 = convert_spatial(XX[1::3, :], smap)
    XX_2 = convert_spatial(XX[2::3, :], smap)

    sample = np.array([XX_0, XX_1, XX_2])
    sample = sample.reshape(-1, )
    sample = sample.reshape((3, config.width, config.height, time_end - time_start))

    print XX[channel, :]
    x, y = smap[channel_map_index]
    print sample[channel_type, x, y, :]


def convert_data(files, train):
    smap = read_spatial_map()
    for filename in files:
        print "\nLoading: ", filename
        if train:
            full_filename = config.train_folder + filename
            xf = open_x_file(filename, config.train_spat_folder)
        else:
            full_filename = config.test_folder + filename
            xf = open_x_file(filename, config.test_spat_folder)

        data = loadmat(full_filename, squeeze_me=True)
        data_x = data['X']

        data_x -= data_x.mean(0)
        data_x = np.nan_to_num(data_x / data_x.std(0))

        # test_mapping(data_x[0])
        # return

        for XX in data_x:
            XX = XX[:, 125:(125 + config.time_slice)]
            XX_0 = convert_spatial(XX[0::3, :], smap)
            XX_1 = convert_spatial(XX[1::3, :], smap)
            XX_2 = convert_spatial(XX[2::3, :], smap)

            sample = np.array([XX_0, XX_1, XX_2]).reshape(-1)
            xf.write(",".join(map(str, sample)) + '\n')

        if train:
            data_y = data['y']
            yf = open_y_file(filename, config.train_spat_folder)
            for y in data_y:
                yf.write(str(y) + '\n')

            yf.close()

        xf.close()


def read_data(data_files):
    x_files, y_files = data_files
    XX, Y = None, None
    for x_file in x_files:
        print 'Reading..', x_file
        sys.stdout.flush()
        X_subject = np.recfromtxt(x_file, delimiter=',')
        if XX is None:
            XX = X_subject
        else:
            XX = np.concatenate((XX, X_subject))

    XX = XX.reshape((XX.shape[0], 3, config.width, config.height, config.time_slice))
    #print XX.shape

    for y_file in y_files:
        print 'Reading..', y_file
        sys.stdout.flush()
        Y_subject = np.recfromtxt(y_file)
        if Y is None:
            Y = Y_subject
        else:
            Y = np.concatenate((Y, Y_subject))

    return XX, Y


def cast_dataset((data_x, data_y)):
    cast_x = data_x.astype(theano.config.floatX)
    if data_y is not None:
        cast_y = data_y.astype('int32')
    else:
        cast_y = None

    return cast_x, cast_y


def read(files, index):
    x_files = map(lambda x: get_x_name(config.train_spat_folder, x), files)
    y_files = map(lambda x: get_y_name(config.train_spat_folder, x), files)

    #print x_files
    #print y_files

    test_files = [x_files[index]], [y_files[index]]
    valid_files = [x_files[index + 1]], [y_files[index + 1]]
    train_files = x_files[index + 2:], y_files[index + 2:]

    XX_train, Y_train = cast_dataset(read_data(train_files))
    XX_valid, Y_valid = cast_dataset(read_data(valid_files))
    XX_test, Y_test = cast_dataset(read_data(test_files))

    XX_train = XX_train.swapaxes(4, 2)
    XX_valid = XX_valid.swapaxes(4, 2)
    XX_test = XX_test.swapaxes(4, 2)

    return XX_train, Y_train, XX_valid, Y_valid, XX_test, Y_test

def read_te(files):
    x_files = map(lambda x: get_x_name(config.test_spat_folder, x), files)
    y_files = []

    XX, _ = cast_dataset(read_data((x_files, y_files)))
    XX = XX.swapaxes(4, 2)

    return XX

def read_tr_te(files, index):
    x_files = map(lambda x: get_x_name(config.train_spat_folder, x), files)
    y_files = map(lambda x: get_y_name(config.train_spat_folder, x), files)

    #print x_files
    #print y_files

    test_files = x_files[index:index + 2], y_files[index:index + 2]
    train_files = x_files[index + 2:], y_files[index + 2:]

    XX_train, Y_train = cast_dataset(read_data(train_files))
    XX_test, Y_test = cast_dataset(read_data(test_files))

    XX_train = XX_train.swapaxes(4, 2)
    XX_test = XX_test.swapaxes(4, 2)

    XX_train, Y_train = shuffle(XX_train, Y_train, random_state=42)

    return XX_train, Y_train, XX_test, Y_test


def read_all(files):
    x_files = map(lambda x: get_x_name(config.train_spat_folder, x), files)
    y_files = map(lambda x: get_y_name(config.train_spat_folder, x), files)

    XX_train, Y_train = cast_dataset(read_data((x_files, y_files)))
    XX_train = XX_train.swapaxes(4, 2)
    XX_train, Y_train = shuffle(XX_train, Y_train, random_state=42)

    XX_test, Y_test = shuffle(XX_train, Y_train, random_state=84, n_samples=500)

    return XX_train, Y_train, XX_test, Y_test

def prepare():
    if not os.path.isdir(config.train_spat_folder):
        os.mkdir(config.train_spat_folder)

    if not os.path.isdir(config.test_spat_folder):
        os.mkdir(config.test_spat_folder)


if __name__ == '__main__':
    prepare()
    #convert_data(config.trainfiles, True)
    convert_data(config.testfiles, False)

    #read(config.trainfiles[:3], 0)
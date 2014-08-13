import config
import os
import numpy as np
from scipy.io import loadmat
from glob import glob
from datetime import datetime
import utils

loc_data_dir_train = config.train_folder + "*mat" 
loc_data_dir_test = config.test_folder + "*mat" 

train = False

def create_features(XX):
    print "Applying the desired time window."
    XX = XX[:, :, 125:]

    print "Features Normalization."
    XX -= XX.mean(0)
    XX = np.nan_to_num(XX / XX.std(0))

    print "2D Reshaping: concatenating all 306 timeseries."
    XX = XX.reshape(XX.shape[0], XX.shape[1] * XX.shape[2])

    return XX

def to_vw_file(file):
    lines_wrote = 0
    start = datetime.now()

    if train:
        folder = config.train_napca_folder
    else:
        folder = config.test_napca_folder

    vw_file = folder + os.path.basename(file) + '.vw'

    with open(vw_file, "wb") as outfile:
        print "\n\nLoading:", file
        data = loadmat(folder + file, squeeze_me=True)
        XX = data['X']
        XX -= XX.mean(0)
        XX = np.nan_to_num(XX / XX.std(0))
        XX = XX.reshape(XX.shape[0], XX.shape[1] * XX.shape[2])
        print "XX:", XX.shape
        if train:
            yy = data['y']
            print "yy:", yy.shape

        print "\nAdding to Vowpal Wabbit formatted file:", vw_file
        print "\n#sample\t#total\t#wrote\ttime spend"
        for trial_nr, X in enumerate(XX):
            outline = ""
            if train:
                if yy[trial_nr] == 1:
                    label = 1
                else:
                    label = -1 #change label from 0 to -1 for binary
                outline += str(label) + " '" + str(lines_wrote) + " |f"
            else:
                file_nr = config.testfiles.index(file)
                label = 1 #dummy label for test set
                id = 17000 + (file_nr*1000) + trial_nr
                outline += str(label) + " '" + str(id) + " |f"
            for feature_nr, val in enumerate(X):
                outline += " " + str(feature_nr) + ":" + str(val)
            outfile.write( outline + "\n" )
            lines_wrote += 1
            if trial_nr % 100 == 0:
                print "%s\t%s\t%s\t%s" % (trial_nr, XX.shape[0], lines_wrote, datetime.now()-start)



def to_vw():
    if train:
        files = config.trainfiles
        num = 16
    else:
        files = config.testfiles
        num = 7

    utils.parmap(to_vw_file, files, num)

if __name__ == '__main__':
    start = datetime.now()
    to_vw()
    print "\nTotal script running time:", datetime.now()-start

import config
import os
import utils

from glob import glob
from itertools import izip
from subprocess import Popen, PIPE


#modelname = 'vw_norm_before_reshape'
modelname = 'baseline'
loc_vw = config.train_folder + modelname + '/*vw'


def make_folders():
    folders = [config.train_folder + modelname, config.model_folder + modelname,
               config.result_folder + modelname]

    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)


def get_train_cl(index, all_files):
    train_files = all_files[:index] + all_files[index + 1:]
    model_file = config.model_folder + modelname + '/' + str(index) + '.vw'
    #return ['cat'] + train_files + ['|', config.vw_path, '-c', '-k', '--passes', '60',
    #    '--loss_function', 'hinge', '--binary', '-f', model_file]
    return [config.vw_path, '-c', '-k', '--passes', '60',
            '--loss_function', 'hinge', '--binary', '-f', model_file]


def get_test_cl(index, all_files):
    test_file = all_files[index]
    model_file = config.model_folder + modelname + '/' + str(index) + '.vw'
    result_file = config.result_folder + modelname + '/' + str(index) + '.txt'
    return [config.vw_path, test_file, '-t', '-i', model_file, '-p', result_file]


def get_vw_files():
    return glob(loc_vw)


def precision(index):
    test_file = get_vw_files()[index]
    result_file = config.result_folder + modelname + '/' + str(index) + '.txt'
    i = 0
    correct = 0
    with open(result_file, 'r') as r, open(test_file, 'r') as t:
        for r_line, t_line in izip(r, t):
            r_label, _ = r_line.split(' ')
            t_label = t_line.split(' ')[0]
            r_label, t_label = int(float(r_label)), int(t_label)
            if r_label == t_label:
                correct += 1

            i += 1

    print float(correct) / i


def run_train(index):
    all_files = get_vw_files()
    train_files = all_files[:index] + all_files[index + 1:]
    cl = get_train_cl(0, all_files)
    print ' '.join(cl)

    ff = [open(x, 'r') for x in train_files]

    process = Popen(cl, stdin=PIPE, stdout=PIPE, bufsize=1)
    utils.run_log_thread(process)
    for line in utils.FilesIO(ff):
        process.stdin.write(line)

    process.stdin.close()
    print process.wait()


def run():
    #print ' '.join(get_test_cl(0, get_vw_files()))
    run_train(0)
    #precision(0)


if __name__ == '__main__':
    make_folders()
    run()
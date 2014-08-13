import gen_vw
import config
import itertools

from glob import glob
from sklearn import neighbors

train_files_glob = config.train_folder + "train_subject0[1-2].mat"


def get_train_files(index):
    all_files = glob(train_files_glob)
    return all_files[:index] + all_files[index + 1:]

def get_test_files(index):
    all_files = glob(train_files_glob)
    return [all_files[index]]

def precision(y_test, y_pred):
    correct = len(filter(lambda (x, y): x == y, itertools.izip(y_test, y_pred)))
    return float(correct) / len(y_test)


if __name__ == '__main__':
    X_train, y_train = gen_vw.read_train_data(get_train_files(0))

    clf = neighbors.KNeighborsClassifier(5, metric='euclidean')
    clf.fit(X_train, y_train)

    print 'Fitting done'

    X_test, y_test = gen_vw.read_train_data(get_test_files(0))

    print 'Testing'

    y_pred = clf.predict(X_test)

    print precision(y_test, y_pred)





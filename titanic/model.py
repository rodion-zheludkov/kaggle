import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def read_data(filename, is_train=True):
    sex_segment = {'male': '1', 'female': '0'}
    embark_segment = {'C': '0', 'S': '1', 'Q': '2', '': '1'}
    label_index = 1

    labels = []
    with open(filename, 'rb') as file_io:
        csv_file_object = csv.reader(file_io)
        csv_file_object.next()
        train_data = []
        for row in csv_file_object:
            if is_train:
                labels.append(row[label_index])
                row = row[:label_index] + row[label_index + 1:]
            row = [row[i] for i in [1, 3, 4, 5, 6, 8, 10]]
            row[1] = sex_segment[row[1]]
            row[6] = embark_segment[row[6]]

            #print row

            train_data.append(row)

    if is_train:
        labels = np.array(labels)
    train_data = np.array(train_data)

    train_data[train_data[0::, 2] == '', 2] = np.median(
        train_data[train_data[0::, 2] != '', 2].astype(np.float))

    #All missing ebmbarks just make them embark from most common place
    train_data[train_data[0::, 5] == '', 5] = np.round(
        np.mean(train_data[train_data[0::, 5] != '', 5].astype(np.float)))

    if is_train:
        return (train_data, labels)

    return train_data

def run():
    (train_data, labels) = read_data('train.csv')
    test_data = read_data('test.csv', is_train=False)

    forest = RandomForestClassifier(n_estimators=100)

    #print train_data
    #print labels
    forest = forest.fit(train_data, labels)

    print 'Predicting'
    output = forest.predict(test_data)

    open_file_object = csv.writer(open("result.csv", "wb"))
    open_file_object.writerow(["PassengerId", "Survived"])
    ids = test_data[::, 1]
    open_file_object.writerows(zip(ids, output))

if __name__ == '__main__':
    run()
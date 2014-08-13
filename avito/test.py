import numpy
from sklearn.externals import joblib
import sys
import utils


def read_test_nb(filename, transformers):
    lines = []
    ids = []

    for parts in utils.read_test(filename, True):
        # desc = cleantext.clean(parts[4], False)
        desc = parts[4]
        lines.append(desc)
        ids.append(parts[0])

    features = lines
    for transformer in transformers:
        features = transformer.transform(features)

    print 'features: ', features.shape[0]
    print 'ids: ', len(ids)

    return features, ids


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Usage " + sys.argv[0] + " file_test model_file"
        sys.exit(1)

    test_filename = sys.argv[1]
    model_filename = sys.argv[2]

    clf, transformers = joblib.load(model_filename)

    features, ids = read_test_nb(test_filename, transformers)

    predicted_scores = clf.predict_proba(features).T[1]

    with open('submission.csv', 'w') as f:
        f.write("id\n")
        for pred_score, item_id in sorted(zip(predicted_scores, ids), reverse=True):
            f.write(item_id + "\n")
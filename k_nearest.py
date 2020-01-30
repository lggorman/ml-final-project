from libsvm import read_libsvm
import numpy as np
from collections import Counter
import csv
from helpers import log_transform, trim_sparse_features

def predict(k, example, x_train, y_train):
    distances = {}
    for j in range(len(x_train)):
        distance = np.linalg.norm(example - x_train[j])
        distances[j] = distance

    sorted_distances = sorted(distances.items(), key=lambda kv: (kv[1], kv[0]))
    top_distances = sorted_distances[:k]
    labels = []
    for key, val in top_distances:
        labels.append(y_train[key])
    return Counter(labels).most_common(1)[0][0]


def predict_k_nearest(k, x_train, y_train, x_test, y_test):
    correct = 0
    for i in range(len(x_test)):
        top_label = predict(k, x_test[i], x_train, y_train)
        if top_label == y_test[i]:
            correct += 1
    return correct / x_test.shape[0]

def cross_validate(k):
    scores = []
    for i in range(1,6):
        x_folds = []
        y_folds = []
        first = True
        x, y, num_features = read_libsvm('data/data.train')
        x = np.asarray(x.todense())
        num_per_fold = len(x) // 6
        count = 0
        for j in range(1,6):
            if j != i and first:
                x_folds.append(x[count:count+num_per_fold])
                y_folds.append(y[count:count+num_per_fold])
            count += num_per_fold
        x_train = log_transform(np.concatenate(x_folds))
        y_train = np.concatenate(y_folds)

        i -= 1
        x_test = log_transform(x[i*num_per_fold:i*num_per_fold + num_per_fold])
        y_test = y[i*num_per_fold:i*num_per_fold + num_per_fold]
        result_accuracy = predict_k_nearest(k, x_train, y_train, x_test, y_test)
        scores.append(result_accuracy)
    return sum(scores) / float(len(scores))

def write_answers(k, x_test, x_train, y_train):
    ids = []
    with open('data/eval.id') as f:
        for line in f:
            ids.append(line.strip())
    with open('answers_k_nearest.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['example_id','label'])
        for i in range(x_test.shape[0]):
            prediction = predict(k, x_test[i], x_train, y_train)
            writer.writerow([ids[i], prediction])

def run_k_nearest_neighbors(write=False):
    best_accuracy = 0
    best_k = None
    print('Cross Validation')
    print('+-------+--------------------+')
    print('|   k   |  Average Accuracy  |')
    print('+-------+--------------------+')
    for k in [1, 3, 5, 10, 20]:
        result = cross_validate(k)
        print('|{:>7}'.format(str(k))+'|{:>20}|'.format(str(result)))
        if result > best_accuracy:
            best_accuracy = result
            best_k = k
    print('+-------+--------------------+')

    print('Best hyper-parameter (k):', best_k)
    print('Accuracy for best hyper-parameter:', best_accuracy)

    x_train, y_train, num_features = read_libsvm(fname='data/data.train')
    x_train = log_transform(np.asarray(x_train.todense()))
    print('Training Accuracy: N/A')

    x_test, y_test, num_features = read_libsvm(fname='data/data.test')
    x_test = log_transform(np.asarray(x_test.todense()))
    print('Test Accuracy', predict_k_nearest(1, x_train, y_train, x_test, y_test))

    if write:
        x_test, y_test, num_features = read_libsvm(fname='data/data.eval.anon')
        x_test = log_transform(np.asarray(x_test.todense()))
        write_answers(best_k, x_test, x_train, y_train)

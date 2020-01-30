from libsvm import read_libsvm
import numpy as np
from collections import Counter
import itertools
from helpers import log_transform
import csv

def predict(X, w, b):
    predicted = np.dot(w, X) + b
    if predicted >= 0:
        return 1
    else:
        return 0

def accuracy(X, y, w, b):
    correct = 0
    for i in range(X.shape[0]):
        prediction = predict(X[i], w, b)
        if prediction == y[i]:
            correct += 1
    return correct / X.shape[0]

def shuffle_arrays(X, y):
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]

def majority_baseline(y):
    most_common_label = Counter(y).most_common()[0][0]
    correct = 0
    for label in y:
        if label == most_common_label:
            correct += 1
    return correct / len(y)


def train_svm(X_train, y_train, C=10, epochs=10, lr=0.01, best_epoch=False):
    w = np.random.uniform(-0.01, 0.01, size=X_train.shape[1])  # initialize w
    b = np.random.uniform(-0.01, 0.01)
    rate = lr
    updates = 0
    best_accuracy = 0
    all_accuracies = []
    all_epochs = []
    for count in range(epochs):
        shuffled_x, shuffled_y = shuffle_arrays(X_train, y_train)
        for i in range(shuffled_x.shape[0]):

            # SVM PART
            y = 1
            if shuffled_y[i] == 0:
                y = -1

            if y * (np.dot(w, shuffled_x[i]) + b) <= 1:
                w = np.dot((1 - rate), w) + np.dot(rate * C * y, shuffled_x[i])
                b = (1-rate)*b + rate * C * y
            else:
                w = np.dot(1-rate,w)
                b = (1-rate) * b
        rate = rate / (1+count)

        if best_epoch:
            epoch_accuracy = accuracy(X_train, y_train, w, b)
            if epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                best_w = w
                best_b = b
            all_accuracies.append(epoch_accuracy)
            all_epochs.append(count+1)
    if best_epoch:
        return best_w, best_b, updates
    else:
        return w, b, updates


def cross_validate(learning_rate, C, training_function):
    scores = []
    for i in range(1,6):
        x_folds = []
        y_folds = []
        first = True
        x, y, num_features = read_libsvm('data/data.train')
        x = np.asarray(x.todense())
        num_per_fold = len(x) // 6
        count = 0

        for j in range(1, 6):
            # path = 'data/CVfolds/fold' + str(j)
            if j != i and first:
                x_folds.append(x[count:count + num_per_fold])
                y_folds.append(y[count:count + num_per_fold])
            count += num_per_fold
        x_train = np.concatenate(x_folds)
        y_train = np.concatenate(y_folds)
        w, b, updates = training_function(log_transform(x_train), y_train, C=C, epochs=20, lr=learning_rate)

        x_test = x[i * num_per_fold:i * num_per_fold + num_per_fold]
        y_test = y[i * num_per_fold:i * num_per_fold + num_per_fold]
        result_accuracy = accuracy(log_transform(x_test), y_test, w, b)
        scores.append(result_accuracy)
    return sum(scores) / float(len(scores))

def run_svm(write=False):
    best_params = None
    best_accuracy = 0
    learning_rates = [1, 10**(-1), 10**(-2), 10**(-3), 10**(-4)]
    C = [10, 1, 10 ** (-1), 10 ** (-2), 10 ** (-3), 10 ** (-4)]
    combos = list(itertools.product(learning_rates, C))
    print('Cross Validation')
    print('+---------------+-------+---------------------+')
    print('| Learning rate |   C   |   Average Accuracy  |')
    print('+---------------+-------+---------------------+')
    for combo in combos:
        result = cross_validate(combo[0], combo[1], train_svm)
        print('|{:>15}'.format(str(combo[0]))+'|{:>8}'.format(str(combo[1])) +'|{:>20}|'.format(str(result)))
        if result > best_accuracy:
            best_accuracy = result
            best_params = combo
    print('+---------------+-------+---------------------+')

    print('Best hyper-parameter (learning rate):', best_params[0])
    print('Best hyper-parameter (C):', best_params[1])
    print('Average Accuracy for best hyper-parameter:', best_accuracy)

    x_train, y_train, num_features = read_libsvm(fname='data/data.train')
    w, b, updates = train_svm(log_transform(np.asarray(x_train.todense())), y_train, epochs=20, lr=best_params[0], C=best_params[1], best_epoch=True)
    training_acc = accuracy(log_transform(np.asarray(x_train.todense())), y_train, w, b)
    print('Training Accuracy:', training_acc)

    x_test, y_test, num_features = read_libsvm(fname='data/data.test')
    test_acc = accuracy(log_transform(np.asarray(x_test.todense())), y_test, w, b)
    print('Test Accuracy:', test_acc)
    print()

    if write:
        x_test, y_test, num_features = read_libsvm(fname='data/data.eval.anon')
        write_answers(log_transform(np.asarray(x_test.todense())), y_test, w, b)

def write_answers(X, y, w, b):
    ids = []
    with open('data/eval.id') as f:
        for line in f:
            ids.append(line.strip())
    with open('answers_svm.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['example_id','label'])
        for i in range(X.shape[0]):
            prediction = predict(X[i], w, b)
            writer.writerow([ids[i], prediction])









from libsvm import read_libsvm
from itertools import product
import numpy as np
from collections import Counter
from helpers import log_transform
import csv

def predict(X, w, b):
    predicted = np.dot(w, X) + b
    if predicted >= 0:
        return 1
    else:
        return 0

def accuracy(X, y, w, b):
    errors = 0
    correct = 0
    for i in range(X.shape[0]):
        prediction = predict(X[i], w, b)
        if prediction == y[i]:
            correct += 1
        else:
            errors += 1
    return correct / (errors + correct)

def update(x, y, w, b, lr):
    if y == 0:
        y = -1
    w = w + lr * (y * x)
    b = b + lr * y
    return w, b

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


def train_simple(X_train, y_train, epochs=10, lr=0.01, best_epoch=False, decaying=False, has_margin=False, margin=None):
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
            prediction = predict(shuffled_x[i], w, b)
            if has_margin:
                if shuffled_y[i] * (np.dot(w, shuffled_x[i]) + b) < margin:
                    updates += 1
                    w, b = update(shuffled_x[i], shuffled_y[i], w, b, rate)
            else:
                if prediction != shuffled_y[i]:
                    updates += 1
                    w, b = update(shuffled_x[i], shuffled_y[i], w, b, rate)
        if decaying:
            rate = lr / (count + 1)
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

def train_decaying(X_train, y_train, epochs=10, lr=0.01, best_epoch=False):
    return train_simple(X_train, y_train, epochs, lr, best_epoch, decaying=True)

def train_averaged(X_train, y_train, epochs=10, lr=0.01, best_epoch=False):
    w = np.random.uniform(-0.01, 0.01, size=X_train.shape[1])  # initialize w
    a = np.zeros(X_train.shape[1])
    ab = 0
    b = np.random.uniform(-0.01, 0.01)
    updates = 0
    best_accuracy = 0
    all_accuracies = []
    all_epochs = []
    for count in range(epochs):
        shuffled_x, shuffled_y = shuffle_arrays(X_train, y_train)
        for i in range(shuffled_x.shape[0]):
            prediction = predict(shuffled_x[i], w, b)
            if prediction != shuffled_y[i]:
                updates += 1
                w, b = update(shuffled_x[i], shuffled_y[i], w, b, lr)
            a = a + w
            ab = ab + b
        if best_epoch:
                epoch_accuracy = accuracy(X_train, y_train, a, ab)
                if epoch_accuracy > best_accuracy:
                    best_accuracy = epoch_accuracy
                    best_w = a
                    best_b = ab
                all_accuracies.append(epoch_accuracy)
                all_epochs.append(count+1)

    if best_epoch:
        return best_w, best_b, updates
    else:
        return a, ab, updates

def train_margin(X_train, y_train, epochs=10, lr=0.01, best_epoch=False, margin=1):
    return train_simple(X_train, y_train, epochs, lr, best_epoch, has_margin=True, margin=margin)


def train_pocket(X_train, y_train, epochs=10, lr=0.01, best_epoch=False):
    w = np.random.uniform(-0.01, 0.01, size=X_train.shape[1])  # initialize w
    p = np.zeros(X_train.shape[1])
    pb = 0
    b = np.random.uniform(-0.01, 0.01)
    updates = 0
    best_accuracy = 0
    all_accuracies = []
    all_epochs = []
    current_counter = 0
    pocket_counter = 0
    for count in range(epochs):
        shuffled_x, shuffled_y = shuffle_arrays(X_train, y_train)
        for i in range(shuffled_x.shape[0]):
            current_prediction = predict(shuffled_x[i], w, b)
            if current_prediction != shuffled_y[i]:
                if current_counter > pocket_counter or pocket_counter == 0:
                    pocket_counter = current_counter
                    current_counter = 0
                    p = w
                    pb = b
                updates += 1
                w, b = update(shuffled_x[i], shuffled_y[i], w, b, lr)
            else:
                current_counter += 1
        if best_epoch:
            epoch_accuracy = accuracy(X_train, y_train, p, pb)
            if epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                best_w = p
                best_b = pb
            all_accuracies.append(epoch_accuracy)
            all_epochs.append(count+1)
    if best_epoch:
        return best_w, best_b, updates
    else:
        return p, pb, updates

def cross_validate(learning_rate, training_function, margin=None):
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
            #path = 'data/CVfolds/fold' + str(j)
            if j != i and first:
                x_folds.append(x[count:count+num_per_fold])
                y_folds.append(y[count:count+num_per_fold])
            count += num_per_fold
        x_train = np.concatenate(x_folds)
        y_train = np.concatenate(y_folds)
        if margin:
            w, b, updates = training_function(log_transform(x_train), y_train, epochs=200, lr=learning_rate, margin=margin)
        else:
            w, b, updates = training_function(log_transform(x_train), y_train, epochs=200, lr=learning_rate)
        i -= 1
        x_test = log_transform(x[i*num_per_fold:i*num_per_fold + num_per_fold])
        y_test = y[i*num_per_fold:i*num_per_fold + num_per_fold]
        result_accuracy = accuracy(x_test, y_test, w, b)
        scores.append(result_accuracy)
    return sum(scores) / float(len(scores))

np.random.seed(0)

def test_perceptron(percep_function, write=False, margin=False):
    best_rate = None
    best_accuracy = 0
    to_test = [1, 0.1, 0.01]
    if margin:
        to_test = product([1, 0.1, 0.01], repeat=2)
    print('Cross Validation')
    print('+-------------------+--------------------+')
    print('|   Learning Rate   |  Average Accuracy  |')
    print('+-------------------+--------------------+')
    for rate in to_test:
        if margin:
            lr, margin = rate
            result = cross_validate(lr, percep_function, margin)
        else:
            result = cross_validate(rate, percep_function)
        print('|{:>19}'.format(str(rate))+'|{:>20}|'.format(str(result)))

        if result > best_accuracy:
            best_accuracy = result
            best_rate = rate
    print('+-------------------+--------------------+')

    print('Best hyper-parameter (learning rate):', best_rate)
    print('Accuracy for best hyper-parameter:', best_accuracy)


    x_train, y_train, num_features = read_libsvm(fname='data/data.train')
    if margin:
        lr, margin = best_rate
        w, b, updates = percep_function(log_transform(np.asarray(x_train.todense())), y_train, epochs=200, lr=lr, best_epoch=True, margin=margin)
    else:
        w, b, updates = percep_function(log_transform(np.asarray(x_train.todense())), y_train, epochs=200, lr=best_rate, best_epoch=True)
    print('Number of updates:', updates)
    training_acc = accuracy(log_transform(np.asarray(x_train.todense())), y_train, w, b)
    print('Training Accuracy', training_acc)

    x_test, y_test, num_features = read_libsvm(fname='data/data.test')
    test_acc = accuracy(log_transform(np.asarray(x_test.todense())), y_test, w, b)
    print('Test Accuracy: ', test_acc)
    print()

    if write:
        x_test, y_test, num_features = read_libsvm(fname='data/data.eval.anon')
        write_answers(log_transform(np.asarray(x_test.todense())), y_test, w, b)

def write_answers(X, y, w, b):
    ids = []
    with open('data/eval.id') as f:
        for line in f:
            ids.append(line.strip())
    with open('answers/answers.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['example_id','label'])
        for i in range(X.shape[0]):
            prediction = predict(X[i], w, b)
            writer.writerow([ids[i], prediction])

# Just syntactic sugar to make it match the format of the other algs
def run_perceptron(write=False):
    test_perceptron(train_averaged, write)



# x_train, y_train, num_features = read_libsvm(fname='data/data.train')
# w, b, updates = train_averaged(log_transform(np.asarray(x_train.todense())), y_train, epochs=250, lr=1, best_epoch=True)
# training_acc = accuracy(log_transform(np.asarray(x_train.todense())), y_train, w, b)
# print('Training Accuracy', training_acc)
#
# x_test, y_test, num_features = read_libsvm(fname='data/data.eval.anon')
# #test_acc = accuracy(log_transform(np.asarray(x_test.todense())), y_test, w, b)
#
# write_answers(log_transform(np.asarray(x_test.todense())), y_test, w, b)
# # test_acc = accuracy(np.asarray(x_test.todense()), y_test, w, b)
# #print('Test Accuracy', test_acc)


# x_train, y_train, num_features = read_libsvm(fname='data/data.train')
# print('Majority Baseline Training: ', majority_baseline(y_train))
# x_test, y_test, num_features = read_libsvm(fname='data/data.test')
# print('Majority Baseline Test: ', majority_baseline(y_test))
# print()
# print('---------------------')
# print()
#
# test_perceptron('Simple Perceptron', train_simple)
# print()
# print('---------------------')
# print()
# test_perceptron('Decaying Simple Perceptron', train_decaying)
# print()
# print('---------------------')
# print()
# test_perceptron('Averaged Perceptron', train_averaged)
# print()
# print('---------------------')
# print()
# test_perceptron('Train Pocket', train_pocket)
# print()
# print('---------------------')
# print()
# test_perceptron('Train Margin', train_margin)
# print()
# print('---------------------')
# print()






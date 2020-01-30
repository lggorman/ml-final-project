from libsvm import read_libsvm
import numpy as np
from math import log
import csv
from helpers import trim_sparse_features

def count_labels(labels):
    counts = {}
    for label in labels:
        current_count = counts.get(label, 0)
        counts[label] = current_count + 1
    return counts

def learn_naive_bayes(x, y, smoothing):
    overall_counts = count_labels(y)

    conditional_probs = {}

    # loop over features to build distribution
    for feature_index in range(x.shape[1]):
        conditional_probs[feature_index] = {}
        counts = {}

        # count up the number of each value with each label
        for i in range(len(y)):
            val = x[i][feature_index]
            label = y[i]
            if val not in counts:
                counts[val] = {}

            count = counts[val].get(label, 0)
            counts[val][label] = count + 1

        # calculate conditional probabilities
        for val, labels in counts.items():
            conditional_probs[feature_index][val] = {}
            for label in labels:
                prob = (counts[val][label] + smoothing) / (overall_counts[label] + x.shape[1])
                conditional_probs[feature_index][val][label] = prob
    priors = {}
    for label, count in overall_counts.items():
        priors[label] = count / len(y)

    return priors, conditional_probs, overall_counts

def test_accuracy(classifier, x, y, smoothing):
    priors, conditional_probs, overall_counts = classifier
    possible_labels = list(set(y))

    correct = 0

    # loop over each example
    for i in range(len(y)):
        probs = []

        # check each label to find the one with the highest probability
        for j in range(len(possible_labels)):
            label = possible_labels[j]
            probs.append(log(priors[label]))

            # multiply the conditional probability for each value
            for feature in range(len(x[i])):
                val = x[i][feature]

                # if we haven't seen this value/label combination before
                if val not in conditional_probs[feature] or label not in conditional_probs[feature][val]:
                    probs[j] += log(smoothing / (overall_counts[label] + x.shape[1]))
                else:
                    probs[j] += log(conditional_probs[feature][val][label])

        best_index = np.array(probs).argmax()
        best_label = possible_labels[best_index]


        if best_label == y[i]:
            correct += 1

    return correct / len(y)

def cross_validate(smoothing):
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
        x_train, medians = binarize(x_train)
        classifier = learn_naive_bayes(x_train, y_train, smoothing)

        x_test = x[i * num_per_fold:i * num_per_fold + num_per_fold]
        y_test = y[i * num_per_fold:i * num_per_fold + num_per_fold]
        x_test, medians = binarize(x_test, medians)
        result_accuracy = test_accuracy(classifier, x_test, y_test, smoothing)
        scores.append(result_accuracy)
    return sum(scores) / float(len(scores))

# Welp this doesn't work great
def binarize(x, medians=None):
    if not medians:
        medians = []
        for i in range(x.shape[1]):
            medians.append(np.percentile(x[:,i], 60))

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] >= medians[j]:
                x[i][j] = 1
            else:
                x[i][j] = 0
    return x, medians

def write_answers(classifier, x, y, smoothing):
    ids = []
    with open('data/eval.id') as f:
        for line in f:
            ids.append(line.strip())
    with open('answers_naive_bayes.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['example_id','label'])
        priors, conditional_probs, overall_counts = classifier
        possible_labels = list(priors.keys())

        # loop over each example
        for i in range(len(y)):
            probs = []

            # check each label to find the one with the highest probability
            for j in range(len(possible_labels)):
                label = possible_labels[j]
                probs.append(log(priors[label]))

                # multiply the conditional probability for each value
                for feature in range(len(x[i])):
                    val = x[i][feature]

                    # if we haven't seen this value/label combination before
                    if val not in conditional_probs[feature] or label not in conditional_probs[feature][val]:
                        probs[j] += log(smoothing / (overall_counts[label] + x.shape[1]))
                    else:
                        probs[j] += log(conditional_probs[feature][val][label])
            best_index = np.array(probs).argmax()
            best_label = possible_labels[best_index]

            writer.writerow([ids[i], best_label])

def run_naive_bayes(write=False):
    best_smoothing = None
    best_accuracy = 0
    print('Cross Validation')
    print('+----------------+--------------------+')
    print('| Smoothing Term |  Average Accuracy  |')
    print('+----------------+--------------------+')
    for smoothing in [2, 1.5, 1, 0.5]:
        result = cross_validate(smoothing)
        print('|{:>16}'.format(str(smoothing))+'|{:>20}|'.format(str(result)))
        if result > best_accuracy:
            best_accuracy = result
            best_smoothing = smoothing
    print('+----------------+--------------------+')
    print('Best hyper-parameter (smoothing term):', best_smoothing)
    print('Average Accuracy for best hyper-parameter:', best_accuracy)

    x_train, y_train, num_features = read_libsvm(fname='data/data.train')
    x_train = np.asarray(x_train.todense())
    x_train, medians = binarize(x_train)
    classifier = learn_naive_bayes(x_train, y_train, best_smoothing)
    print('Training Accuracy:', test_accuracy(classifier, x_train, y_train, best_smoothing))

    x_test, y_test, num_features = read_libsvm(fname='data/data.test')
    x_test = np.asarray(x_test.todense())
    x_test, medians = binarize(x_test, medians)
    print('Test Accuracy:', test_accuracy(classifier, x_test, y_test, best_smoothing))

    if write:
        x_test, y_test, num_features = read_libsvm(fname='data/data.eval.anon')
        x_test = np.asarray(x_test.todense())
        x_test, medians = binarize(x_test, medians)
        write_answers(classifier,x_test, y_test,  best_smoothing)






from libsvm import read_libsvm
import numpy as np
from collections import Counter
import math
import csv
import itertools

class Node:
    def __init__(self, label=None):
        self.label = label
        self.children = []
        self.attribute = None
        self.val = None

def get_entropy(s, labels):
    labels_set = set(labels)
    overall_entropy = 0
    for l in labels_set:
        if isinstance(labels, list):
            prop_with_label = labels.count(l) / len(s)
        else:
            prop_with_label = labels.tolist().count(l) / len(s)
        overall_entropy -= (prop_with_label * math.log2(prop_with_label))
    return overall_entropy

def get_best_attribute(s, attributes, labels):
    # calculate overall entropy
    overall_entropy = get_entropy(s, labels)

    # check all attributes to see their information gain
    biggest_information_gain = 0
    best_attribute = None
    best_vals = None
    for att in attributes:
        expected_entropy = 0
        all_vals = set(s[:, att])

        for val in attributes[att]['possible_vals']:
            subset_with_val = []
            matched_labels = []
            for i in range(len(s)):
                if val[0] <= s[i][att] < val[1]:
                    subset_with_val.append(s[i])
                    matched_labels.append(labels[i])

            # get partition entropy for one value of an attribute
            partition_entropy = get_entropy(subset_with_val, matched_labels)

            # sum up to get overall expected entropy
            expected_entropy += (partition_entropy * (len(subset_with_val)/len(s)))

        # pick the biggest information gain
        if (overall_entropy - expected_entropy) >= biggest_information_gain:
            biggest_information_gain = overall_entropy - expected_entropy
            best_attribute = att
    return best_attribute, biggest_information_gain

def get_most_common(labels):
    # labels = s.get_column('label')
    return Counter(labels).most_common()[0][0]

def id3(s, labels, attributes, depth=None, depth_limit=None):
    if depth and depth >= depth_limit:
        node = Node(get_most_common(labels))
        return node
    # if all nodes have the same label, return single node
    labels_set = set(labels)
    if len(labels_set) == 1:
        node = Node(list(labels)[0])
        return node
    root = Node()
    attribute = get_best_attribute(s, attributes, labels)[0]
    root.attribute = attribute

    # loop over possible vals of attribute
    # possible_vals = set(s[:, attribute])
    for val in attributes[attribute]['possible_vals']:
        subset = []
        labels_subset = []
        for i in range(len(s)):
            if float(val[0]) <= s[i][attribute] < float(val[1]):
                subset.append(s[i])
                labels_subset.append(labels[i])
        #subset = s.get_row_subset(attribute.name, val)
        if len(subset) == 0:
            node = Node(get_most_common(labels))
            node.val = val
            root.children.append(node)
        else:
            new_attributes = attributes.copy()
            del new_attributes[attribute]

            subset = np.array(subset)
            if depth is not None:
                new_node = id3(subset, labels_subset, new_attributes, depth+1, depth_limit)
            else:
                new_node = id3(subset, labels_subset, new_attributes)
            new_node.val = val
            root.children.append(new_node)
    return root


def discretize(s, divide_by):
    attributes = {}
    for col in range(s.shape[1]):
        vals = s[:, col]
        max_val = max(vals)

        chunk_size = 100 / divide_by
        percentiles = []
        total = 0
        while total < 100:
            total += chunk_size
            percentiles.append(total)

        divide_on = np.percentile(vals, percentiles)

        attributes[col] = {'possible_vals': []}
        if max_val == 0:
            attributes[col]['possible_vals'].append((0, 1))
        else:
            last = 0
            for i in range(len(divide_on)):
                attributes[col]['possible_vals'].append((last, divide_on[i]))
                last = divide_on[i]
    return attributes

def build_forest(s, labels, k, divide_by, depth=1):
    trees = []
    for i in range(k):
        # Get 100 examples
        sample_indices = np.random.choice(s.shape[0], 100)
        samples = []
        sample_labels = []
        for j in sample_indices:
            samples.append(s[j])
            sample_labels.append(labels[j])
        samples = np.array(samples)

        # Get 50 features
        attributes = discretize(samples, divide_by)
        sampled_attributes = {}

        atts = range(s.shape[1])
        atts = [att for att in atts if len(set(s[:,att])) > 1]
        #print(len(atts))
        attribute_indices = np.random.choice(atts, 50)
        for i in attribute_indices:
            sampled_attributes[i] = attributes[i]

        tree = id3(samples, sample_labels, sampled_attributes, 0, depth)
        trees.append(tree)
    return trees

def get_prediction(tree, example):
    current_node = tree

    while len(current_node.children) > 0:
        # what attribute are we branching on
        attribute = current_node.attribute

        # what is the value of that attribute for the current row
        val = example[attribute]

        # loop through children to figure out which one fits
        found = 0
        biggest = 0
        biggest_node = None
        for node in current_node.children:
            if node.val[0] <= val < node.val[1]:
                 current_node = node
                 found = 1
                 break
            if node.val[1] > biggest:
                biggest = node.val[1]
                biggest_node = node

        if found == 0:
            current_node = biggest_node

    return current_node.label

def test_forest(trees, s, labels):
    correct = 0
    for example_index in range(s.shape[0]):
        predictions = []
        for tree in trees:
            predictions.append(get_prediction(tree, s[example_index]))
        majority_label = Counter(predictions).most_common()[0][0]
        if majority_label == labels[example_index]:
            correct += 1
    return correct / s.shape[0]

def cross_validate(k, divide_by, depth):
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
        x_train = np.concatenate(x_folds)
        y_train = np.concatenate(y_folds)

        forest = build_forest(x_train, y_train, k, divide_by, depth)

        i -= 1
        x_test = x[i*num_per_fold:i*num_per_fold + num_per_fold]
        y_test = y[i*num_per_fold:i*num_per_fold + num_per_fold]
        result_accuracy = test_forest(forest, x_test, y_test)
        scores.append(result_accuracy)
    return sum(scores) / float(len(scores))

def run_random_forest(write=False):
    chunk_size = 4
    k = [50, 100, 150, 200]
    depth = [1, 2, 3, 4]
    combos = list(itertools.product(k, depth))
    best_accuracy = 0

    print('Cross Validation')
    print('+-------+---------+--------------------+')
    print('|   k   |  depth  |  Average Accuracy  |')
    print('+-------+---------+--------------------+')
    for k, depth in combos:
        accuracy = cross_validate(k=k, divide_by=chunk_size, depth=1)
        print('|{:>7}'.format(str(k))+'|{:>9}'.format(str(depth))+'|{:>20}|'.format(str(accuracy)))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
            best_depth = depth
    print('+-------+---------+--------------------+')

    print('Best hyper-parameter (k):', best_k)
    print('Average Accuracy for best hyper-parameter:', best_accuracy)

    x_train, y_train, num_features = read_libsvm(fname='data/data.train')
    x_train = np.asarray(x_train.todense())
    forest = build_forest(x_train, y_train, k=best_k, divide_by=4, depth=best_depth)
    print('Training Accuracy:', test_forest(forest, x_train, y_train))

    x_test, y_test, num_features = read_libsvm(fname='data/data.test')
    x_test = np.asarray(x_test.todense())
    print('Test Accuracy:', test_forest(forest, x_test, y_test))

    if write:
        x_test, y_test, num_features = read_libsvm(fname='data/data.eval.anon')
        x_test = np.asarray(x_test.todense())
        write_forest_answers(forest, x_test)

def write_forest_answers(forest, s):
    ids = []
    with open('data/eval.id') as f:
        for line in f:
            ids.append(line.strip())
    with open('answers_forest.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['example_id', 'label'])
        for i in range(s.shape[0]):
            predictions = []
            for tree in forest:
                predictions.append(get_prediction(tree, s[i]))
            majority_label = Counter(predictions).most_common()[0][0]
            writer.writerow([ids[i], majority_label])



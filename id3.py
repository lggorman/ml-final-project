import numpy as np
from math import log2
from collections import Counter
from libsvm import read_libsvm
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
        overall_entropy -= (prop_with_label * log2(prop_with_label))
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
        # don't pick an attribute where all values are 0
        # if len(all_vals) == 1:
        #     continue

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


def test_tree(tree, s, labels, write=False):
    errors = 0
    correct = 0
    i = 0
    ids = []
    with open('data/eval.id') as f:
        for line in f:
            ids.append(line.strip())
    with open('answers_id3.csv', 'w') as csvfile:
        if write == True:
            writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['example_id', 'label'])
        for row in s:
            current_node = tree

            while len(current_node.children) > 0:
                # what attribute are we branching on
                attribute = current_node.attribute
                #att_index = data_obj.attributes[attribute].index

                # what is the value of that attribute for the current row
                val = row[attribute]

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
                # FIX THIS
                if found == 0:
                    current_node = biggest_node
            if write:
                writer.writerow([ids[i], current_node.label])
            if current_node.label == labels[i]:
                correct += 1
            else:
                errors += 1
            i += 1

    return correct / float((correct+errors)) * 100

def cross_validate(divide_by=4, depth=10):
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
        attributes = discretize(x_train, divide_by)
        tree = id3(x_train, y_train, attributes, 0, depth)
        i -= 1
        x_test = x[i*num_per_fold:i*num_per_fold + num_per_fold]
        y_test = y[i*num_per_fold:i*num_per_fold + num_per_fold]
        #x_test = np.asarray(x_test.todense())
        scores.append(test_tree(tree, x_test, y_test))
    return sum(scores) / float(len(scores))

def run_id3(write=False):
    chunk_sizes = [3, 4, 5, 8]
    depth = [8, 10, 12]
    combos = list(itertools.product(chunk_sizes, depth))
    best_accuracy = 0
    best_chunk_size = 4
    best_depth = 12

    # print('Cross Validation')
    # print('+-------+-------------+---------------------+')
    # print('| Depth |  Intervals  |   Average Accuracy  |')
    # print('+-------+-------------+---------------------+')
    # for chunk_size, depth in combos:
    #     accuracy = cross_validate(chunk_size, depth)
    #     print('|{:>7}'.format(str(depth)) + '|{:>13}'.format(chunk_size) + '|{:>20}|'.format(str(accuracy)))
    #     if accuracy > best_accuracy:
    #         best_accuracy = accuracy
    #         best_chunk_size = chunk_size
    #         best_depth = depth
    # print('+-------+-------------+---------------------+')
    #
    # print('Best hyper-parameter (intervals for discretization):', best_chunk_size)
    # print('Best hyper-parameter (depth):', depth)
    # print('Accuracy for best hyper-parameter:', best_accuracy)

    x_train, y_train, num_features = read_libsvm(fname='data/data.train')
    x_train = np.asarray(x_train.todense())
    attributes = discretize(x_train, best_chunk_size)
    tree = id3(x_train, y_train, attributes, 0, best_depth)
    print('Training Accuracy:', test_tree(tree, x_train, y_train, write=False))

    x_test, y_test, num_features = read_libsvm(fname='data/data.test')
    x_test = np.asarray(x_test.todense())
    print('Test Accuracy', test_tree(tree, x_test, y_test, write=False))

    if write:
        x_test, y_test, num_features = read_libsvm(fname='data/data.eval.anon')
        x_test = np.asarray(x_test.todense())
        test_tree(tree, x_test, y_test, write=True)



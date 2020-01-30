from k_nearest import run_k_nearest_neighbors
from id3 import run_id3
from naive_bayes import run_naive_bayes
from random_forest import run_random_forest
from svm import run_svm
from perceptron import run_perceptron

# ALL the experiments!

print('PERCEPTRON')
print('-------------------------------------------')
run_perceptron()
print()

print('SVM')
print('-------------------------------------------')
run_svm()
print()

print('K-NEAREST NEIGHBORS')
print('-------------------------------------------')
run_k_nearest_neighbors()
print()

print('NAIVE BAYES')
print('-------------------------------------------')
run_naive_bayes()
print()

print('RANDOM FOREST')
print('-------------------------------------------')
run_random_forest()
print()

print('ID3')
print('-------------------------------------------')
run_id3()
print()





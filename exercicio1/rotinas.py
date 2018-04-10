from knn import Knn
import operator
from functools import reduce
import numpy as np
from visualization import Visualization


class Rotinas:
    def __init__(self):
        self.k_values = [1, 2, 3, 5, 7, 9, 11, 13, 15]
        self.mean_accuracy_list = []
        self.train_set = []
        self.test_set = []

    def mean_accuracy(self, ac):
        tam = len(ac)
        if tam > 0:
            ret = reduce(operator.add, ac)
            self.mean_accuracy_list.append(ret / tam)
        else:
            self.mean_accuracy_list.append(0)

    def set_train_test_set(self, generator):
        for train, test in generator:
            self.train_set.append(train)
            self.test_set.append(test)

    def run(self, alg, bean_list, filename = "img.png", weighted = False):
        accuracy = []
        #self.set_train_test_set(generator,bean_list)
        for k in self.k_values:
            for train in self.train_set:
                # Train
                alg.train([bean_list[a] for a in train])
            for test in self.test_set:
                # Test
                result = alg.teste([bean_list[a] for a in test], k = k)

                # Result
                accuracy.append(Knn.accuracy(result))
                print("acuracy knn {0} with k {1}".format(accuracy[-1], k))

            self.mean_accuracy(accuracy)

        # Plot the graphic
        Visualization.hit_rate_per_k(self.mean_accuracy_list, self.k_values, filename, weighted)

    def clear_mean_accuracy(self):
        self.mean_accuracy_list = []

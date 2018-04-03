from projeto.knn import Knn
import operator
from functools import reduce

class Rotinas:
    def __init__(self):
        self.k_values = [1, 2, 3, 5, 7, 9, 11, 13, 15]
        self.mean_acuracy_list = []

    def mean_accuracy(self, ac):
        tam = len(ac)
        if tam > 0:
            ret = reduce(operator.add, ac)
            self.mean_acuracy_list.append(ret/tam)

    def run(self, alg, generator, bean_list):
        accuracy = []
        for k in self.k_values:
            for train, test in generator:
                train_set = [bean_list[i] for i in train]
                test_set = [bean_list[i] for i in test]
                # Train
                alg.train(test_set)

                # Test
                result = alg.teste(train_set, k = k)

                # Result
                accuracy.append(Knn.accuracy(result))
                print("acuracy simple knn {0} with k {1}".format(accuracy[-1], k))

            self.mean_accuracy(accuracy)


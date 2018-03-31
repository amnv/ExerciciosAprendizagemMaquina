import numpy as np
import math
from collections import Counter

class Knn:

    def __init__(self):
        self.items = {}

    def euclidian_distancy(self, vTested, vBase):
        v1, v2 = np.asarray(vTested, dtype=np.float32), np.array(vBase, dtype=np.float32)
        diff = v1 - v2
        quad_dist = np.dot(diff, diff)
        return math.sqrt(quad_dist)

    def train(self, data):
        for i in data:
            self.items[i.feature] = i.label

    def find_nearst(self, item, k, dist_func):
        nearst = sorted([(x, dist_func(self, item, x)) for x in self.items], key=lambda tup: tup[1])
        return nearst[:k]

    def teste_val(self, item, k, dist_func):
        nearsts = self.find_nearst(item, k, dist_func)
        class_list = map(lambda x: self.items[x[0]], nearsts)
        return Counter(class_list).most_common(1)[0][0]

    def teste(self, data, dist_func = euclidian_distancy, k=1):
        for item in data:
            item.classified = self.teste_val(item.feature, k, dist_func)
        return data

    def accuracy(self, data):
        pass

    def precision(self, data):
        pass

    def recall(self, data):
        pass
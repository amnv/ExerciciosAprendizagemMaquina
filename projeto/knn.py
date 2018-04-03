from collections import Counter
from distancies import Distancy
#from scipy.spatial import distance

class Knn:

    def __init__(self):
        self.items = {}
       # self.euclidian_distancy = distance.euclidean

    def train(self, data):
        for i in data:
            self.items[i.feature] = i.label

    def find_nearst(self, item, k, dist_func):
        nearst = sorted([(x, dist_func(item, x)) for x in self.items], key=lambda tup: tup[1])
        return nearst[:k]

    def teste_val(self, item, k, dist_func):
        nearsts = self.find_nearst(item, k, dist_func)

        # Classes of k closest items
        class_list = map(lambda x: self.items[x[0]], nearsts)

        # Most frequent class
        return Counter(class_list).most_common(1)[0][0]

    def teste(self, data, dist_func = Distancy.euclidian_distancy, k=1):
        for item in data:
            item.classified = self.teste_val(item.feature, k, dist_func)
        return data

    def accuracy(self, data):
        count = 0

        # Checking amount of correct classification
        for i in data:
            if i.classified == i.label:
                count += 1

        return count/len(data)

    def precision(self, data):
        pass

    def recall(self, data):
        pass
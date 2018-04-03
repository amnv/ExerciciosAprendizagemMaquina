from distancies import Distancy

from knn import Knn


class Knn_weight(Knn):

    def __init__(self, class_names):
        super(Knn_weight, self).__init__()
        self.class_names = class_names

    def teste_val(self, item, k, dist_func):
        nearsts = super().find_nearst(item, k, dist_func)

        class_list = {}
        for class_name in self.class_names:
            #aux = filter(lambda x: self.items[x[0]] == class_name, nearsts)

            aux = [a for a in nearsts if class_name in self.items[a[0]]]

            for j in aux:
                class_list[class_name] = Distancy.weight_distancy(item, j[0])

        return max(class_list, key=class_list.get)
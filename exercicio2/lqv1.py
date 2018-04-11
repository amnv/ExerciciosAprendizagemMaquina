import random
import numpy as np

from sklearn.neighbors import KNeighborsClassifier

"""Works using ndArray from numpy
    Avoid using regular python vector"""
class Lqv1:

    @staticmethod
    def get_best_class(prototype_set, prototype_label, pattern):
        """Prototype must be an ndarray"""
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(prototype_set, prototype_label)
        return knn.predict(pattern), knn.kneighbors(pattern)

    @staticmethod
    def test_prototype(prototype_set, prototype_label, test_set):
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(prototype_set, prototype_label)
        return knn.score(test_set)

    @staticmethod
    def build_prototypes(qnt_features, size, labels_list):
        labels = []
        feature_ndarray = np.array([])
        for i in range(labels_list):
            feature = []
            for j in range(qnt_features):
                feature.append(random.random())
            labels.append(random.choice(labels_list))
            np.concatenate((feature_ndarray, feature))

        return feature_ndarray, np.array(labels)

    @staticmethod
    def adjust_learning_rate(learning_rate):
        return 0

    @staticmethod
    def train(train_set, prototype_set, prototype_label, iterations=200, learning_rate=0.001, threshold=0.10):
        err = 1
        while iterations > 1 or err > threshold:
            for item in train_set:
                c, prototype_position = Lqv1.get_best_class(prototype_set, prototype_label, item)

                if c == item.getClass():
                    # Approximate prototype
                    # ei + α(t) × [x − ei]
                    prototype_set[prototype_position] += learning_rate * (c - prototype_set[prototype_position])
                else:
                    # Approximate prototype
                    # ei - α(t) × [x − ei]
                    prototype_set[prototype_position] -= learning_rate * (c - prototype_set[prototype_position])

            # Exponentially decreasing
            learning_rate = Lqv1.adjust_learning_rate(learning_rate)

            # Reduce iterations
            iterations -= 1

            # Testing prototype efficiency
            err = 1 - Lqv1.test_prototype(prototype_set, prototype_label, train_set)


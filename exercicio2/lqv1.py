import random
import numpy as np

from sklearn.neighbors import KNeighborsClassifier

"""Works using ndArray from numpy
    Avoid using regular python vector"""
class Lqv1:
    @staticmethod
    def get_best_class(prototype_set, prototype_label, pattern):
        """Prototype must be an ndarray"""
        print("find best class")
        knn = KNeighborsClassifier(n_neighbors=1)
        p_label = prototype_label.reshape(-1, 1)
        knn.fit(prototype_set, prototype_label.ravel())
        pat = pattern.values.reshape(1, -1)
        return knn.predict(pat), knn.kneighbors(pat, return_distance=False)[0][0]

    @staticmethod
    def test_prototype(prototype_set, prototype_label, test_set, test_label, k=1):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(prototype_set, prototype_label)
        return knn.score(test_set, test_label)

    @staticmethod
    def build_prototypes(qnt_features, size, labels_list):
        labels = []
        feature_ndarray = np.array([], np.float64)
        for i in range(size):
            feature = []
            for j in range(qnt_features):
                feature.append(random.random())
            labels.append(random.choice(labels_list))
            feature_ndarray = np.concatenate((feature_ndarray, np.array(feature, np.float64)))

        return feature_ndarray.reshape((size, qnt_features)), np.array(labels)

    @staticmethod
    def adjust_learning_rate(learning_rate, is_correct_classified):
        s = 1 if is_correct_classified else -1
        d = (1 + s * learning_rate)
        if d == 0:
            return 0
        return learning_rate/d

    @staticmethod
    def train(train_set, train_label, prototype_set, prototype_label, iterations=200, learning_rate=0.001, threshold=0.10):
        err = 1
        while iterations > 1 and err > threshold:
            for i in range(train_set.shape[0]):
                c, prototype_position = Lqv1.get_best_class(prototype_set, prototype_label, train_set.loc[i])
                is_correct_classified = c == train_label.loc[i].item()
                print("is_correct_classified " , is_correct_classified)
                if is_correct_classified:
                    # Approximate prototype
                    # ei + α(t) × [x − ei]
                    prototype_set[prototype_position] += learning_rate * (c - prototype_set[prototype_position])
                else:
                    # Approximate prototype
                    # ei - α(t) × [x − ei]
                    prototype_set[prototype_position] -= learning_rate * (c - prototype_set[prototype_position])

            # Exponentially decreasing
            learning_rate = Lqv1.adjust_learning_rate(learning_rate, is_correct_classified)
            print("learning_rate " + str(learning_rate))
            # Reduce iterations
            iterations -= 1

            # Testing prototype efficiency
            err = 1 - Lqv1.test_prototype(prototype_set, prototype_label, train_set, train_label)
            print("err " + str(err))
        return prototype_set

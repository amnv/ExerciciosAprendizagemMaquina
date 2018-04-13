import numpy
from sklearn.neighbors import KNeighborsClassifier

from lqv1 import Lqv1

class Lqv2_1:
    def get_best_class(self, prototype_set, prototype_label, pattern):
        """Prototype must be an ndarray"""
        print("find best class")
        knn = KNeighborsClassifier(n_neighbors=2)
        p_label = prototype_label.reshape(-1, 1)
        knn.fit(prototype_set, prototype_label.ravel())
        pat = pattern.values.reshape(1, -1)
        return knn.predict(pat), knn.kneighbors(pat, return_distance=False)[0]

    def window(self, x, d1, d2, width = 10):
        dist1 = numpy.linalg.norm(d1 - x)
        dist2 = numpy.linalg.norm(d2 - x)
        m = min(dist1/dist2, dist2/dist1)
        s = 1 - width/ 1 + width
        return m > s

    def train(self, texto, iterations=20, learning_rate=0.001, threshold=0.10):
        classes = texto.loc[:, texto.columns[:texto.columns.size - 1]]
        labels = texto.iloc[:, [texto.columns.size - 1]]

        tam = int(classes.shape[0] / 2)

        prototype_set, prototype_label = Lqv1.build_prototypes(texto.columns.size - 1, 10, texto.defects.unique())
        train = Lqv1.train(classes[:tam], labels[:tam], prototype_set, prototype_label, 20, 1, 0.010)
        err = 1
        while iterations > 1 and err > threshold:
            for i in range(train.shape[0]):
                x = train.loc[i]
                c, prototype_position = self.get_best_class(prototype_set, prototype_label, x)
                if self.window(x, prototype_set[prototype_position[0]], prototype_set[prototype_position[1]]):
                    # if they have same label
                    if labels.loc[prototype_position[0]].item() != labels.loc[prototype_position[1]].item():
                        if c == labels.loc[prototype_position[0]].item():
                            # Approximate prototype
                            # ei = ei + α(t) × [x − ei]
                            # 9: ej = ej − α(t) × [x − ei]
                            prototype_set[prototype_position[0]] += learning_rate * (c - prototype_set[prototype_position[0]])
                            prototype_set[prototype_position[1]] -= learning_rate * (c - prototype_set[prototype_position[1]])
                        elif c == labels.loc[prototype_position[1]].item():
                            # Approximate prototype
                            # ei - α(t) × [x − ei]
                            prototype_set[prototype_position[0]] -= learning_rate * (c - prototype_set[prototype_position[0]])
                            prototype_set[prototype_position[1]] += learning_rate * (c - prototype_set[prototype_position[1]])
                        else:
                            prototype_set[prototype_position[0]] -= learning_rate * (c - prototype_set[prototype_position[0]])
                            prototype_set[prototype_position[1]] -= learning_rate * (c - prototype_set[prototype_position[1]])

            # Exponentially decreasing
            learning_rate = Lqv1.adjust_learning_rate(learning_rate, c == labels[i])
            print("learning_rate " + str(learning_rate))
            # Reduce iterations
            iterations -= 1
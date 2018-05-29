from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

class Knn:

    def run(self, data, label_data, neighbors, k):
        # Training model
        model = KNeighborsClassifier(n_neighbors=neighbors)

        # Slip data mode
        cv = StratifiedKFold(n_splits=k)

        # training data
        tam = data.shape[1] - 1
        feature_new_data = data.iloc[:, :tam]

        # Executing the train and test process
        scores = cross_val_score(model, feature_new_data, label_data, cv=cv)

        print("Model's acuracy")
        print(scores)

        return scores

    def score(self, data, label_data, neighbors=1, k=5):
        return self.run(data, label_data, neighbors, k)

    def avg(self, data, label_data, neighbors=1, k=5):
        scores = self.run(data, label_data, neighbors, k)
        res = 0
        for i in scores:
            res += i
        return res / k

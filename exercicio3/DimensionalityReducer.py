import numpy.linalg as la
import numpy as np
import pandas as pd


class DimensionalityReducer:
    def pca(self, data, x_scaled, k=10):
        cov = data.cov()
        eigenvalues, eigenvectors = la.eig(cov)

        # Get higher eigenvectors
        eigenvalues_indices = np.argsort(eigenvalues)[::-1]
        a = eigenvalues_indices[:k]
        b = eigenvalues_indices[-1:]
        eigenvalues_indices = np.concatenate([a, b])

        # New eigenvector matrix
        new_data = np.matrix(eigenvectors[eigenvalues_indices]) * np.matrix(x_scaled.T)

        # Transformed dataframe
        return pd.DataFrame(new_data.T, columns=data.columns[eigenvalues_indices])

    def lda(self, data, dimensions, x_scaled, k=10):
        classes = data.iloc[:, -1].unique()

        # Dispersão entre classes
        sb = 0

        # Dispersão intra classes
        sw = np.zeros((dimensions, dimensions))

        for l in classes:
            class_patterns = data[data.iloc[:, -1] == l].iloc[:, :dimensions]
            sb = np.add(class_patterns.cov(), sb)

            # Within class covariance
            sw = np.add((class_patterns.iloc[:, :dimensions].cov()), sw)

            # Finding eigenvalues and eigenvectors
            a = np.dot(sw.T, sb)
            eigenvalues, eigenvectors = la.eig(a)

            eigenvalues_indices = np.argsort(eigenvalues)[::-1]
            a = eigenvalues_indices[:k]
            b = eigenvalues_indices[-1:]
            eigenvalues_indices = np.concatenate([a, b])

            new_data = np.matrix(eigenvectors[eigenvalues_indices]) * np.matrix(x_scaled.T)
            return pd.DataFrame(new_data.T, columns=data.columns[eigenvalues_indices])


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from DimensionalityReducer import DimensionalityReducer
from Knn import Knn
from visualization import Visualization


def score_result(reducer_function, data, x_scaled, ini, dimensions, label_data, title):
    knn = Knn()
    score = []
    valor_k = range(ini, dimensions)
    for k in valor_k:
        new_data = reducer_function(data, x_scaled, k)
        score.append(knn.avg(new_data, label_data))

    Visualization.hit_rate_per_k(valor_k, score, title)


def main():
    # base e dados da base
    columns = ["loc", "v(g)", "ev(g)","iv(g)", "n", "v", "l", "d","i", "e", "b", "t", "lOCode",\
                                          "loComment", "lOBlank", "locCodeAndComment", "uniq_Op", "uniq_Opnd", \
                                          "total_Op", "total_Opnd", "branchCount", "defects"]
    df = pd.read_csv("../base3.csv", names=columns)
    size = df.shape[1]

    # Samples for training
    tam = (size - 1)
    feature_data = df.iloc[:, :tam]
    label_data = df.iloc[:, -1]

    x = feature_data.values #returns a numpy array
    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    data = pd.DataFrame(x_scaled, columns=df.columns[:tam])


    # build charts
    score_result(DimensionalityReducer.pca, data, x_scaled, 1, tam, label_data, "pca_visualization")
    data = pd.concat([data, label_data], axis=1)
    score_result(DimensionalityReducer.lda, data, x_scaled, label_data.unique().size, tam, label_data, "lda_visualization")


main()
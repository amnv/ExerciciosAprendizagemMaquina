import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from lqv1 import Lqv1
from lqv2_1 import Lqv2_1

#base e dados da base
df = pd.read_csv("../base2.csv", names=["loc", "v(g)", "ev(g)",\
                                      "iv(g)", "n", "v", "l", "d","i", "e", "b", "t", "lOCode",\
                                      "loComment", "lOBlank", "locCodeAndComment", "uniq_Op", "uniq_Opnd", \
                                      "total_Op", "total_Opnd", "branchCount", "defects"])


x = df.values #returns a numpy array
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
texto = pd.DataFrame(x_scaled, columns=df.columns)


features = texto.loc[:, texto.columns[:texto.columns.size - 1]]
labels = texto.iloc[:, [texto.columns.size - 1]]

skf = StratifiedKFold(n_splits=5)
categories = texto.defects.unique()
generator = skf.split(features, labels)

train_set = []
labels_set = []

print(list(generator))

train_set = np.asarray(train_set)
labels_set = np.asarray(labels_set)

tam = int(texto.shape[0] * 0.8)
#tam = int(texto.shape[0]/2)

#
# prototype_set, prototype_label = Lqv1.build_prototypes(texto.columns.size - 1, 50, categories)
# train = Lqv1.train(features[:tam], labels[:tam], prototype_set, prototype_label, 50, 0.3, 0.001)
# score = Lqv1.test_prototype(train, prototype_label, features[tam:], labels[tam:], 1)


alg = Lqv2_1()
trained_prototype, prototype_label = alg.train(features, labels, tam, texto.columns.size - 1, texto.defects.unique())
score = Lqv1.test_prototype(trained_prototype, prototype_label, features[tam:], labels[tam:], 2)

print("score " + str(score))


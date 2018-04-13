import pandas as pd
from lqv1 import Lqv1

#base e dados da base
texto = pd.read_csv("../base2.csv", names=["loc", "v(g)", "ev(g)",\
                                      "iv(g)", "n", "v", "l", "d","i", "e", "b", "t", "lOCode",\
                                      "loComment", "lOBlank", "locCodeAndComment", "uniq_Op", "uniq_Opnd", \
                                      "total_Op", "total_Opnd", "branchCount", "defects"])

classes = texto.loc[:, texto.columns[:texto.columns.size - 1]]
lables = texto.iloc[:, [texto.columns.size - 1]]

tam = int(classes.shape[0]/2)

prototype_set, prototype_label = Lqv1.build_prototypes(texto.columns.size - 1, 10, texto.defects.unique())
train = Lqv1.train(classes[:tam], lables[:tam], prototype_set, prototype_label, 20, 1, 0.0010)
score = Lqv1.test_prototype(train, prototype_label, classes[tam:], lables[tam:], 1)

print("score " + str(score))
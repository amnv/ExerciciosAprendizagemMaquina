import pandas as pd
from lqv1 import Lqv1

#base e dados da base
texto = pd.read_csv("base1.csv", names=["loc", "v(g)", "ev(g)",\
                                      "iv(g)", "n", "v", "l", "d","i", "e", "b", "t", "lOCode",\
                                      "loComment", "lOBlank", "locCodeAndComment", "uniq_Op", "uniq_Opnd", \
                                      "total_Op", "total_Opnd", "branchCount", "defects"])

classes = texto[:len(texto.columns) - 1]
lables = texto[-1]


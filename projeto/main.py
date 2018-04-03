from bean_object import Bean_object
from knn import Knn
from knn_weight import Knn_weight
from visualization import  Visualization

# Preprocessing
from projeto.distancies import Distancy
from projeto.rotinas import Rotinas

texto = open("../base2.csv")
beans_list = Bean_object.build_bean(texto)

accuracy = []
accuracy_list_per_k = []
data, classification = Bean_object.build_bean_splited(beans_list)
generator = Knn.split_data(data, classification)

# Simple knn
print("Knn simples")
alg = Knn()
rot = Rotinas()
rot.run(alg, generator, beans_list)


print("\n")
# Weighted knn
print("knn com peso")
alg = Knn_weight(Distancy.get_class(beans_list))
rot = Rotinas()
rot.run(alg, generator, beans_list)
from bean_object import Bean_object
from knn import Knn
from knn_weight import Knn_weight
from visualization import  Visualization

# Preprocessing
from projeto.distancies import Distancy
from projeto.rotinas import Rotinas

texto = open("../base1.csv")
beans_list = Bean_object.build_bean(texto)

accuracy = []
accuracy_list_per_k = []
data, classification = Bean_object.build_bean_splited(beans_list)
generator = Knn.split_data(data, classification)

# Simple knn
print("Knn simples")
alg = Knn()
rot = Rotinas()
rot.set_train_test_set(generator)
rot.run(alg, beans_list, "../knn_simples.png")


print("\n")
# Weighted knn
print("knn com peso")
alg = Knn_weight(Distancy.get_class(beans_list))
rot.clear_mean_accuracy()
rot.run(alg, beans_list, "../knn_peso.png", True)

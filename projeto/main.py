from bean_object import Bean_object
from knn import Knn
from knn_weight import Knn_weight
from visualization import  Visualization

# Preprocessing
texto = open("../base1.csv")
beans_list = Bean_object.build_bean(texto, 10)
k_values = [1, 2, 3, 5, 7, 9, 11, 13, 15]
test_set = train_set = beans_list
accuracy = []

# Simple knn
print("Knn simples")
alg = Knn()

for k in k_values:
        #train
        alg.train(test_set)

        #teste
        result = alg.teste(train_set)

        #result
        accuracy.append(Knn.accuracy(result))
        print("acuracy simple knn {0} with k {1}".format(accuracy[-1], k))

print("\n")
# Weighted knn
print("knn com peso")
accuracy = []
for k in k_values:
        a = Knn_weight(["true", "false"])
        a.train(beans_list)
        result = a.teste(beans_list)

        #avaliation result
        accuracy.append(Knn.accuracy(result))
        print("acuracy weighted knn {0} with k {1}".format(Knn_weight.accuracy(result), k))
       


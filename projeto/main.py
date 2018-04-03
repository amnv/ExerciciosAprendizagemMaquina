from projeto.bean_object import Bean_object
from projeto.knn import Knn
from projeto.knn_weight import Knn_weight
from projeto.visualization import  Visualization

texto = open("../base1.csv")
beans_list = []


print("knn com peso")
a = Knn_weight(["true", "false"])
a.train(beans_list)
result = a.teste(beans_list)

for i in result:
        print("classificado: ", i.classified, " esperado: ", i.label)

print("\------------------------------------------/")
print("Knn simples")
alg = Knn()
#train
alg.train(beans_list)
#teste

result = alg.teste(beans_list)

#result
for i in result:
        print("classificado: ", i.classified, " esperado: ", i.label)


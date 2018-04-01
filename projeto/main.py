from bean_object import Bean_object
from knn import Knn
from knn_weight import Knn_weight

texto = open("../base1.csv")
beans_list = []
count = 0
for line in texto:
    aux = Bean_object()
    aux.build_features(line)
    aux.build_label(line)
    beans_list.append(aux)
    if count >= 100:
        break
    count += 1

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


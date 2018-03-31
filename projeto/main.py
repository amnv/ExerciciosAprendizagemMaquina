from bean_object import Bean_object
from knn import Knn

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

alg = Knn()
#train
alg.train(beans_list)
print("vou passar")
#teste

result = alg.teste(beans_list)
print("passei")

#result
for i in result:
        print("classificado: ", i.classified, "esperado: ", i.label)

print("asd")
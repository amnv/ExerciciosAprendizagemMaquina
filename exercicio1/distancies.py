import math
import numpy as np

class Distancy:
    @staticmethod
    def euclidian_distancy(vTested, vBase):
        v1, v2 = np.asarray(vTested, dtype=np.float32), np.array(vBase, dtype=np.float32)
        diff = v1 - v2
        quad_dist = np.dot(diff, diff)
        return math.sqrt(quad_dist)

    @staticmethod
    def weight_distancy(item, base_items):
        sum = 0
        for i in base_items:
            dist = Distancy.euclidian_distancy(item, i)
            if dist > 0:
                sum += 1/dist
        return sum

    @staticmethod
    def build_vdm_category_count(data, category_values):
        count = 0
        for i in data:
            if i == category_values:
                count += 1

        return count

    @staticmethod
    def build_categories_list(data, data_size):
        categories = [[]] * data_size
        for item in data:
            for features, j in item.features, range(data_size):
                categories[j].append(features)

        return categories

    @staticmethod
    def get_categories(data):
        cat = {}
        for i in data:
            cat[i] = 0

        return cat.keys()

    @staticmethod
    def get_class(data):
        classes = {}
        for i in data:
            classes[i.label] = 0
        return classes.keys()

    @staticmethod
    def get_niac(data, cat_size, cat_values_list):
        class_list = Distancy.get_class(data)
        niac = {}

        # Init niac
        for i in class_list:
            niac[i] = [{}]*cat_size
            for j in range(cat_size):
                for cat in cat_values_list:
                    niac[i][j][cat] = 0

        for i in data:
            for j in range(len(i.feature)):
                for value in i.feature:
                    niac[i.label][j][value] += 1

        return niac

    @staticmethod
    def build_vdm(data):
        size_data = len(data[0].features)
        nia = [[]]*size_data
        cat_list = [[]]*size_data
        mat_cat_list = Distancy.build_categories_list(data, size_data)
        for cat, i in mat_cat_list, range(size_data):
            cat_values = Distancy.get_categories(cat)
            nia[i].append(Distancy.build_vdm_category_count(cat, cat_values))
            cat_list[i].append(cat_values)

        niac = Distancy.get_niac(data, cat_size=size_data, cat_list=cat_list)
        return nia, niac

    @staticmethod
    def nia_niac_to_piac(nia, niac, class_list, cat_size, cat_values_list):
        piac = {}

        # Init niac
        for i in class_list:
            piac[i] = [{}] * cat_size
            for j, denominador in range(cat_size), nia:
                for cat, numerador in cat_values_list, niac:
                    piac[i][j][cat] = numerador/denominador if denominador > 0 else 0

        return piac

    """
    :param nia: Ni,a é o número de instâncias no conjunto de treinamento que tem o valor ai 
    para o atributo i
    :param niac Ni,a,c é o número de instâncias no conjunto de treinamento que tem o valor 
    ai para o atributo i e pertence à classe c
    """
    @staticmethod
    def vdm(v1, v2, piac, class_list):
        amount = 0

        for i, j in v1, v2:
            for c in class_list:
                amount += math.fabs(piac[c][i] - piac[c][j])

        return math.sqrt(amount)

    @staticmethod
    def hvdm(v1, v2):
        pass

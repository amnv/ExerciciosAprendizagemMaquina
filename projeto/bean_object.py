class Bean_object:
    def __init__(self, feature=[], label=False):
        self.feature = feature
        self.label = label
        self.classified = -1

    def build_features(self, line):
        features = tuple(line.split(","))
        last_item = len(features) - 1
        self.feature = features[:last_item - 1]

    def build_label(self, line):
        line = line.split("\n")[0]
        features = tuple(line.split(","))
        last_item = len(features) - 1
        self.label = features[last_item]

    @staticmethod
    def build_bean(text, count = -1):
        beans_list = []
        for line in text:
            aux = Bean_object()
            aux.build_features(line)
            aux.build_label(line)
            beans_list.append(aux)
            if count > 0:
                count -= 1
            elif count != -1:
                break

        return beans_list

    @staticmethod
    def build_bean_splited(data):
        data_list = []
        label_list = []

        for i in data:
            data_list.append(i.feature)
            label_list.append(i.label)

        return data_list, label_list

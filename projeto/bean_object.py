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
        features = tuple(line.split(","))
        last_item = len(features) - 1
        self.label = features[last_item]
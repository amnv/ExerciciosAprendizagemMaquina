import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

#base e dados da base
df = pd.read_csv("../base2.csv", names=["loc", "v(g)", "ev(g)",\
                                      "iv(g)", "n", "v", "l", "d","i", "e", "b", "t", "lOCode",\
                                      "loComment", "lOBlank", "locCodeAndComment", "uniq_Op", "uniq_Opnd", \
                                      "total_Op", "total_Opnd", "branchCount", "defects"])

x = df.values #returns a numpy array
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
data = pd.DataFrame(x_scaled, columns=df.columns)
size = data.shape[1]

# Training model
model = KNeighborsClassifier(n_neighbors=1)

# Slip data mode
cv = StratifiedKFold(n_splits=5)

# Samples for training
feature_data = data.iloc[:, :(size - 1)]
label_data = data.iloc[:, -1]

# Executing the train and test process
scores = cross_val_score(model, feature_data, label_data, cv=cv)

print("Model's acuracy")
print(scores)

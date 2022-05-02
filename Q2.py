from typing import Tuple
from math import log
from csv import reader

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss


# Function for getting X and y from csv
def getXy(csv: str) -> Tuple[list]:
    # Use pythons built-in csv parser
    with open(csv, "r") as csv:
        read = reader(csv)
        # [1:] to chop off the title row
        training = list(read)[1:]

    # Split into X and y
    X = [i[:-1] for i in training]
    y = [int(i[11]) for i in training]

    return X, y

# Fucntion for cross-entropy
def crossEntropy(h: list, c: list) -> float:
    ce = 0

    for i in range(len(h)):
        ce += c[i] * log(h[i][0]) + (1 - c[i]) * log(1 - h[i][0])

    return ce / -len(h)

"""
Load data
"""
# Load training data
training = getXy("data/training.csv")
X = training[0]
y = training[1]

# Load validation data
validation = getXy("data/validation.csv")
X_v = validation[0]
y_v = validation[1]

# Combine training and validation
X_b = X
y_b = y
X_b.extend(X_v)
y_b.extend(y_v)

# Load testing data
testing = getXy("data/testing.csv")
X_t = testing[0]
y_t = testing[1]
"""
3a)
"""
gini = DecisionTreeClassifier(criterion = "gini", max_depth = 5, random_state = 1)
gini.fit(X, y)

"""
3b)
"""
val_preds = gini.predict_proba(X_v)
print(f"Cross entropy for gini: {log_loss(y_v, val_preds)}") # log(0) error what the frick bro 81 instances like that

"""
3c) 
"""
info = DecisionTreeClassifier(criterion = "entropy", max_depth = 5, random_state = 1)
info.fit(X, y)

"""
3d)
"""
info_val_preds = info.predict_proba(X_v)
print(f"Cross entropy for info: {log_loss(y_v, info_val_preds)}")

"""
3e)
"""
# Info works better
comb = DecisionTreeClassifier(criterion = "entropy", max_depth = 5, random_state = 1)
comb.fit(X_b, y_b)

comb_val_preds = comb.predict_proba(X_t)
print(f"Cross entropy for comb: {log_loss(y_t, comb_val_preds)}")
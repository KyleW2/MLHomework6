from typing import Tuple
from math import log
from csv import reader

from sklearn.tree import DecisionTreeClassifier

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
3a)
"""
# Load training data
training = getXy("data/training.csv")
X = training[0]
y = training[1]

gini = DecisionTreeClassifier(criterion = "gini", max_depth = 5)
gini.fit(X, y)

"""
3b)
"""
# Load validation data
validation = getXy("data/validation.csv")
X_v = validation[0]
y_v = validation[1]

val_preds = gini.predict_proba(X_v)
print(f"Cross entropy: {crossEntropy(val_preds, y_v)}") # log(0) error what the frick bro 81 instances like that

"""
3c) 
"""
info = DecisionTreeClassifier(criterion = "entropy", max_depth = 5)
info.fit(X, y)

"""
3d)
"""
info_val_preds = info.predict_proba(X_v)
print(f"Cross entropy: {crossEntropy(info_val_preds, y_v)}")
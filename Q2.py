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
tree = DecisionTreeClassifier(criterion = "gini", max_depth = 5)

# Load training data
training = getXy("data/training.csv")
X = training[0]
y = training[1]

tree.fit(X, y)

"""
3b)
"""
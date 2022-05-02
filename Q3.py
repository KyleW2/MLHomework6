from typing import Tuple
from math import log
from csv import reader

from sklearn.ensemble import AdaBoostClassifier
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

def crossEntropy(h: list, c: list) -> float:
    ce = 0

    for i in range(len(h)):
        ce += c[i] * log(h[i][1]) + (1 - c[i]) * log(1 - h[i][1])

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
4a)
"""
one = AdaBoostClassifier(n_estimators = 20, random_state = 1)
two = AdaBoostClassifier(n_estimators = 40, random_state = 1)
three = AdaBoostClassifier(n_estimators = 60, random_state = 1)

one.fit(X, y)
two.fit(X, y)
three.fit(X, y)

"""
4b)
"""
one_preds = one.predict_proba(X_v)
two_preds = two.predict_proba(X_v)
three_preds = three.predict_proba(X_v)

print(f"One's cross entropy: {crossEntropy(one_preds, y_v)}")
print(f"Two's cross entropy: {crossEntropy(two_preds, y_v)}")
print(f"Three's cross entropy: {crossEntropy(three_preds, y_v)}")

"""
4c)
"""
# one/20 works best
comb = AdaBoostClassifier(n_estimators = 20, random_state = 1)
comb.fit(X_b, y_b)

comb_preds = comb.predict_proba(X_t)
print(f"Cross entropy: {crossEntropy(comb_preds, y_t)}")



def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from typing import Tuple
from math import log
from csv import reader

from sklearn.neural_network import MLPClassifier

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
        ce += c[i] * log(h[i][1]) + (1 - c[i]) * log(1 - h[i][1])

    return ce / -len(h)

"""
 2a)
"""
one_hidden_20_units = MLPClassifier(hidden_layer_sizes = (20), random_state = 1)
two_hidden_10_units = MLPClassifier(hidden_layer_sizes = (10, 10), random_state = 1)

# Load training data
training = getXy("data/training.csv")
X = training[0]
y = training[1]

# Fit models
one_hidden_20_units.fit(X, y)
two_hidden_10_units.fit(X, y)

"""
 2b)
"""
# Load validation data
validation = getXy("data/validation.csv")
X_v = validation[0]
y_v = validation[1]

# Make probabilistic predictions with predict_proba
first_preds = one_hidden_20_units.predict_proba(X_v)
second_preds = two_hidden_10_units.predict_proba(X_v)

print(f"First model cross entropy: {crossEntropy(first_preds, y_v)}")
print(f"Second model cross entropy: {crossEntropy(second_preds, y_v)}")

"""
 2c)
"""
# First network appears to preform better
# Combine training and validation
X_b = X
y_b = y
X_b.extend(X_v)
y_b.extend(y_v)

new_network = MLPClassifier(hidden_layer_sizes = (20), random_state = 1)
new_network.fit(X_b, y_b)

testing = getXy("data/testing.csv")
X_t = testing[0]
y_t = testing[1]

# Make new predictions
new_preds = new_network.predict_proba(X_t)
print(f"New model cross entropy: {crossEntropy(new_preds, y_t)}")
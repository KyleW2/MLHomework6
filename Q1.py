from sklearn.neural_network import MLPClassifier
from csv import reader

def getXy(csv: str):
    # Use pythons built-in csv parser to load rraining data
    with open(csv, "r") as csv:
        read = reader(csv)
        # [1:] to chop off the title row
        training = list(read)[1:]

    # Split training into X and y
    X = [i[:-1] for i in training]
    y = [i[11] for i in training]

    return X, y

# 2a)
one_hidden_20_units = MLPClassifier(solver = 'lbfgs', alpha = 1e-5, hidden_layer_sizes = (20), random_state = 1)
two_hidden_10_units = MLPClassifier(solver = 'lbfgs', alpha = 1e-5, hidden_layer_sizes = (10, 10), random_state = 1)

#   Load training data
training = getXy("data/training.csv")
X = training[0]
y = training[1]

#   Fit models
one_hidden_20_units.fit(X, y)
two_hidden_10_units.fit(X, y)

# 2b)
#   Load validation data
validation = getXy("data/validation.csv")
X_v = validation[0]
y_v = validation[1]

#   Make probabilistic predictions with predict_proba
first_preds = one_hidden_20_units.predict_proba(X_v)
second_preds = two_hidden_10_units.predict_proba(X_v)

print(first_preds)
print()
print(second_preds)
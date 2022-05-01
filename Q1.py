from sklearn.neural_network import MLPClassifier
from csv import reader

# Use pythons built-in csv parser to load rraining data
with open('data/training.csv', 'r') as csv:
    read = reader(csv)
    # [1:] to chop off the title row
    training = list(read)[1:]

# Split training into X and y
X = [i[:-1] for i in training]
y = [i[11] for i in training]

# 2a)
one_hidden_20_units = MLPClassifier(solver = 'lbfgs', alpha = 1e-5, hidden_layer_sizes = (20), random_state = 1)
two_hidden_10_units = MLPClassifier(solver = 'lbfgs', alpha = 1e-5, hidden_layer_sizes = (10, 10), random_state = 1)

one_hidden_20_units.fit(X, y)
two_hidden_10_units.fit(X, y)

# 2b)
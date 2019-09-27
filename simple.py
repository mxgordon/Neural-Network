import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_d(x):
    return x * (1 - x)


training_inputs = np.array([[0, 0, 1],
                            [1, 1, 1],
                            [1, 0, 1]])
                            # [0, 1, 1]])

training_outputs = np.array([[0, 1, 1]]).T

np.random.seed(1)

synaptic_weights = 2 * np.random.random((3, 1)) - 1
print(f"weights: \n{synaptic_weights}")

for iterr in range(20000):

    input_layer = training_inputs

    outputss = sigmoid(np.dot(input_layer, synaptic_weights))

    error = training_outputs - outputss

    adjustments = error * sigmoid_d(outputss)

    synaptic_weights += np.dot(input_layer.T, adjustments)

print(f'weights: \n{synaptic_weights}')

print(f'after training: \n{outputss}')

print(f'excpeced: \n{training_outputs}')
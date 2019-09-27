import numpy as np


class NeuralNetwork:

    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        return x * (1 - x)

    def sigmoid_driv(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, training_inputs, training_outputs, training_iterations):

        for iteration in range(training_iterations):

            output = self.think(training_inputs)

            error = training_outputs - output

            adjustments = np.dot(training_inputs.T, error * self.sigmoid_driv(output))

            self.synaptic_weights += adjustments

    def think(self, inputs):

        inputs = inputs.astype(float)

        outputs = self.sigmoid(np.dot(inputs, self.synaptic_weights))

        return outputs

if __name__ == "__main__":
    nn = NeuralNetwork()

    print(f"Synaptic Weights: \n{nn.synaptic_weights}")

    training_inputs = np.array([[0, 0, 1],
                                [1, 1, 1],
                                [1, 0, 1]])

    training_outputs = np.array([[0, 1, 1]]).T

    nn.train(training_inputs, training_outputs, 10000)

    print(f"synaptic Weights after training: {nn.synaptic_weights}")

    while True:

        A = int(input('Input 1: '))
        B = int(input('Input 2: '))
        C = int(input('Input 3: '))

        print(f'New situation: {[A, B, C]}')
        print(f'Output: {nn.think(np.array([A, B, C]))}')
import random
import unittest

# Euler's number (approximation of e)
E = 2.718281828459045

# Very simple implementation of a simple feedforward neural network with one hidden layer
# using the sigmoid activation function and an XOR example to test the network.
# Discretion: this was implemented by ChatGPT
class NeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize hidden layer weights and biases with random values
        self.hidden_weights = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(input_size)]
        self.hidden_biases = [random.uniform(-1, 1) for _ in range(hidden_size)]

        # Initialize output layer weights and biases with random values
        self.output_weights = [[random.uniform(-1, 1) for _ in range(output_size)] for _ in range(hidden_size)]
        self.output_biases = [random.uniform(-1, 1) for _ in range(output_size)]

    # Custom implementation of the exponential function
    def exponential(self, x):
        return E ** x

    # Activation function (sigmoid)
    def sigmoid(self, x):
        return 1 / (1 + self.exponential(-x))

    # Derivative of the sigmoid function
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # Calculate the output of the hidden layer for a given input
    def calculate_hidden_layer(self, inputs):
        hidden_layer = [
            self.sigmoid(
                sum(
                    inputs[i] * self.hidden_weights[i][j]
                    for i in range(self.input_size)
                ) + self.hidden_biases[j]
            )
            for j in range(self.hidden_size)
        ]

        return hidden_layer

    # Calculate the output of the neural network based on the hidden layer output
    def calculate_output_layer(self, hidden_layer):
        output_layer = [
            self.sigmoid(
                sum(
                    hidden_layer[i] * self.output_weights[i][j]
                    for i in range(self.hidden_size)
                ) + self.output_biases[j]
            )
            for j in range(self.output_size)
        ]
        return output_layer

    # Calculate the output of the neural network for a given input
    def feed_forward(self, inputs):
        hidden_layer = self.calculate_hidden_layer(inputs)
        output_layer = self.calculate_output_layer(hidden_layer)
        return output_layer

    # Update weights based on the calculated deltas
    def update_weights(self, weights, deltas, layer_output, learning_rate):
        for i in range(len(weights)):
            for j in range(len(weights[0])):
                weights[i][j] += learning_rate * deltas[j] * layer_output[i]

    # Update biases based on the calculated deltas
    def update_biases(self, biases, deltas, learning_rate):
        for i in range(len(biases)):
            biases[i] += learning_rate * deltas[i]

    # Train the neural network using backpropagation
    def train(self, training_inputs, training_outputs, epochs, learning_rate = 0.1):

        for epoch in range(epochs):
            for inputs, targets in zip(training_inputs, training_outputs):

                # Feedforward
                hidden_layer = self.calculate_hidden_layer(inputs)
                output_layer = self.calculate_output_layer(hidden_layer)

                # Perform backpropagation
                output_errors = [targets[i] - output_layer[i] for i in range(self.output_size)]

                output_deltas = [
                    output_errors[i] * self.sigmoid_derivative(output_layer[i])
                    for i in range(self.output_size)
                ]

                hidden_errors = [
                    sum(
                        self.output_weights[i][j] * output_deltas[j]
                        for j in range(self.output_size)
                    )
                    for i in range(self.hidden_size)
                ]

                hidden_deltas = [
                    hidden_errors[i] * self.sigmoid_derivative(hidden_layer[i])
                    for i in range(self.hidden_size)
                ]

                # Update weights and biases
                self.update_weights(self.output_weights, output_deltas, hidden_layer, learning_rate)
                self.update_weights(self.hidden_weights, hidden_deltas, inputs, learning_rate)
                self.update_biases(self.output_biases, output_deltas, learning_rate)
                self.update_biases(self.hidden_biases, hidden_deltas, learning_rate)


class Simple_Neural_Network_Test_Cases(unittest.TestCase):

    def test_simple_xor(selfs):

        # Create a small dataset for the XOR problem
        training_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
        training_outputs = [[0], [1], [1], [0]]

        # Initialize the neural network with 2 input neurons, 2 hidden neurons, and 1 output neuron
        neural_network = NeuralNetwork(2, 2, 1)

        # Train the neural network
        epochs = 100000
        learning_rate = 0.2
        neural_network.train(training_inputs, training_outputs, epochs, learning_rate)

        print("Testing the neural network")

        for inputs in training_inputs:
            output = neural_network.feed_forward(inputs)
            print(f"Input: {inputs}, Output: {output}")

        # Helper function which we use to unpack our feed_forward output
        unpack_output = lambda x: round(x[0]) if isinstance(x, list) else round(x)

        # Assert that our output is correct
        assert unpack_output(neural_network.feed_forward([0, 0])) == 0
        assert unpack_output(neural_network.feed_forward([0, 1])) == 1
        assert unpack_output(neural_network.feed_forward([1, 0])) == 1
        assert unpack_output(neural_network.feed_forward([1, 1])) == 0

if __name__ == "__main__":
    unittest.main()
import unittest

from matplotlib import pyplot as plt, pyplot

import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from sklearn import datasets

DEFAULT_SEED = 88

# A simple sigmoid neural network which lets us visualize the network architecture and
# the decision boundary that's formed. Example usage: nn = NeuralNetwork([2, 3, 4, 1])
class NeuralNetwork(object):

    def __init__(self, neuron_layer_sizes, seed = DEFAULT_SEED):
        np.random.seed(seed)
        self.neuron_layer_sizes = neuron_layer_sizes
        self.weights = [
            # declare a normally distributed random matrix the number of output neuron layer rows
            # and the number of input layer neuron columns
            np.random.randn(self.neuron_layer_sizes[neuron_layer_idx], self.neuron_layer_sizes[neuron_layer_idx - 1])
            # and multiply it by the square root of the 1 / number of input elements
            * np.sqrt(1 / self.neuron_layer_sizes[neuron_layer_idx - 1])
            # for each specified layer in our network
            for neuron_layer_idx in range(1, len(self.neuron_layer_sizes))
        ]

        # Create random biases for every neuron layer other than our first input layer
        self.biases = [
            np.random.rand(num_neurons, 1)
            for num_neurons
            in self.neuron_layer_sizes[1:]
        ]

    # Sigmoid activation function
    def activation( self, input, derivative=False):
        if not derivative:
            return 1 / (1 + np.exp(-input))
        else:
            sigmoid = self.activation(input)
            return sigmoid * (1 - sigmoid)

    # Calculate the cost for regression problems (or the cost of its derivative).
    def cost_function( self, true_values, predicted_values, derivative=False):
        if not derivative:
            num_outputs = predicted_values.shape[1]
            cost = (1. / (2 * num_outputs)) * np.sum((true_values - predicted_values) ** 2)
            return cost
        else:
            return predicted_values - true_values

    def feed_forward(self, input_data):
        # Initialize lists to store pre-activation outputs and activation outputs for each layer
        pre_activations = []
        activations = [input_data]

        # Set the input data as the next input to start the feed-forward process
        next_input = input_data

        # Iterate through each layer's weights and biases in the neural network
        for weight, bias in zip(self.weights, self.biases):
            # Compute the pre-activation output for the current layer
            pre_activation_output = np.dot(weight, next_input) + bias

            # Store the pre-activation output
            pre_activations.append(pre_activation_output)

            # Compute the activation output for the current layer using the activation function
            output = self.activation(pre_activation_output)

            # Store the activation output
            activations.append(output)

            # Set the current output as the next input for the next layer in the feed-forward process
            next_input = output

        # Return a dictionary containing our results
        return {
            "output": output,
            "pre_activations": pre_activations,
            "activations": activations
        }

    # Compute and return the deltas (errors) for each neural network layer
    def compute_deltas(self, pre_activations, true_values, predicted_values):

        # Calculate the loss delta using the cost function's derivative and the derivative of the activation function
        loss_delta = self.cost_function(true_values, predicted_values, derivative=True) \
                     * self.activation(pre_activations[-1], derivative=True)

        # Get the total number of layers in the neural network
        num_nn_layers = len(self.neuron_layer_sizes)

        # Initialize a list to store the deltas for each layer (excluding the input layer)
        nn_layer_deltas = [0] * (num_nn_layers - 1)

        # Set the last delta (output layer) to the computed loss_delta
        nn_layer_deltas[-1] = loss_delta

        # Iterate through each layer (except the output and input layers) in reverse order and compute the deltas
        for cur_layer_idx in range(len(nn_layer_deltas) - 2, -1, -1):
            # Calculate the deltas for the current layer using the transpose of the next layer's weights
            nn_layer_deltas[cur_layer_idx] = np.dot(
                self.weights[cur_layer_idx + 1].transpose(),
                nn_layer_deltas[cur_layer_idx + 1]
            ) * self.activation(
                input=pre_activations[cur_layer_idx],
                derivative=True
            )

        return nn_layer_deltas

    # Compute backpropagation using the fed-in deltas and activations
    def backpropagate(self, deltas, activations):
        # Initialize lists to store weight and bias gradients for each layer
        weight_gradients = []
        bias_gradients = []

        # Prepend a zero delta to represent the input layer (no updates needed)
        deltas = [0] + deltas

        # Iterate through each layer (except the input layer) to compute gradients
        for cur_layer_idx in range(1, len(self.neuron_layer_sizes)):
            # Compute the weight gradient for the current layer
            weight_loss = np.dot(
                deltas[cur_layer_idx],
                activations[cur_layer_idx - 1].transpose()
            )

            # Compute the bias gradient for the current layer
            bias_loss = deltas[cur_layer_idx]

            # Append the computed gradients to the respective lists
            weight_gradients.append(weight_loss)
            bias_gradients.append(np.expand_dims(bias_loss.mean(axis=1), 1))

        # Return the weight and bias gradients as a tuple
        return (weight_gradients, bias_gradients)

    def predict(self, x):
        # Iterate through each layer's weights and biases in the neural network
        for weights, bias in zip(self.weights, self.biases):
            # Perform the pre-activation step by multiplying the input data with weights and adding the bias
            pre_activation = np.dot(weights, x) + bias
            # Apply the activation function to the pre-activation to get the output of the current layer
            x = self.activation(pre_activation)

        # Convert the final output to binary values (0 or 1) based on a threshold (0.5)
        predictions = (x > 0.5).astype(int)

        return predictions

    # Function used to plot the decision boundary for our neural network
    def plot_decision_regions(self, inputs, outputs, iteration, training_loss, validation_loss, training_accuracy
                              , validation_accuracy, resolution=0.01):
        # Transpose the input data to make it compatible with the following calculations
        inputs, outputs = inputs.T, outputs.T

        # Calculate the minimum and maximum values for the x and y coordinates with some padding
        padding = 0.5
        min_x, max_x = inputs[:, 0].min() - padding, inputs[:, 0].max() + padding
        min_y, max_y = inputs[:, 1].min() - padding, inputs[:, 1].max() + padding

        # Create a grid of points
        xx, yy = np.meshgrid(np.arange(min_x, max_x, resolution),
                             np.arange(min_y, max_y, resolution))

        # Predict the output for all points in the grid using the neural network's predict function
        predicted_output = self.predict(np.c_[xx.ravel(), yy.ravel()].T)
        predicted_output = predicted_output.reshape(xx.shape)

        # Create a filled contour plot based on the predicted output
        plt.contourf(xx, yy, predicted_output, alpha=0.5)

        # Set the plot limits to match the grid's limits
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())

        # Scatter plot the original data points with their corresponding colors
        plt.scatter(inputs[:, 0], inputs[:, 1], c=outputs.reshape(-1), alpha=0.2)

        # Create and plot results
        results = f'Iteration: {iteration} | ' \
                  f'Training Loss: {training_loss} | ' \
                  f'Validation Loss: {validation_loss} | ' \
                  f'Training Accuracy: {training_accuracy} | ' \
                  f'Validation Accuracy: {validation_accuracy}'

        plt.title(results)

    # Draw the neural network architecture
    def draw(self):
        network = NeuralNetworkVisual(self.neuron_layer_sizes)
        network.draw()

    def train(self, inputs, outputs, batch_size, epochs, learning_rate, validation_split=0.2, print_every=10):
        # Initialize lists to store training and test history
        training_loss_history = []
        training_accuracy_history = []
        test_loss_history = []
        test_accuracy_history = []

        # Split the data into training and test sets using the specified validation split ratio
        x_train, x_test, y_train, y_test = train_test_split(inputs.T, outputs.T, test_size=validation_split)

        # Transpose the data to match the required shape for the following calculations
        x_train = x_train.T
        x_test = x_test.T
        y_train = y_train.T
        y_test = y_test.T

        # Let's do our training by iterating through each epoch
        for epoch in range(epochs):

            # Calculate the number of batches
            if x_train.shape[1] % batch_size == 0:
                num_batches = int(x_train.shape[1] / batch_size)
            else:
                num_batches = int(x_train.shape[1] / batch_size) - 1

            # Shuffle the training data for each epoch to improve training performance
            x_train, y_train = shuffle(x_train.T, y_train.T)
            x_train, y_train = x_train.T, y_train.T

            # Divide the data into batches
            x_batches = [x_train[:, batch_size * i:batch_size * (i + 1)] for i in range(0, num_batches)]
            y_batches = [y_train[:, batch_size * i:batch_size * (i + 1)] for i in range(0, num_batches)]

            # Initialize lists to store losses and accuracies for training and test sets within this epoch
            train_losses = []
            train_accuracies = []
            test_losses = []
            test_accuracies = []

            # Initialize gradients for weights and biases for this epoch
            dw_per_epoch = [np.zeros(w.shape) for w in self.weights]
            db_per_epoch = [np.zeros(b.shape) for b in self.biases]

            # Iterate through each batch and perform forward and backward propagation
            for x_batch, y_batch in zip(x_batches, y_batches):

                # Perform forward propagation for the current batch
                feed_forward_output = self.feed_forward(x_batch)
                batch_y_pred = feed_forward_output.get("output")
                pre_activations = feed_forward_output.get("pre_activations")
                activations = feed_forward_output.get("activations")

                # Compute the deltas for the current batch
                deltas = self.compute_deltas(pre_activations, y_batch, batch_y_pred)
                dw, db = self.backpropagate(deltas, activations)

                # Update the gradients for weights and biases for this batch
                for i, (dw_i, db_i) in enumerate(zip(dw, db)):
                    dw_per_epoch[i] += dw_i / batch_size
                    db_per_epoch[i] += db_i / batch_size

                # Predict outputs and calculate losses and accuracies for the training set
                batch_y_train_pred = self.predict(x_batch)
                train_loss = self.cost_function(y_batch, batch_y_train_pred)
                train_losses.append(train_loss)
                train_accuracy = accuracy_score(y_batch.T, batch_y_train_pred.T)
                train_accuracies.append(train_accuracy)

                # Predict outputs and calculate losses and accuracies for the test set
                batch_y_test_pred = self.predict(x_test)
                test_loss = self.cost_function(y_test, batch_y_test_pred)
                test_losses.append(test_loss)
                test_accuracy = accuracy_score(y_test.T, batch_y_test_pred.T)
                test_accuracies.append(test_accuracy)

            # Update weights and biases for this epoch using the computed gradients
            for i, (dw_epoch, db_epoch) in enumerate(zip(dw_per_epoch, db_per_epoch)):
                self.weights[i] = self.weights[i] - learning_rate * dw_epoch
                self.biases[i] = self.biases[i] - learning_rate * db_epoch

            # Store the average losses and accuracies for this epoch in the history lists
            training_loss_history.append(np.mean(train_losses))
            training_accuracy_history.append(np.mean(train_accuracies))
            test_loss_history.append(np.mean(test_losses))
            test_accuracy_history.append(np.mean(test_accuracies))

            # Print progress every few epochs
            if not print_every <= 0:
                if epoch % print_every == 0:
                    print(
                        f'Epoch {epoch} / {epochs} | '
                        f'Training Loss: {np.round(np.mean(train_losses), 4)} | '
                        f'Training Accuracy: {np.round(np.mean(train_accuracies), 4)} | '
                        f'Validation Loss: {np.round(np.mean(test_losses), 4)} | '
                        f'Validation Accuracy: {np.round(np.mean(test_accuracies), 4)} '
                    )

                    self.plot_decision_regions(x_train, y_train, epoch,
                                               np.round(np.mean(train_losses), 4),
                                               np.round(np.mean(test_losses), 4),
                                               np.round(np.mean(train_accuracies), 4),
                                               np.round(np.mean(test_accuracies), 4),
                                               )
                    plt.show()

        # Return a dictionary containing our results
        history = {
            'epochs': epochs,
            'train_loss': training_loss_history,
            'train_accuracy': training_accuracy_history,
            'test_loss': test_loss_history,
            'test_accuracy': test_accuracy_history
        }

        return history

# Visualization for a single neuron
class NeuronVisual():

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, neuron_radius):
        # Draw a circle representing the neuron
        circle = plt.Circle((self.x, self.y), radius=neuron_radius, fill=False)
        plt.gca().add_patch(circle)

# Visualization for a neural network layers
class LayerVisual():

    def __init__(self, network, number_of_neurons, number_of_neurons_in_widest_layer, font_size = 11):
        # Set distances and dimensions for the neural network visualization
        self.vertical_distance_between_layers = 10
        self.horizontal_distance_between_neurons = 5
        self.neuron_radius = 0.5
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer

        # Set the label font size
        self.font_size = font_size

        # Initialize properties for the current layer
        self.previous_layer = self.previous_layer(network)
        self.y = self.calculate_layer_y_position()
        self.neurons = self.initialize_neurons(number_of_neurons)

    def initialize_neurons(self, number_of_neurons):
        neurons = []
        x = self.calculate_left_margin_so_layer_is_centered(number_of_neurons)

        # Create neuron instances and position them horizontally
        for _ in range(number_of_neurons):
            neuron = NeuronVisual(x, self.y)
            neurons.append(neuron)
            x += self.horizontal_distance_between_neurons

        return neurons

    # Calculate the left margin to center the layer
    def calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return self.horizontal_distance_between_neurons * (
            self.number_of_neurons_in_widest_layer - number_of_neurons) / 2

    # Calculate the vertical position of the layer
    def calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + self.vertical_distance_between_layers
        else:
            return 0

    # Get the previous layer in the network
    def previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def line_between_two_neurons(self, neuron1, neuron2):
        # Calculate the angle between the two neurons
        angle = np.arctan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))

        # Calculate adjustments for the line coordinates based on the neuron radius and angle
        x_adjustment = self.neuron_radius * np.sin(angle)
        y_adjustment = self.neuron_radius * np.cos(angle)

        # Create a Line2D object to draw the line between the two neurons
        line = plt.Line2D(
              xdata = (neuron1.x - x_adjustment, neuron2.x + x_adjustment)
            , ydata = (neuron1.y - y_adjustment, neuron2.y + y_adjustment)
        )

        # Add the line to the current plot
        plt.gca().add_line(line)

    def draw(self, layer_type=0):
        # Draw neurons and connections for the current layer
        for neuron in self.neurons:
            neuron.draw(self.neuron_radius)
            if self.previous_layer:
                for prev_neuron in self.previous_layer.neurons:
                    self.line_between_two_neurons(neuron, prev_neuron)

        x_text = self.number_of_neurons_in_widest_layer * self.horizontal_distance_between_neurons

        # Add text label for the layer
        if layer_type == 0:
            plt.text(x_text, self.y, 'Input Layer', fontsize=self.font_size)
        elif layer_type == -1:
            plt.text(x_text, self.y, 'Output Layer', fontsize=self.font_size)
        else:
            plt.text(x_text, self.y, 'Hidden Layer ' + str(layer_type), fontsize=self.font_size)

# Used for visualizing the entire neural network
class NeuralNetworkVisual():

    def __init__(self, neuron_layer_sizes):

        # Find the widest layer by taking the maximum number of neurons among all layers
        num_neurons_in_widest_layer = max(neuron_layer_sizes)
        self.num_neurons_in_widest_layer = num_neurons_in_widest_layer

        # Initialize layers
        self.layers = []
        for num_neurons in neuron_layer_sizes:
            self.add_layer_visual(num_neurons)

    # Create a layer visual
    def add_layer_visual(self, number_of_neurons):
        layer = LayerVisual(self, number_of_neurons, self.num_neurons_in_widest_layer)
        self.layers.append(layer)

    # Draw the entire neural network architecture
    def draw(self):
        # Create a new plot figure
        plt.figure()
        for i, layer in enumerate(self.layers):
            # Determine the layer type (input, hidden, or output) and draw it
            layer_type = i if i != len(self.layers) - 1 else -1
            layer.draw(layer_type)

        # Set plot axis properties
        plt.axis('scaled')  # Maintain equal scaling in both x and y directions
        plt.axis('off')  # Turn off axis labels and ticks

        # Set plot title
        plt.title('Neural Network Architecture', fontsize=14)

        # Display the plot
        plt.show()

# Test cases used to validate our network
class VisualNeuralNetTestCases(unittest.TestCase):

    # Test case for a dataset generated using make_blobs data set
    def test_blob_data_set(self):

        # Generate isotropic Gaussian blobs for clustering with 2 centers and 1000 samples
        data = datasets.make_blobs(n_samples=1000, centers=2, random_state=2)

        X = data[0].T  # Input features (transposed for neural network compatibility)
        y = np.expand_dims(data[1], 1).T  # Corresponding labels (transposed)

        # Create a neural network with the following layers -> [2, 7, 7, 1] (2 input layers and 1 output)
        neural_net = NeuralNetwork([2, 7, 7, 1], seed=40)
        neural_net.draw()  # Visualize the neural network architecture

        # Train the neural network on the generated data and obtain the results
        results = neural_net.train(
            inputs=X, outputs=y,
            batch_size=16,
            epochs=150,
            learning_rate=0.4,
            print_every=30,
            validation_split=0.2
        )

        train_accuracy = results.get('train_accuracy').pop()
        test_accuracy = results.get('test_accuracy').pop()

        # Ensure that the resultant training accuracy is above 90 percent
        min_accuracy = 0.90
        assert train_accuracy >= min_accuracy
        assert test_accuracy >= min_accuracy

    # Test case for a dataset generated using make_moons data set
    def test_moon_data_set(self):

        # Generate 1000 samples segmented into two interleaving half circles (moon-shaped sets)
        data = datasets.make_moons(n_samples=1000, noise=0.1)

        X = data[0].T  # Input features (transposed)
        y = np.expand_dims(data[1], 1).T  # Corresponding labels (transposed)

        # Create a neural network with the following layers -> [2, 8, 8, 6, 1] (2 input layers and 1 output)
        neural_net = NeuralNetwork([2, 8, 8, 6, 1], seed=20)
        neural_net.draw()  # Visualize the neural network architecture

        # Train the neural network on the generated data and obtain the results
        results = neural_net.train(
            inputs=X, outputs=y,
            batch_size=32,
            epochs=700,
            learning_rate=0.4,
            print_every=100,
            validation_split=0.2
        )

        train_accuracy = results.get('train_accuracy').pop()
        test_accuracy = results.get('test_accuracy').pop()

        # Ensure that the resultant accuracy is above 80 percent
        min_accuracy = 0.80
        assert train_accuracy >= min_accuracy
        assert test_accuracy >= min_accuracy

if __name__ == '__main__':
    unittest.main()
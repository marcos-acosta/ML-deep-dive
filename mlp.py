# Following instructions at https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

from random import seed, random
from math import exp

# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network

# Calculate neuron activation for an input
def activate(weights, inputs):
    # Start with bias
    activation = weights[-1]
    # Add weights
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation

# Activation function
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))

def transfer_derivative(output):
    return output * (1 - output)

# Feed forward
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

def backward_propagate_error(network, expected):
    # For each layer
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        # Not output layer
        if i != len(network) - 1:
            # For each neuron in this layer
            for j in range(len(layer)):
                error = 0.0
                # For each neuron in next layer
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        # Output layer
        else:
            # For each neuron in this layer
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        # For each neuron in this layer
        for j in range(len(layer)):
            neuron = layer[j]
            # Set delta
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

def update_weights(network, row, l_rate):
    # For each layer
    for i in range(len(network)):
        # Default inputs
        inputs = row[:-1]
        # If not first layer
        if i != 0:
            # Get inputs from previous layer
            inputs = [neuron['output'] for neuron in network[i - 1]]
        # For every neuron in this layer
        for neuron in network[i]:
            # Update all weights from inputs
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            # Update bias
            neuron['weights'][-1] += l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        # For each sample
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] - outputs[i])**2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        print('Epoch {}, lrate={:.3f}, error={:.3f}'.format(epoch, l_rate, sum_error))

def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))

# Test training backprop algorithm
seed(1)
dataset = [
    [2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]
]
n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
network = initialize_network(n_inputs, 2, n_outputs)
train_network(network, dataset, 0.5, 1000, n_outputs)

# Predict
for row in dataset:
	prediction = predict(network, row)
	print('Expected={}, Got={}'.format(row[-1], prediction))
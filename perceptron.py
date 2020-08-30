# Following instructions from https://www.kdnuggets.com/2019/11/build-artificial-neural-network-scratch-part-1.html

import numpy as np

# Independent variables (X)
input_set = np.array(
    [[0,1,0],
    [0,0,1],
    [1,0,0],
    [1,1,0],
    [1,1,1],
    [0,1,1],
    [0,1,0]])
# Dependent variable (y)
labels = np.array([[1, 0, 0, 1, 1, 0, 1]])
labels = labels.reshape(7, 1)

# Hyperparameters
np.random.seed(42)
weights = np.random.rand(3,1)
bias = np.random.rand(1)
lr = 0.05
epochs = 25000

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Train
for epoch in range(epochs):
    # Feed forward
    inputs = input_set
    XW = np.dot(inputs, weights) + bias
    z = sigmoid(XW)
    # Find error
    error = z - labels
    print('Error: {}'.format(error.sum()))
    # Backpropagate
    dcost = error
    dpred = sigmoid_derivative(z)
    # Slope = input * dcost * dpred
    z_del = dcost * dpred
    weights = weights - lr * np.dot(input_set.T, z_del)

    # Update bias
    for num in z_del:
        bias = bias - lr * num

# Predict
single_pt = np.array([1, 0, 0])
result = sigmoid(np.dot(single_pt, weights) + bias)
print(result)

single_pt = np.array([0, 1, 0])
result = sigmoid(np.dot(single_pt, weights) + bias)
print(result)
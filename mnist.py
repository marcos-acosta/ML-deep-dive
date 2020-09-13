import numpy as np
import math
from matplotlib import pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import SGD
from sklearn.model_selection import KFold
from numpy import mean, std

def load_data(verbose=True):
    ''' Load data '''
    # Read train data: shape = (42001, 785)
    train = np.genfromtxt('data/mnist/train.csv', delimiter=',')
    # Read test data: shape = (28001, 784)
    test = np.genfromtxt('data/mnist/test.csv', delimiter=',')

    ''' Split into train, valid, and test '''
    # Shave off X with shape (42001, 784)
    X = train[1:, 1:]
    # Transform X data to shape (42001, 26, 26)
    X = np.reshape(X, newshape=(X.shape[0], 28, 28, 1))
    # Shave off y with shape (42001,)
    y = train[1:, 0]
    y = to_categorical(y, num_classes=10, dtype='int32')
    # Similarly for test
    test = test[1:, :]
    testX = np.reshape(test, newshape=(test.shape[0], 28, 28, 1))

    if verbose:
        print('X: {}, testX: {}.'.format(X.shape, testX.shape))
        print('y: {}'.format(y.shape))

    return X, y, testX

def normalize(x):
    x = x.astype('float32')
    x_norm = x / 255.0
    return x_norm

def prep_data(X, testX):
    X_norm = normalize(X)
    testX_norm = normalize(testX)
    return X_norm, testX_norm

# Define CNN model
def define_model():
	model = Sequential()
    # Apply 32 filters (depth of 32), each with shape 3x3. This results in a (26,26,32) volume
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    # Pool. This results in a (13,13,32) volume
	model.add(MaxPooling2D((2, 2)))
    # Flatten into a 1D vector. This will have shape (5408,)
	model.add(Flatten())
    # Typical neural network. Hidden layer with size 128
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    # Dense output layer
	model.add(Dense(10, activation='softmax'))
	# Compile model
	opt = SGD(lr=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

# Evaluate a model using k-fold cross-validation
def evaluate_model(dataX, datay, n_folds=5):
	scores, histories = list(), list()
	# Prepare cross validation
	kfold = KFold(n_folds, shuffle=True, random_state=1)
	# Enumerate splits
	for train_ix, test_ix in kfold.split(dataX):
		# Define model
		model = define_model()
		# Select rows for train and test
		trainX, trainy, validX, validy = dataX[train_ix], datay[train_ix], dataX[test_ix], datay[test_ix]
		# Fit model
		history = model.fit(trainX, trainy, epochs=10, batch_size=32, validation_data=(validX, validy), verbose=1)
		# Evaluate model
		_, acc = model.evaluate(validX, validy, verbose=0)
		print('> %.3f' % (acc * 100.0))
		# Stores scores
		scores.append(acc)
		histories.append(history)
	return scores, histories

# Plot diagnostic learning curves
def summarize_diagnostics(histories):
	for i in range(len(histories)):
		# plot loss
		plt.subplot(2, 1, 1)
		plt.title('Cross Entropy Loss')
		plt.plot(histories[i].history['loss'], color='blue', label='train')
		plt.plot(histories[i].history['val_loss'], color='orange', label='test')
		# plot accuracy
		plt.subplot(2, 1, 2)
		plt.title('Classification Accuracy')
		plt.plot(histories[i].history['accuracy'], color='blue', label='train')
		plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
	plt.show()

# Summarize model performance
def summarize_performance(scores):
	# Print summary
	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
	# box and whisker plots of results
	plt.boxplot(scores)
	plt.show()

# Run the test harness for evaluating a model
def run_test_harness():
	# Load dataset
	X, y, testX = load_data(verbose=True)
	# Prepare pixel data
	X, testX = prep_data(X, testX)
	# Evaluate model
	scores, histories = evaluate_model(X, y)
	# Learning curves
	summarize_diagnostics(histories)
	# Summarize estimated performance
	summarize_performance(scores)

run_test_harness()
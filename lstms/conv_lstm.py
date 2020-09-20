from numpy import array, reshape
from keras.models import Sequential
from keras.layers import LSTM, Dense, ConvLSTM2D, Flatten

def split_sequences(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence) - n_steps):
        # find the end of this pattern
        end_ix = i + n_steps
        # gather input and output parts
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# Dummy data
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
''' Need input to be [samples, timesteps, rows, columns, features] '''
# Set timestep length
n_steps = 4
# Split into sequences for the LSTM
X, y = split_sequences(raw_seq, n_steps)
# Reshape from [samples, timesteps] to [samples, timesteps, features]
n_features = 1
n_seq = 2
n_steps = 2
# X.shape[0] samples, each with 1 row and 2 columns where each 'pixel' has 2 timesteps, each with 1 feature
X = reshape(X, newshape=(X.shape[0], n_seq, 1, n_steps, n_features))
# for i in range(len(X)):
#     print(X[i])
#     print(y[i])
# Define model
model = Sequential()
model.add(ConvLSTM2D(filters=64, kernel_size=(1, 2), activation='relu', input_shape=(n_seq, 1, n_steps, n_features)))
model.add(Flatten())
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=500, verbose=0)
# Demonstrate prediction
x_input = array([60, 70, 80, 90])
x_input = reshape(x_input, newshape=(1, n_seq, 1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
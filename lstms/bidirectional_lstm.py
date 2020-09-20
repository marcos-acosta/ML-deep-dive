from numpy import array, reshape
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional

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
# Set timestep length
n_steps = 3
# Split into sequences for the LSTM
X, y = split_sequences(raw_seq, n_steps)
# Reshape from [samples, timesteps] to [samples, timesteps, features]
n_features = 1
X = reshape(X, newshape=(X.shape[0], X.shape[1], n_features))
# Define model
model = Sequential()
model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=200, verbose=0)
# Demonstrate prediction
x_input = array([100, 110, 120])
x_input = reshape(x_input, newshape=(1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
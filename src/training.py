from tensorflow import keras
from keras.layers import Dense
from keras.layers import LSTM

def training(X_train_data, Y_train_data):
    regressor = keras.models.Sequential()
    regressor.add(LSTM(units = 60, return_sequences =True, input_shape = (X_train_data.shape[1], 4)))
    regressor.add(LSTM(units = 80))
    regressor.add(Dense(units = 1))
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    regressor.fit(X_train_data, Y_train_data, epochs = 50, batch_size=32)
    regressor.save("../models/")
    return regressor
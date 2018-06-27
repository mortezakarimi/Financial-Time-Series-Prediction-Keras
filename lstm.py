import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.preprocessing as preprocessing
from sklearn.metrics import mean_squared_error
from utils import wavelet_denoising, new_dataset, theil, MAPE
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU
from keras.layers import LSTM
from keras import losses

dataset = pd.read_csv('000001.SS.csv', usecols=[1, 2, 3, 4])
# CREATING OWN INDEX FOR FLEXIBILITY

OHLC_avg = wavelet_denoising(dataset.mean(axis=1))

OHLC_avg = np.reshape(OHLC_avg, (len(OHLC_avg), 1))

scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
OHLC_avg = scaler.fit_transform(OHLC_avg)

n = len(OHLC_avg)

train_start = 0
train_end = int(np.floor(0.8 * n))
validation_start = train_end
validation_end = validation_start + int(np.ceil(0.1 * n))
test_start = validation_end
test_end = n

data_train = OHLC_avg[train_start: train_end, :]
data_validation = OHLC_avg[validation_start: validation_end, :]
data_test = OHLC_avg[np.arange(test_start, test_end), :]

trainX, trainY = new_dataset(data_train, 1)
validX, validY = new_dataset(data_validation, 1)
testX, testY = new_dataset(data_test, 1)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
validX = np.reshape(validX, (validX.shape[0], 1, validX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

step_size = 1

model = Sequential()
model.add(LSTM(32, input_shape=(1, 1), return_sequences=True))
model.add(LeakyReLU(alpha=0.3))
model.add(LSTM(16, dropout=0.25))

model.add(Dense(1))

model.compile(loss=losses.mean_absolute_percentage_error, optimizer='rmsprop')
model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=1, validation_data=(validX, validY))

trainPredict = model.predict(trainX)
validPredict = model.predict(validX)
testPredict = model.predict(testX)

# DE-NORMALIZING FOR PLOTTING
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
validPredict = scaler.inverse_transform(validPredict)
validY = scaler.inverse_transform([validY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# TRAINING theil and MAPE
trainScore = theil(trainY[0], trainPredict[:, 0])
train_mape = MAPE(trainY[0], trainPredict[:, 0])
train_mse = mean_squared_error(trainY[0], trainPredict[:, 0])
print('Train Theil: %.4f And Train MAPE: %.4f And Train MSE: %.4f' % (trainScore, train_mape, train_mse))

# Validation theil and MAPE
validScore = theil(validY[0], validPredict[:, 0])
valid_mape = MAPE(validY[0], validPredict[:, 0])
valid_mse = mean_squared_error(validY[0], validPredict[:, 0])
print('Validation Theil: %.4f And Validation MAPE: %.4f And Validation MSE: %.4f' % (validScore, valid_mape, valid_mse))

# Test theil and MAPE
testScore = theil(testY[0], testPredict[:, 0])
test_mape = MAPE(testY[0], testPredict[:, 0])
test_mse = mean_squared_error(testY[0], testPredict[:, 0])
print('Test Theil: %.4f And Test MAPE: %.4f And Test MSE: %.4f' % (testScore, test_mape, test_mse))

# CREATING SIMILAR DATASET TO PLOT TRAINING PREDICTIONS
trainPredictPlot = np.empty_like(OHLC_avg)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[step_size:len(trainPredict) + step_size, :] = trainPredict

# CREATING SIMILAR DATASSET TO PLOT Validation PREDICTIONS
validPredictPlot = np.empty_like(OHLC_avg)
validPredictPlot[:, :] = np.nan
validPredictPlot[validation_start + step_size:validation_end, :] = validPredict

# CREATING SIMILAR DATASSET TO PLOT TEST PREDICTIONS
testPredictPlot = np.empty_like(OHLC_avg)
testPredictPlot[:, :] = np.nan
testPredictPlot[test_start + step_size:test_end, :] = testPredict

# DE-NORMALIZING MAIN DATASET
OHLC_avg = scaler.inverse_transform(OHLC_avg)

# PLOT OF MAIN OHLC VALUES, TRAIN PREDICTIONS AND TEST PREDICTIONS
plt.plot(OHLC_avg, label='original dataset')
plt.plot(trainPredictPlot, '--', label='training set')
plt.plot(validPredictPlot, '--', label='predicted stock price/validation set')
plt.plot(testPredictPlot, '--', label='predicted stock price/test set')
plt.xlim(xmin=0, xmax=len(OHLC_avg))
plt.ylim(ymin=1500, ymax=5500)
plt.legend(loc='upper right')
plt.xlabel('Time in Days')
plt.ylabel('OHLC Value of SHI')
plt.savefig('Prediction result.png')
plt.show()

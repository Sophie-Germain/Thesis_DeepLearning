#LSTM for time series prediction

import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os
#os.chdir("/home/sophie/tesis/data")
os.chdir("/home/sophie/tesis/data/final_code")

#============================================================================
#==============================PREPROCESSING=================================
#============================================================================

# fix random seed for reproducibility
numpy.random.seed(103596)
# loading the dataset

print('Graficamos la serie de tiempo: ')
dataframe_noest = pandas.read_csv('data_noest.csv', usecols=[0], engine='python')
plt.plot(dataframe_noest, color='#5e6d96')
plt.ylabel('Demanda sin transformar (MWs)')
plt.xlabel('Tiempo')
#plt.show()


#dataset = dataset.astype('float32')

# Hacemos los datos estacionarios en R e importamos el csv
print('Graficamos la serie de tiempo estacionaria: ')
dataframe = pandas.read_csv('data_est.csv', usecols=[0], engine='python')
plt.plot(dataframe, color='#5e6d96')
plt.ylabel('Demanda estacionaria')
plt.xlabel('Tiempo')
#plt.show()
# we only keep the underlying values
dataset = dataframe.values

# scale the dataset to (0,1) to be able to use a LSTM
# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# This transformation is often used as an alternative to zero mean, unit variance scaling.
scaler = MinMaxScaler(feature_range=(0, 1))
#fit to data, then transform it
dataset = scaler.fit_transform(dataset)

print('Graficamos la serie de tiempo estacionaria y escalada en (0,1): ')
plt.plot(dataset, color='#5e6d96')
plt.ylabel('Demanda estacionaria y escalada en (0,1)')
plt.xlabel('Tiempo')
#plt.show()

print('Creamos dataset de entrenamiento y prueba: ')
# split into train and test sets. (to be refined for time series prediction)
train_size = int(len(dataset) * 0.8)
#test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# reshape into X=t and Y=t+1
##### convert an array of values into a dataset matrix for time series prediction
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

#Cambiamos el tamaño de la ventana
look_back = 9
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

#currently data is in the form of [samples, features]
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))
#============================================================================
#===============================MODEL========================================
#============================================================================
# create and fit the LSTM network
#The network has a visible layer with 1 input, a hidden layer with 4 LSTM blocks or neurons and an output layer that makes a single value prediction. The default sigmoid activation function is used for the LSTM blocks. The network is trained for 100 epochs and a batch size of 1 is used.

print('Creando arquitectura LSTM: ')
batch_size = 1
model = Sequential()

#look_back=input_length, input_dim=1
#Deep LSTM with 2 stacked layers with 4 blocks each
##Stateful=true porque no se resetea en cada tiempo
model.add(LSTM(6, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=False))
#model.add(Dropout(0.5))
#model.add(LSTM(10, return_sequences=False, stateful=True))
#model.add(LSTM(10, return_sequences=True))
#model.add(LSTM(10, return_sequences=True))
#model.add(LSTM(10, return_sequences=True))
#model.add(LSTM(10, return_sequences=True))
#model.add(LSTM(10, return_sequences=True))
#model.add(LSTM(10, return_sequences=True))
#model.add(LSTM(10, return_sequences=True))
#model.add(LSTM(10, return_sequences=True))

#output layers. Size=1 (1 timestep)
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
print('Entrenando modelo: ')
for i in range(200):
    model.fit(trainX, trainY, nb_epoch=1, batch_size=batch_size, verbose=2, shuffle=False)
    model.reset_states()
print('El numero de parametros en cada capa es: ')
model.summary()

#============================================================================
#===============================EVALUATION===================================
#============================================================================
# make predictions
#print('Haciendo predicciones: ')
trainPredict = model.predict(trainX, batch_size=batch_size)
model.reset_states()
testPredict = model.predict(testX, batch_size=batch_size)
# invert predictions to its original units
print('Desescalando los datos: ')
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error

print('Evaluando modelo en entrenamiento y prueba: ')
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.5f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.5f RMSE' % (testScore))

#============================================================================
#===============================VISUALIZATION================================
#============================================================================
print('Graficando datos originales y estimados por el modelo: ')
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
#dataset=np.exp(dataset)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

model.summary()



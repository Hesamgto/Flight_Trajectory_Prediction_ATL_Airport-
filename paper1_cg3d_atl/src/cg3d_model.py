#%% Starting Tensorflow

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

#%% Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from numba import njit

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras
from keras.layers import Input, GRU, LSTM, MaxPooling1D, BatchNormalization
from keras.layers import Dropout, Dense, Flatten, Conv1D
from keras.layers import Conv3D, MaxPool3D, GlobalAveragePooling3D
from keras.models import Model
from keras.layers import concatenate
from keras.utils import plot_model
from keras.optimizers import Adam
from tensorflow.keras import regularizers

#%% Part 1 Data Preprocessing
#%% Importing the Dataset

DATA_PATH = 'data/sample_flight_data.csv'  # point this at the real ATL ADS-B extract

col_list = ['time', 'lat', 'lon', 'heading', 'velocity', 'vertrate', 'hour']

dataset = pd.read_csv(DATA_PATH, usecols=col_list, low_memory=False)

#%% Taking care of missing data in dataset with CubicSpline interpolation method

# NOTE: pandas' copy-on-write means dataset['col'] hands back an independent
# copy, so interpolating that copy "inplace" never touched dataset itself --
# the fill has to be assigned back explicitly or it's silently a no-op.

time = dataset['time'].interpolate(method='cubicspline', limit_direction='both')
dataset['time'] = time

lat = dataset['lat'].interpolate(method='cubicspline', limit_direction='both')
dataset['lat'] = lat

lon = dataset['lon'].interpolate(method='cubicspline', limit_direction='both')
dataset['lon'] = lon

velocity = dataset['velocity'].interpolate(method='cubicspline', limit_direction='both')
dataset['velocity'] = velocity

heading = dataset['heading'].interpolate(method='cubicspline', limit_direction='both')
dataset['heading'] = heading

vertrate = dataset['vertrate']
vertrate[0:1] = 0
vertrate = vertrate.interpolate(method='cubicspline', limit_direction='both')
dataset['vertrate'] = vertrate

hour = dataset['hour'].interpolate(method='cubicspline', limit_direction='both')
dataset['hour'] = hour

#%% Part 2 Methodology
#%% Categorize the datasrt into two groups: 1. spatial, 2. temporal

Sp = dataset[["lat", "lon", "heading"]]
Sp = Sp.apply(pd.to_numeric, errors='coerce')
Sp = Sp.to_numpy(dtype='float')

Tp = dataset[["time", "velocity", "vertrate", "hour"]]
Tp = Tp.apply(pd.to_numeric, errors='coerce')
Tp = Tp.to_numpy(dtype='float')

dataset = dataset.apply(pd.to_numeric, errors='coerce')
dataset = dataset.to_numpy(dtype='float')

#%% Sliding Window and Splitting the dataset

@njit
def sliding_window_Sp(data, window_size, step_size):
    X_Sp = []
    Y_Sp = []

    for i in range(window_size, len(data)):
        X_Sp_b = data[i-window_size: i]
        Y_Sp_b = data[i: i+1]
        X_Sp.append(X_Sp_b)
        Y_Sp.append(Y_Sp_b)

    return X_Sp, Y_Sp

Sp = np.array(Sp, dtype='float')
data = Sp

window_size = 100
step_size = 5

X = sliding_window_Sp(data, window_size, step_size)

X_Sp = []
Y_Sp = []

for i in range(np.shape(X[0])[0]):
    a = np.reshape(X[0][i], (1, -1))
    X_Sp.append(a)
    Y_Sp.append(X[1][i])

Y_Sp = np.array(Y_Sp, dtype='float')
X_Sp = np.array(X_Sp, dtype='float')

X_Sp_train, X_Sp_test, Y_Sp_train, Y_Sp_test = train_test_split(X_Sp, Y_Sp, test_size=0.2, random_state=1, shuffle=True)

@njit
def sliding_window_Tp(data, window_size, step_size):
    X_Tp = []
    Y_Tp = []

    for i in range(window_size, len(data)):
        X_Tp_b = data[i-window_size: i]
        Y_Tp_b = data[i: i+1]
        X_Tp.append(X_Tp_b)
        Y_Tp.append(Y_Tp_b)

    return X_Tp, Y_Tp

Tp = np.array(Tp, dtype='float')
data = Tp

window_size = 100
step_size = 5

X = sliding_window_Tp(data, window_size, step_size)

X_Tp = []
Y_Tp = []

for i in range(np.shape(X[0])[0]):
    a = np.reshape(X[0][i], (1, -1))
    X_Tp.append(a)
    Y_Tp.append(X[1][i])

Y_Tp = np.array(Y_Tp, dtype='float')
X_Tp = np.array(X_Tp, dtype='float')

X_Tp_train, X_Tp_test, Y_Tp_train, Y_Tp_test = train_test_split(X_Tp, Y_Tp, test_size=0.2, random_state=1, shuffle=True)

@njit
def sliding_window_dataset(data, window_size, step_size):
    X = []
    Y = []

    for i in range(window_size, len(data)):
        X_b = data[i-window_size: i]
        Y_b = data[i: i+1]
        X.append(X_b)
        Y.append(Y_b)

    return X, Y

dataset = np.array(dataset, dtype='float')
data = dataset

window_size = 100
step_size = 5

X_data = sliding_window_dataset(data, window_size, step_size)

X = []
Y = []

for i in range(np.shape(X_data[0])[0]):
    a = np.reshape(X_data[0][i], (1, -1))
    X.append(a)
    Y.append(X_data[1][i])

Y = np.array(Y, dtype='float')
X = np.array(X, dtype='float')

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1, shuffle=True)

#%% Speeding up the model with PCA method

scaler = StandardScaler()

scaler.fit(X_train.reshape(X_train.shape[0], X_train.shape[2]))
X_train = scaler.transform(X_train.reshape(X_train.shape[0], X_train.shape[2]))
X_test = scaler.transform(X_test.reshape(X_test.shape[0], X_test.shape[2]))

scaler.fit(Y_train.reshape(Y_train.shape[0], Y_train.shape[2]))
Y_train = scaler.transform(Y_train.reshape(Y_train.shape[0], Y_train.shape[2]))
Y_test = scaler.transform(Y_test.reshape(Y_test.shape[0], Y_test.shape[2]))

scaler.fit(X_Sp_train.reshape(X_Sp_train.shape[0], X_Sp_train.shape[2]))
X_Sp_train = scaler.transform(X_Sp_train.reshape(X_Sp_train.shape[0], X_Sp_train.shape[2]))
X_Sp_test = scaler.transform(X_Sp_test.reshape(X_Sp_test.shape[0], X_Sp_test.shape[2]))

scaler.fit(Y_Sp_train.reshape(Y_Sp_train.shape[0], Y_Sp_train.shape[2]))
Y_Sp_train = scaler.transform(Y_Sp_train.reshape(Y_Sp_train.shape[0], Y_Sp_train.shape[2]))
Y_Sp_test = scaler.transform(Y_Sp_test.reshape(Y_Sp_test.shape[0], Y_Sp_test.shape[2]))

scaler.fit(X_Tp_train.reshape(X_Tp_train.shape[0], X_Tp_train.shape[2]))
X_Tp_train = scaler.transform(X_Tp_train.reshape(X_Tp_train.shape[0], X_Tp_train.shape[2]))
X_Tp_test = scaler.transform(X_Tp_test.reshape(X_Tp_test.shape[0], X_Tp_test.shape[2]))

scaler.fit(Y_Tp_train.reshape(Y_Tp_train.shape[0], Y_Tp_train.shape[2]))
Y_Tp_train = scaler.transform(Y_Tp_train.reshape(Y_Tp_train.shape[0], Y_Tp_train.shape[2]))
Y_Tp_test = scaler.transform(Y_Tp_test.reshape(Y_Tp_test.shape[0], Y_Tp_test.shape[2]))

pca = PCA(.95)

pca.fit(X_train.reshape(X_train.shape[0], X_train.shape[1]))
X_train = pca.transform(X_train.reshape(X_train.shape[0], X_train.shape[1]))
X_test = pca.transform(X_test.reshape(X_test.shape[0], X_test.shape[1]))

pca.fit(Y_train.reshape(Y_train.shape[0], Y_train.shape[1]))
Y_train = pca.transform(Y_train)
Y_test = pca.transform(Y_test)

pca.fit(X_Sp_train.reshape(X_Sp_train.shape[0], X_Sp_train.shape[1]))
X_Sp_train = pca.transform(X_Sp_train)
X_Sp_test = pca.transform(X_Sp_test.reshape(X_Sp_test.shape[0], X_Sp_test.shape[1]))

pca.fit(Y_Sp_train.reshape(Y_Sp_train.shape[0], Y_Sp_train.shape[1]))
Y_Sp_train = pca.transform(Y_Sp_train)
Y_Sp_test = pca.transform(Y_Sp_test.reshape(Y_Sp_test.shape[0], Y_Sp_test.shape[1]))

pca.fit(X_Tp_train.reshape(X_Tp_train.shape[0], X_Tp_train.shape[1]))
X_Tp_train = pca.transform(X_Tp_train.reshape(X_Tp_train.shape[0], X_Tp_train.shape[1]))
X_Tp_test = pca.transform(X_Tp_test.reshape(X_Tp_test.shape[0], X_Tp_test.shape[1]))

pca.fit(Y_Tp_train.reshape(Y_Tp_train.shape[0], Y_Tp_train.shape[1]))
Y_Tp_train = pca.transform(Y_Tp_train)
Y_Tp_test = pca.transform(Y_Tp_test.reshape(Y_Tp_test.shape[0], Y_Tp_test.shape[1]))

output_dim = Y_train.shape[1]  # however many components PCA keeps to explain 95% variance

#%% CNN-GRU

X_Sp_train = X_Sp_train.reshape((X_Sp_train.shape[0], 1, X_Sp_train.shape[1]))
Y_Sp_train = Y_Sp_train.reshape((Y_Sp_train.shape[0], 1, Y_Sp_train.shape[1]))

X_Sp_test = X_Sp_test.reshape((X_Sp_test.shape[0], 1, X_Sp_test.shape[1]))
Y_Sp_test = Y_Sp_test.reshape((Y_Sp_test.shape[0], 1, Y_Sp_test.shape[1]))

X_Tp_train = X_Tp_train.reshape((X_Tp_train.shape[0], 1, X_Tp_train.shape[1]))
Y_Tp_train = Y_Tp_train.reshape((Y_Tp_train.shape[0], 1, Y_Tp_train.shape[1]))

X_Tp_test = X_Tp_test.reshape((X_Tp_test.shape[0], 1, X_Tp_test.shape[1]))
Y_Tp_test = Y_Tp_test.reshape((Y_Tp_test.shape[0], 1, Y_Tp_test.shape[1]))

# CNN part of CNN-GRU

visible11 = Input(shape=(1, X_Sp_train.shape[2]))
conv11 = Conv1D(filters=32, kernel_size=3, padding='same', kernel_initializer='he_uniform')(visible11)
pool11 = MaxPooling1D(3, padding='same')(conv11)
act11 = keras.layers.ReLU()(pool11)
batch11 = BatchNormalization()(act11)

conv12 = Conv1D(filters=64, kernel_size=3, padding='same', kernel_initializer='he_uniform')(batch11)
pool12 = MaxPooling1D(3, padding='same')(conv12)
act12 = keras.layers.ReLU()(pool12)
batch12 = BatchNormalization()(act12)
drop12 = Dropout(0.1)(batch12)

conv13 = Conv1D(filters=64, kernel_size=3, padding='same', kernel_initializer='he_uniform')(drop12)
pool13 = MaxPooling1D(3, padding='same')(conv13)
act13 = keras.layers.ReLU()(pool13)
batch13 = BatchNormalization()(act13)
drop13 = Dropout(0.1)(batch13)

hidden11 = Dense(128, kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l2(0.01))(drop13)
act14 = keras.layers.ReLU()(hidden11)
batch14 = BatchNormalization()(act14)
flat14 = Flatten()(batch14)

output11 = Dense(output_dim, activation='linear')(flat14)
flat12 = Flatten()(output11)
model_1 = Model(inputs=visible11, outputs=output11)
model_1.summary()

# GRU part of CNN-GRU

visible21 = Input(shape=(1, X_Tp_train.shape[2]))
gru21 = GRU(20, return_sequences=True)(visible21)
batch21 = BatchNormalization()(gru21)
drop21 = Dropout(0.1)(batch21)
output21 = Dense(output_dim, activation='linear')(drop21)
flat22 = Flatten()(output21)
model_2 = Model(inputs=visible21, outputs=output21)
model_2.summary()

# Merge the CNN and GRU to build CNN-GRU model

merge = concatenate([flat12, flat22])

hidden31 = Dense(512, kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l2(0.01))(merge)
act31 = keras.layers.ReLU()(hidden31)
batch31 = BatchNormalization()(act31)
drop31 = Dropout(0.2)(batch31)

output31 = Dense(output_dim, activation='linear')(drop31)
model_3 = Model(inputs=[visible11, visible21], outputs=output31)
model_3.summary()

# Training the modle

model_3.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae', keras.metrics.RootMeanSquaredError()])

history_callback_1 = model_3.fit([X_Sp_train, X_Tp_train], Y_train, epochs=500, batch_size=512, validation_data=([X_Sp_test, X_Tp_test], Y_test), shuffle=False)

# Plot graph

plot_model(model_3, to_file='CNN-GRU_model.png', show_shapes=True, dpi=96)

# Model prediction

CNN_GRU_pred = model_3.predict([X_Sp_test, X_Tp_test])

# Saving the model

model_3.save('cnn_gru.keras')

# loss/mae/rmse history to csv

history_1 = pd.DataFrame(history_callback_1.history)
history_1.to_csv('history_CNN-GRU.csv', index_label='epoch')

# Plot the Figures

fig, ax = plt.subplots()
ax.plot(history_callback_1.history['loss'])
ax.plot(history_callback_1.history['mae'])
ax.plot(history_callback_1.history['val_loss'])
ax.plot(history_callback_1.history['val_mae'])
ax.legend(["Loss", "MAE", "Val_loss", "Val_MAE"], loc="upper right")
plt.savefig('CNN-GRU_loss_mae.png')
plt.close(fig)

#%% GRU model

X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
Y_train = Y_train.reshape(Y_train.shape[0], 1, Y_train.shape[1])

X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
Y_test = Y_test.reshape(Y_test.shape[0], 1, Y_test.shape[1])

visible41 = Input(shape=(1, X_train.shape[2]))
gru41 = GRU(32, return_sequences=True)(visible41)
batch41 = BatchNormalization()(gru41)
output41 = Dense(output_dim, activation='linear')(batch41)
flat42 = Flatten()(output41)

model_4 = Model(inputs=visible41, outputs=output41)
model_4.summary()

model_4.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae', keras.metrics.RootMeanSquaredError()])

history_callback_2 = model_4.fit(X_train, Y_train, epochs=500, batch_size=512, shuffle=False, validation_data=(X_test, Y_test))

# Plot graph

plot_model(model_4, to_file='GRU_model.png', show_shapes=True, dpi=96)

# Prediction

GRU_pred = model_4.predict(X_test)

history_2 = pd.DataFrame(history_callback_2.history)
history_2.to_csv('history_GRU.csv', index_label='epoch')

# Plot the results

fig, ax = plt.subplots()
ax.plot(history_callback_2.history['mae'])
ax.plot(history_callback_2.history['val_mae'])
ax.plot(history_callback_2.history['loss'])
ax.plot(history_callback_2.history['val_loss'])
plt.title('GRU_model')
plt.xlabel('epoch')
plt.ylabel('mae')
plt.legend(['mae', 'val_mae', 'loss', 'val_loss'], loc='upper left')
plt.savefig('GRU_history.png')
plt.close(fig)

#%% LSTM model

visible51 = Input(shape=(1, X_train.shape[2]))
lstm51 = LSTM(32, return_sequences=True)(visible51)
batch51 = BatchNormalization()(lstm51)
output51 = Dense(output_dim, activation='linear')(batch51)
flat52 = Flatten()(output51)

model_5 = Model(inputs=visible51, outputs=output51)
model_5.summary()

model_5.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae', keras.metrics.RootMeanSquaredError()])

history_callback_3 = model_5.fit(X_train, Y_train, epochs=500, batch_size=512, shuffle=False, validation_data=(X_test, Y_test))

# Plot graph

plot_model(model_5, to_file='LSTM_model.png', show_shapes=True, dpi=96)

# Prediction

LSTM_pred = model_5.predict(X_test)

history_3 = pd.DataFrame(history_callback_3.history)
history_3.to_csv('history_LSTM.csv', index_label='epoch')

# plot of the results

fig, ax = plt.subplots()
ax.plot(history_callback_3.history['mae'])
ax.plot(history_callback_3.history['val_mae'])
ax.plot(history_callback_3.history['loss'])
ax.plot(history_callback_3.history['val_loss'])
plt.title('LSTM_model')
plt.xlabel('epoch')
plt.ylabel('mae')
plt.legend(['mae', 'val_mae', 'loss', 'val_loss'], loc='upper left')
plt.savefig('LSTM_history.png')
plt.close(fig)

#%% MLP model

visible61 = Input(shape=(1, X_train.shape[2]))
hidden61 = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l2(0.01))(visible61)
batch61 = BatchNormalization()(hidden61)
output61 = Dense(output_dim, activation='linear')(batch61)
flat62 = Flatten()(output61)

model_6 = Model(inputs=visible61, outputs=output61)
model_6.summary()

# Model Training

model_6.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['mae', keras.metrics.RootMeanSquaredError()])

history_callback_4 = model_6.fit(X_train, Y_train, batch_size=512, epochs=500, verbose=2, validation_data=(X_test, Y_test))

# Plot graph

plot_model(model_6, to_file='MLP_model.png', show_shapes=True, dpi=96)

# Prediction

MLP_pred = model_6.predict(X_test)

history_4 = pd.DataFrame(history_callback_4.history)
history_4.to_csv('history_MLP.csv', index_label='epoch')

# Plot training and validation accuracy and loss

fig, ax = plt.subplots()
ax.plot(history_callback_4.history['mae'])
ax.plot(history_callback_4.history['val_mae'])
ax.plot(history_callback_4.history['loss'])
ax.plot(history_callback_4.history['val_loss'])
plt.title('MLP_model')
plt.xlabel('epoch')
plt.ylabel('mae')
plt.legend(['mae', 'val_mae', 'loss', 'val_loss'], loc='upper left')
plt.savefig('MLP_history.png')
plt.close(fig)

#%% C3D

# Introduce the channel dimention in the input dataset

X_train = X_train.reshape(X_train.shape[0], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[2])

Xtrain = np.ndarray((X_train.shape[0], X_train.shape[1], 3))
Xtest = np.ndarray((X_test.shape[0], X_test.shape[1], 3))

# iterate in train and test, add the 3rd dimention which in this time series problem is time

def add_3rd_dimention(array):
    scaler_map = cm.ScalarMappable(cmap="Oranges")
    array = scaler_map.to_rgba(array)[:, :-1]
    return array

for i in range(X_train.shape[0]):
    Xtrain[i] = add_3rd_dimention(X_train[i])
for i in range(X_test.shape[0]):
    Xtest[i] = add_3rd_dimention(X_test[i])

# Convert to 1 + 4D space (1st argument represents number of rows in the dataset)

Xtrain = Xtrain.reshape(X_train.shape[0], X_train.shape[1], 1, 1, 3)
Xtest = Xtest.reshape(X_test.shape[0], X_test.shape[1], 1, 1, 3)

Y_train = Y_train.reshape(Y_train.shape[0], Y_train.shape[2])
Y_test = Y_test.reshape(Y_test.shape[0], Y_test.shape[2])

# Model

visible71 = Input(shape=(X_train.shape[1], 1, 1, 3))
conv71 = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', kernel_initializer='he_uniform')(visible71)
pool71 = MaxPool3D(pool_size=(2, 2, 2), padding='same')(conv71)
act71 = keras.layers.ReLU()(pool71)
batch71 = BatchNormalization()(act71)

conv72 = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', kernel_initializer='he_uniform')(batch71)
pool72 = MaxPool3D(pool_size=(2, 2, 2), padding='same')(conv72)
act72 = keras.layers.ReLU()(pool72)
batch72 = BatchNormalization()(act72)

conv73 = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', kernel_initializer='he_uniform')(batch72)
pool73 = MaxPool3D(pool_size=(2, 2, 2), padding='same')(conv73)
act73 = keras.layers.ReLU()(pool73)
batch73 = BatchNormalization()(act73)

conv74 = Conv3D(filters=128, kernel_size=(3, 3, 3), padding='same', kernel_initializer='he_uniform')(batch73)
pool74 = MaxPool3D(pool_size=(2, 2, 2), padding='same')(conv74)
act74 = keras.layers.ReLU()(pool74)
batch74 = BatchNormalization()(act74)

conv75 = Conv3D(filters=128, kernel_size=(3, 3, 3), padding='same', kernel_initializer='he_uniform')(batch74)
pool75 = MaxPool3D(pool_size=(2, 2, 2), padding='same')(conv75)
act75 = keras.layers.ReLU()(pool75)
batch75 = BatchNormalization()(act75)

conv76 = Conv3D(filters=256, kernel_size=(3, 3, 3), padding='same', kernel_initializer='he_uniform')(batch75)
pool76 = MaxPool3D(pool_size=(2, 2, 2), padding='same')(conv76)
act76 = keras.layers.ReLU()(pool76)
batch76 = BatchNormalization()(act76)

conv77 = Conv3D(filters=256, kernel_size=(3, 3, 3), padding='same', kernel_initializer='he_uniform')(batch76)
pool77 = MaxPool3D(pool_size=(2, 2, 2), padding='same')(conv77)
act77 = keras.layers.ReLU()(pool77)
batch77 = BatchNormalization()(act77)
drop77 = Dropout(0.1)(batch77)

pool78 = GlobalAveragePooling3D()(drop77)
hidden72 = Dense(512, kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l2(0.01))(pool78)

drop74 = Dropout(0.1)(hidden72)

output71 = Dense(units=output_dim, activation='linear')(drop74)

model_7 = Model(inputs=visible71, outputs=output71)
model_7.summary()

# Model Training

model_7.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['mae', keras.metrics.RootMeanSquaredError()])

history_callback_5 = model_7.fit(Xtrain, Y_train, batch_size=512, epochs=500, validation_data=(Xtest, Y_test), shuffle=False)

# Plot graph

plot_model(model_7, to_file='C3D_model.png', show_shapes=True, dpi=96)

# Prediction

C3D_pred = model_7.predict(Xtest)

history_5 = pd.DataFrame(history_callback_5.history)
history_5.to_csv('history_C3D.csv', index_label='epoch')

fig, ax = plt.subplots()
ax.plot(history_callback_5.history['loss'])
ax.plot(history_callback_5.history['root_mean_squared_error'])
ax.set_title('Loss and RMSE for C3D')
ax.legend(["Loss", "RMSE"], loc="upper right")
plt.savefig('C3D_loss_rmse.png')
plt.close(fig)

#%% CRC3D : Combine the CNN-GRU and C3D and probably name it CRC3D

# CNN part of CNN-GRU for CRC3D
# (X_Sp_train / X_Tp_train are already (N, 1, F) from the CNN-GRU section
# above -- no need to reshape them again here.)

visible81 = Input(shape=(1, X_Sp_train.shape[2]))
conv81 = Conv1D(filters=32, kernel_size=6, padding='same', kernel_initializer='he_uniform')(visible81)
pool81 = MaxPooling1D(3, padding='same')(conv81)
act81 = keras.layers.ReLU()(pool81)
batch81 = BatchNormalization()(act81)

conv82 = Conv1D(filters=64, kernel_size=3, padding='same', kernel_initializer='he_uniform')(batch81)
pool82 = MaxPooling1D(3, padding='same')(conv82)
act82 = keras.layers.ReLU()(pool82)
batch82 = BatchNormalization()(act82)
drop82 = Dropout(0.1)(batch82)

conv83 = Conv1D(filters=128, kernel_size=3, padding='same', kernel_initializer='he_uniform')(drop82)
pool83 = MaxPooling1D(3, padding='same')(conv83)
act83 = keras.layers.ReLU()(pool83)
batch83 = BatchNormalization()(act83)
drop83 = Dropout(0.1)(batch83)

hidden81 = Dense(256, kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l2(0.01))(drop83)
act84 = keras.layers.ReLU()(hidden81)
batch84 = BatchNormalization()(act84)
flat81 = Flatten()(batch84)

output81 = Dense(output_dim, activation='linear')(flat81)
flat82 = Flatten()(output81)
model_8 = Model(inputs=visible81, outputs=output81)
model_8.summary()

# GRU part of CNN-GRU for CRC3D

visible91 = Input(shape=(1, X_Tp_train.shape[2]))
gru91 = GRU(20, return_sequences=True)(visible91)
batch91 = BatchNormalization()(gru91)
drop91 = Dropout(0.1)(batch91)
output91 = Dense(output_dim, activation='linear')(drop91)
flat92 = Flatten()(output91)
model_9 = Model(inputs=visible91, outputs=output91)
model_9.summary()

# Merge the CNN and GRU to build CNN-GRU model

merge = concatenate([flat82, flat92])

hidden101 = Dense(512, kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l2(0.01))(merge)
act101 = keras.layers.ReLU()(hidden101)
batch101 = BatchNormalization()(act101)
drop101 = Dropout(0.2)(batch101)

output101 = Dense(output_dim, activation='linear')(drop101)
flat101 = Flatten()(output101)
model_10 = Model(inputs=[visible81, visible91], outputs=output101)
model_10.summary()

# Plot graph

plot_model(model_10, to_file='CNN-GRU_1_model.png', show_shapes=True, dpi=96)

# C3D for CRC3D

# Introduce the channel dimention in the input dataset

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])

Xtrain = np.ndarray((X_train.shape[0], X_train.shape[1], 3))
Xtest = np.ndarray((X_test.shape[0], X_test.shape[1], 3))

for i in range(X_train.shape[0]):
    Xtrain[i] = add_3rd_dimention(X_train[i])
for i in range(X_test.shape[0]):
    Xtest[i] = add_3rd_dimention(X_test[i])

Xtrain = Xtrain.reshape(X_train.shape[0], X_train.shape[1], 1, 1, 3)
Xtest = Xtest.reshape(X_test.shape[0], X_test.shape[1], 1, 1, 3)

# Model

visible111 = Input(shape=(X_train.shape[1], 1, 1, 3))
conv111 = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', kernel_initializer='he_uniform')(visible111)
pool111 = MaxPool3D(pool_size=(2, 2, 2), padding='same')(conv111)
act111 = keras.layers.ReLU()(pool111)
batch111 = BatchNormalization()(act111)

conv112 = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', kernel_initializer='he_uniform')(batch111)
pool112 = MaxPool3D(pool_size=(2, 2, 2), padding='same')(conv112)
act112 = keras.layers.ReLU()(pool112)
batch112 = BatchNormalization()(act112)
drop112 = Dropout(0.1)(batch112)

conv113 = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', kernel_initializer='he_uniform')(drop112)
pool113 = MaxPool3D(pool_size=(2, 2, 2), padding='same')(conv113)
act113 = keras.layers.ReLU()(pool113)
batch113 = BatchNormalization()(act113)
drop113 = Dropout(0.1)(batch113)

conv114 = Conv3D(filters=128, kernel_size=(3, 3, 3), padding='same', kernel_initializer='he_uniform')(drop113)
pool114 = MaxPool3D(pool_size=(2, 2, 2), padding='same')(conv114)
act114 = keras.layers.ReLU()(pool114)
batch114 = BatchNormalization()(act114)
drop114 = Dropout(0.1)(batch114)

conv115 = Conv3D(filters=128, kernel_size=(3, 3, 3), padding='same', kernel_initializer='he_uniform')(drop114)
pool115 = MaxPool3D(pool_size=(2, 2, 2), padding='same')(conv115)
act115 = keras.layers.ReLU()(pool115)
batch115 = BatchNormalization()(act115)

conv116 = Conv3D(filters=256, kernel_size=(3, 3, 3), padding='same', kernel_initializer='he_uniform')(batch115)
pool116 = MaxPool3D(pool_size=(2, 2, 2), padding='same')(conv116)
act116 = keras.layers.ReLU()(pool116)
batch116 = BatchNormalization()(act116)
drop116 = Dropout(0.2)(batch116)

conv117 = Conv3D(filters=256, kernel_size=(3, 3, 3), padding='same', kernel_initializer='he_uniform')(drop116)
pool117 = MaxPool3D(pool_size=(2, 2, 2), padding='same')(conv117)
act117 = keras.layers.ReLU()(pool117)
batch117 = BatchNormalization()(act117)
drop117 = Dropout(0.2)(batch117)

pool118 = GlobalAveragePooling3D()(drop117)
hidden111 = Dense(units=512, kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l2(0.01))(pool118)
act118 = keras.layers.ReLU()(hidden111)
batch118 = BatchNormalization()(act118)
drop118 = Dropout(0.25)(batch118)

output111 = Dense(units=output_dim, activation='linear')(drop118)
flat112 = Flatten()(output111)

model_11 = Model(inputs=visible111, outputs=output111)
model_11.summary()

# CRC3D model

merge2 = concatenate([flat101, flat112])

hidden121 = Dense(512, kernel_regularizer=regularizers.l1(0.01), activity_regularizer=regularizers.L1(0.01))(merge2)
act121 = keras.layers.ReLU()(hidden121)
batch121 = BatchNormalization()(act121)
drop121 = Dropout(0.3)(batch121)

output121 = Dense(output_dim, activation='linear')(drop121)
model_12 = Model(inputs=[visible81, visible91, visible111], outputs=output121)
model_12.summary()

# Training the modle

model_12.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['mae', keras.metrics.RootMeanSquaredError()])

history_callback_6 = model_12.fit([X_Sp_train, X_Tp_train, Xtrain], Y_train, epochs=500, batch_size=512, validation_data=([X_Sp_test, X_Tp_test, Xtest], Y_test), shuffle=False)

# Plot graph

plot_model(model_12, to_file='CRC3D_model.png', show_shapes=True, dpi=96)

# Model prediction

CRC3D_pred = model_12.predict([X_Sp_test, X_Tp_test, Xtest])

# Saving the model

model_12.save('CRC3D.keras')

history_6 = pd.DataFrame(history_callback_6.history)
history_6.to_csv('history_CRC3D.csv', index_label='epoch')

fig, ax = plt.subplots()
ax.plot(history_callback_6.history['loss'])
ax.plot(history_callback_6.history['mae'])
ax.set_title('Loss and Mean Absolute Error for CRC3D')
ax.legend(["Loss", "Mean Absolute Error"], loc="upper right")
plt.savefig('CRC3D_loss_mae.png')
plt.close(fig)

#%% CNN model

X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

Y_train = Y_train.reshape(Y_train.shape[0], 1, Y_train.shape[1])
Y_test = Y_test.reshape(Y_test.shape[0], 1, Y_test.shape[1])

visible131 = Input(shape=(1, X_train.shape[2]))
conv131 = Conv1D(filters=32, kernel_size=6, padding='same', kernel_initializer='he_uniform')(visible131)
pool131 = MaxPooling1D(3, padding='same')(conv131)
act131 = keras.layers.ReLU()(pool131)
batch131 = BatchNormalization()(act131)

conv132 = Conv1D(filters=64, kernel_size=3, padding='same', kernel_initializer='he_uniform')(batch131)
pool132 = MaxPooling1D(3, padding='same')(conv132)
act132 = keras.layers.ReLU()(pool132)
batch132 = BatchNormalization()(act132)

conv133 = Conv1D(filters=128, kernel_size=3, padding='same', kernel_initializer='he_uniform')(batch132)
pool133 = MaxPooling1D(3, padding='same')(conv133)
act133 = keras.layers.ReLU()(pool133)
batch133 = BatchNormalization()(pool133)

hidden131 = Dense(256, kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l2(0.01))(batch133)
act134 = keras.layers.ReLU()(hidden131)
batch134 = BatchNormalization()(act134)
drop134 = Dropout(0.1)(batch134)

output131 = Dense(output_dim, activation='linear')(drop134)
flat132 = Flatten()(output131)
model_13 = Model(inputs=visible131, outputs=output131)
model_13.summary()

# Training the modle

model_13.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['mae', keras.metrics.RootMeanSquaredError()])

history_callback_7 = model_13.fit(X_train, Y_train, epochs=500, batch_size=512, validation_data=(X_test, Y_test), shuffle=False)

# Plot graph

plot_model(model_13, to_file='CNN_model.png', show_shapes=True, dpi=96)

# Model prediction

CNN_pred = model_13.predict(X_test)

model_13.save('CNN.keras')

history_7 = pd.DataFrame(history_callback_7.history)
history_7.to_csv('history_CNN.csv', index_label='epoch')

#%% apply the uncertanty by means of Bayesian method with Monte Carlo Dropout for CRC3D

plt.style.use("ggplot")

class MCDropout(Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)

# CRC3D with MC Dropout : Combine the CNN-GRU and C3D and probably name it CRC3D

# CNN part of CNN-GRU for CRC3D

visible141 = Input(shape=(1, X_Sp_train.shape[2]))
conv141 = Conv1D(filters=32, kernel_size=6, padding='same', kernel_initializer='he_uniform')(visible141)
pool141 = MaxPooling1D(3, padding='same')(conv141)
act141 = keras.layers.ReLU()(pool141)
batch141 = BatchNormalization()(act141)

conv142 = Conv1D(filters=64, kernel_size=3, padding='same', kernel_initializer='he_uniform')(act141)
pool142 = MaxPooling1D(3, padding='same')(conv142)
act142 = keras.layers.ReLU()(pool142)
batch142 = BatchNormalization()(act142)
drop142 = MCDropout(0.1)(batch142)

conv143 = Conv1D(filters=64, kernel_size=3, padding='same', kernel_initializer='he_uniform')(drop142)
pool143 = MaxPooling1D(3, padding='same')(conv143)
act143 = keras.layers.ReLU()(pool143)
batch143 = BatchNormalization()(act143)
drop143 = MCDropout(0.2)(batch143)

hidden141 = Dense(128, kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l2(0.01))(drop143)
act144 = keras.layers.ReLU()(hidden141)
batch144 = BatchNormalization()(act144)
flat141 = Flatten()(batch144)

output141 = Dense(output_dim, activation='linear')(flat141)
flat142 = Flatten()(output141)
model_14 = Model(inputs=visible141, outputs=output141)
model_14.summary()

# GRU part of CNN-GRU for CRC3D

visible151 = Input(shape=(1, X_Tp_train.shape[2]))
gru151 = GRU(20, return_sequences=True)(visible151)
batch151 = BatchNormalization()(gru151)
drop151 = MCDropout(0.1)(batch151)
output151 = Dense(output_dim, activation='linear')(drop151)
flat152 = Flatten()(output151)
model_15 = Model(inputs=visible151, outputs=output151)
model_15.summary()

# Merge the CNN and GRU to build CNN-GRU model

merge3 = concatenate([flat142, flat152])

hidden161 = Dense(512, kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l2(0.01))(merge3)
act161 = keras.layers.ReLU()(hidden161)
batch161 = BatchNormalization()(act161)
drop161 = MCDropout(0.5)(batch161)

output161 = Dense(output_dim, activation='linear')(drop161)
flat161 = Flatten()(output161)
model_16 = Model(inputs=[visible141, visible151], outputs=output161)
model_16.summary()

# Plot graph

plot_model(model_16, to_file='CNN-GRU_MC_model.png', show_shapes=True, dpi=96)

# C3D for CRC3D

X_train = X_train.reshape(X_train.shape[0], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[2])

Xtrain = np.ndarray((X_train.shape[0], X_train.shape[1], 3))
Xtest = np.ndarray((X_test.shape[0], X_test.shape[1], 3))

for i in range(X_train.shape[0]):
    Xtrain[i] = add_3rd_dimention(X_train[i])
for i in range(X_test.shape[0]):
    Xtest[i] = add_3rd_dimention(X_test[i])

Xtrain = Xtrain.reshape(X_train.shape[0], X_train.shape[1], 1, 1, 3)
Xtest = Xtest.reshape(X_test.shape[0], X_test.shape[1], 1, 1, 3)

Y_train = Y_train.reshape(Y_train.shape[0], Y_train.shape[2])
Y_test = Y_test.reshape(Y_test.shape[0], Y_test.shape[2])

# Model

visible171 = Input(shape=(X_train.shape[1], 1, 1, 3))
conv171 = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', kernel_initializer='he_uniform')(visible171)
pool171 = MaxPool3D(pool_size=(2, 2, 2), padding='same')(conv171)
act171 = keras.layers.ReLU()(pool171)
batch171 = BatchNormalization()(act171)

conv172 = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', kernel_initializer='he_uniform')(batch171)
pool172 = MaxPool3D(pool_size=(2, 2, 2), padding='same')(conv172)
act172 = keras.layers.ReLU()(pool172)
batch172 = BatchNormalization()(act172)
drop172 = MCDropout(0.1)(batch172)

conv173 = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', kernel_initializer='he_uniform')(drop172)
pool173 = MaxPool3D(pool_size=(2, 2, 2), padding='same')(conv173)
act173 = keras.layers.ReLU()(pool173)
batch173 = BatchNormalization()(act173)
drop173 = MCDropout(0.1)(batch173)

conv174 = Conv3D(filters=128, kernel_size=(3, 3, 3), padding='same', kernel_initializer='he_uniform')(drop173)
pool174 = MaxPool3D(pool_size=(2, 2, 2), padding='same')(conv174)
act174 = keras.layers.ReLU()(pool174)
batch174 = BatchNormalization()(act174)
drop174 = MCDropout(0.2)(batch174)

conv175 = Conv3D(filters=128, kernel_size=(3, 3, 3), padding='same', kernel_initializer='he_uniform')(drop174)
pool175 = MaxPool3D(pool_size=(2, 2, 2), padding='same')(conv175)
act175 = keras.layers.ReLU()(pool175)
batch175 = BatchNormalization()(act175)

conv176 = Conv3D(filters=256, kernel_size=(3, 3, 3), padding='same', kernel_initializer='he_uniform')(batch175)
pool176 = MaxPool3D(pool_size=(2, 2, 2), padding='same')(conv176)
act176 = keras.layers.ReLU()(pool176)
batch176 = BatchNormalization()(act176)
drop176 = MCDropout(0.2)(batch176)

conv177 = Conv3D(filters=256, kernel_size=(3, 3, 3), padding='same', kernel_initializer='he_uniform')(drop176)
pool177 = MaxPool3D(pool_size=(2, 2, 2), padding='same')(conv177)
act177 = keras.layers.ReLU()(pool177)
batch177 = BatchNormalization()(act177)
drop177 = MCDropout(0.2)(batch177)

pool178 = GlobalAveragePooling3D()(drop177)
hidden171 = Dense(units=512, kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l2(0.01))(pool178)
act178 = keras.layers.ReLU()(hidden171)
batch178 = BatchNormalization()(act178)
drop178 = MCDropout(0.5)(batch178)

output171 = Dense(units=output_dim, activation='linear')(drop178)
flat172 = Flatten()(output171)

model_17 = Model(inputs=visible171, outputs=output171)
model_17.summary()

# Plot graph

plot_model(model_17, to_file='C3D_MC_model.png', show_shapes=True, dpi=96)

# CRC3D model

merge4 = concatenate([flat161, flat172])

hidden181 = Dense(512, kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l2(0.01))(merge4)
act181 = keras.layers.ReLU()(hidden181)
batch181 = BatchNormalization()(act181)
drop181 = MCDropout(0.5)(batch181)

output181 = Dense(output_dim, activation='linear')(drop181)

model_18 = Model(inputs=[visible141, visible151, visible171], outputs=output181)
model_18.summary()

# Training the modle

model_18.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae', keras.metrics.RootMeanSquaredError()])

history_callback_8 = model_18.fit([X_Sp_train, X_Tp_train, Xtrain], Y_train, validation_data=([X_Sp_test, X_Tp_test, Xtest], Y_test), epochs=500, batch_size=512, shuffle=False)

# Plot graph

plot_model(model_18, to_file='CRC3D_MC_model.png', show_shapes=True, dpi=96)

# Model prediction

CRC3D_MC_pred = model_18.predict([X_Sp_test, X_Tp_test, Xtest])

# Saving the model

model_18.save('CRC3D_MC.keras')

history_8 = pd.DataFrame(history_callback_8.history)
history_8.to_csv('history_CRC3D_MC.csv', index_label='epoch')

# Loss and mae plot

fig, ax = plt.subplots()
ax.plot(history_callback_8.history['mae'])
ax.plot(history_callback_8.history['val_mae'])
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend(['mae', 'val_mae'], loc='upper right')
plt.savefig('CRC3D_MC_mae.png')
plt.close(fig)

fig, ax = plt.subplots()
ax.plot(history_callback_8.history['loss'])
ax.plot(history_callback_8.history['val_loss'])
ax.set_title('Loss and Mean Absolute Error for CRC3D with MCDropout')
ax.legend(["Loss", "Val Loss"], loc="upper right")
plt.savefig('CRC3D_MC_loss.png')
plt.close(fig)

#%% Comparing all the models

print('CNN-GRU        -- MAE: %.5f  RMSE: %.5f' % (history_callback_1.history['val_mae'][-1], history_callback_1.history['val_root_mean_squared_error'][-1]))
print('GRU            -- MAE: %.5f  RMSE: %.5f' % (history_callback_2.history['val_mae'][-1], history_callback_2.history['val_root_mean_squared_error'][-1]))
print('LSTM           -- MAE: %.5f  RMSE: %.5f' % (history_callback_3.history['val_mae'][-1], history_callback_3.history['val_root_mean_squared_error'][-1]))
print('MLP            -- MAE: %.5f  RMSE: %.5f' % (history_callback_4.history['val_mae'][-1], history_callback_4.history['val_root_mean_squared_error'][-1]))
print('C3D            -- MAE: %.5f  RMSE: %.5f' % (history_callback_5.history['val_mae'][-1], history_callback_5.history['val_root_mean_squared_error'][-1]))
print('CRC3D (CG3D)   -- MAE: %.5f  RMSE: %.5f' % (history_callback_6.history['val_mae'][-1], history_callback_6.history['val_root_mean_squared_error'][-1]))
print('CNN            -- MAE: %.5f  RMSE: %.5f' % (history_callback_7.history['val_mae'][-1], history_callback_7.history['val_root_mean_squared_error'][-1]))
print('CRC3D + MCDrop -- MAE: %.5f  RMSE: %.5f' % (history_callback_8.history['val_mae'][-1], history_callback_8.history['val_root_mean_squared_error'][-1]))

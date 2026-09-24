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
from scipy.interpolate import CubicSpline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from numba import njit

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras
from keras.layers import Input, GRU, MaxPooling1D, Dropout, Dense, Flatten, Conv1D, BatchNormalization
from keras.models import Model
from keras.layers import concatenate
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.callbacks import History
from tensorflow.keras import regularizers

#%% Part 1 Data Preprocessing
#%% Importing the dataset

DATA_PATH = 'data/sample_flight_data.csv'  # point this at the real ADS-B extract for KATL -> KSTL

col_list = ['time', 'lat', 'lon', 'heading', 'velocity', 'vertrate', 'hour']

dataset = pd.read_csv(DATA_PATH, usecols=col_list, low_memory=False)

dataset.sort_values(by=['time'], inplace=True, ascending=True)

#%% Taking care of missing data with Interpolation

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
#%% Categorizing the dataset

Sp = dataset[["lat", "lon", "heading"]]
Sp = Sp.apply(pd.to_numeric, errors='coerce')
Sp = Sp.to_numpy(dtype='float')

Tp = dataset[["time", "velocity", "vertrate", "hour"]]
Tp = Tp.apply(pd.to_numeric, errors='coerce')
Tp = Tp.to_numpy(dtype='float')

dataset = dataset.apply(pd.to_numeric, errors='coerce')
dataset = dataset.to_numpy(dtype='float')

#%% Sliding Window for Splitted dataset

@njit
def sliding_window_Sp(data, window_size, step_size):
    X_Sp = []
    Y_Sp = []

    for i in range(window_size, len(data)):
        X_Sp_b = data[i - window_size: i]
        Y_Sp_b = data[i: i + 1]
        X_Sp.append(X_Sp_b)
        Y_Sp.append(Y_Sp_b)

    return X_Sp, Y_Sp

Sp = np.array(Sp, dtype='float')
data = Sp

window_size = 50
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
        X_Tp_b = data[i - window_size: i]
        Y_Tp_b = data[i: i + 1]
        X_Tp.append(X_Tp_b)
        Y_Tp.append(Y_Tp_b)

    return X_Tp, Y_Tp

Tp = np.array(Tp, dtype='float')
data = Tp

window_size = 50
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

#%% Sliding Window for one piece dataset input

@njit
def sliding_window_dataset(data, window_size, step_size):
    X = []
    Y = []

    for i in range(window_size, len(data)):
        X_b = data[i - window_size: i]
        Y_b = data[i: i + 1]
        X.append(X_b)
        Y.append(Y_b)

    return X, Y

dataset = np.array(dataset, dtype='float')
data = dataset

window_size = 50
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

#%% Rescaling the dataset for splitted input

scaler = StandardScaler()

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

#%% Rescaling the dataset for one piece input

scaler = StandardScaler()

scaler.fit(X_train.reshape(X_train.shape[0], X_train.shape[2]))
X_train = scaler.transform(X_train.reshape(X_train.shape[0], X_train.shape[2]))
X_test = scaler.transform(X_test.reshape(X_test.shape[0], X_test.shape[2]))

scaler.fit(Y_train.reshape(Y_train.shape[0], Y_train.shape[2]))
Y_train = scaler.transform(Y_train.reshape(Y_train.shape[0], Y_train.shape[2]))
Y_test = scaler.transform(Y_test.reshape(Y_test.shape[0], Y_test.shape[2]))

pca = PCA(.95)

pca.fit(X_train.reshape(X_train.shape[0], X_train.shape[1]))
X_train = pca.transform(X_train.reshape(X_train.shape[0], X_train.shape[1]))
X_test = pca.transform(X_test.reshape(X_test.shape[0], X_test.shape[1]))

pca.fit(Y_train.reshape(Y_train.shape[0], Y_train.shape[1]))
Y_train = pca.transform(Y_train)
Y_test = pca.transform(Y_test)

#%% CNN-GRU with Splitted dataset input

# Reshaping the splitted data

X_Sp_train = X_Sp_train.reshape((X_Sp_train.shape[0], 1, X_Sp_train.shape[1]))
Y_Sp_train = Y_Sp_train.reshape((Y_Sp_train.shape[0], 1, Y_Sp_train.shape[1]))

X_Sp_test = X_Sp_test.reshape((X_Sp_test.shape[0], 1, X_Sp_test.shape[1]))
Y_Sp_test = Y_Sp_test.reshape((Y_Sp_test.shape[0], 1, Y_Sp_test.shape[1]))

X_Tp_train = X_Tp_train.reshape((X_Tp_train.shape[0], 1, X_Tp_train.shape[1]))
Y_Tp_train = Y_Tp_train.reshape((Y_Tp_train.shape[0], 1, Y_Tp_train.shape[1]))

X_Tp_test = X_Tp_test.reshape((X_Tp_test.shape[0], 1, X_Tp_test.shape[1]))
Y_Tp_test = Y_Tp_test.reshape((Y_Tp_test.shape[0], 1, Y_Tp_test.shape[1]))

output_dim = Y_train.shape[1]  # PCA keeps however many components explain 95% variance

# CNN part of CNN-GRU

visible11 = Input(shape=(1, X_Sp_train.shape[2]))
conv11 = Conv1D(filters=256, kernel_size=6, padding='same')(visible11)
pool11 = MaxPooling1D(3, padding='same')(conv11)
batch11 = BatchNormalization()(pool11)
act11 = keras.layers.ReLU()(batch11)

conv12 = Conv1D(filters=512, kernel_size=3, padding='same')(act11)
pool12 = MaxPooling1D(3, padding='same')(conv12)
batch12 = BatchNormalization()(pool12)
act12 = keras.layers.ReLU()(batch12)
drop12 = Dropout(0.1)(act12)

conv13 = Conv1D(filters=256, kernel_size=3, padding='same')(drop12)
pool13 = MaxPooling1D(3, padding='same')(conv13)
batch13 = BatchNormalization()(pool13)
act13 = keras.layers.ReLU()(batch13)
flat11 = Flatten()(act13)

hidden11 = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l2(0.01))(flat11)

output11 = Dense(3, activation='linear')(hidden11)
flat12 = Flatten()(output11)
model_1 = Model(inputs=visible11, outputs=output11)
model_1.summary()

# GRU part of CNN-GRU

visible21 = Input(shape=(1, X_Tp_train.shape[2]))
gru21 = GRU(150, return_sequences=True)(visible21)
output21 = Dense(3, activation='linear')(gru21)
flat22 = Flatten()(output21)
model_2 = Model(inputs=visible21, outputs=output21)
model_2.summary()

# Merge the CNN and GRU to build CNN-GRU model

merge = concatenate([flat12, flat22])

hidden31 = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l2(0.01))(merge)
hidden32 = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l2(0.01))(hidden31)
output31 = Dense(output_dim, activation='linear')(hidden32)
model_3 = Model(inputs=[visible11, visible21], outputs=output31)
model_3.summary()

# Training the modle

model_3.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae', keras.metrics.RootMeanSquaredError()])

history_callback_1 = model_3.fit([X_Sp_train, X_Tp_train], Y_train, epochs=500, batch_size=512, validation_data=([X_Sp_test, X_Tp_test], Y_test), shuffle=False)

# Plot graph

plot_model(model_3, to_file='CNN-GRU_Separated_model.png', show_shapes=True, dpi=96)

# Model prediction

CNN_GRU_pred = model_3.predict([X_Sp_test, X_Tp_test])

# Saving the model

model_3.save('cnn_gru_separated.keras')

# loss and mae/rmse history to csv

history_1 = pd.DataFrame(history_callback_1.history)
history_1.to_csv('history_CNN-GRU_Separated.csv', index_label='epoch')

# Plot

fig, ax = plt.subplots(1, 1)
ax.plot(history_callback_1.history['root_mean_squared_error'])
ax.plot(history_callback_1.history['val_root_mean_squared_error'])
plt.xlabel('Number of Epochs')
plt.ylabel('Error')
plt.legend(['RMSE', 'Validation RMSE'])
plt.savefig('rmse_CNN-GRU_Separated.png')  # savefig instead of show() so this runs headless too
plt.close(fig)

#%% CNN-GRU with One piece dataset Input

# Reshaping the one piece data

X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
Y_train = Y_train.reshape((Y_train.shape[0], 1, Y_train.shape[1]))

X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
Y_test = Y_test.reshape((Y_test.shape[0], 1, Y_test.shape[1]))

# Model

visible41 = Input(shape=(1, X_train.shape[2]))
conv41 = Conv1D(filters=256, kernel_size=6, padding='same')(visible41)
pool41 = MaxPooling1D(3, padding='same')(conv41)
batch41 = BatchNormalization()(pool41)
act41 = keras.layers.ReLU()(batch41)

conv42 = Conv1D(filters=512, kernel_size=3, padding='same')(act41)
pool42 = MaxPooling1D(3, padding='same')(conv42)
batch42 = BatchNormalization()(pool42)
act42 = keras.layers.ReLU()(batch42)

conv43 = Conv1D(filters=256, kernel_size=3, padding='same')(act42)
pool43 = MaxPooling1D(6, padding='same')(conv43)
batch43 = BatchNormalization()(pool43)
act43 = keras.layers.ReLU()(batch43)
drop43 = Dropout(0.1)(act43)

hidden41 = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l2(0.01))(drop43)

gru41 = GRU(128, return_sequences=True)(hidden41)
batch44 = BatchNormalization()(gru41)
drop44 = Dropout(0.35)(batch44)
output41 = Dense(output_dim, activation='linear')(drop44)

model_4 = Model(inputs=visible41, outputs=output41)
model_4.summary()

model_4.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae', keras.metrics.RootMeanSquaredError()])

history_callback_4 = model_4.fit(X_train, Y_train, epochs=500, batch_size=512, validation_data=(X_test, Y_test), shuffle=False)

# Model prediction

CNN_GRU_One_Piece = model_4.predict(X_test)

# Saving the model

model_4.save('cnn_gru_unified.keras')

# loss and mae/rmse history to csv

history_4 = pd.DataFrame(history_callback_4.history)
history_4.to_csv('history_CNN-GRU_Unified.csv', index_label='epoch')

# Plot

fig, ax = plt.subplots(1, 1)
ax.plot(history_callback_4.history['root_mean_squared_error'])
ax.plot(history_callback_4.history['val_root_mean_squared_error'])
plt.xlabel('Number of Epochs')
plt.ylabel('Error')
plt.legend(['RMSE', 'Validation RMSE'])
plt.savefig('rmse_CNN-GRU_Unified.png')
plt.close(fig)

#%% Comparing the two architectures

print('CNN-GRU separated  -- MAE: %.5f  RMSE: %.5f' % (
    history_callback_1.history['val_mae'][-1], history_callback_1.history['val_root_mean_squared_error'][-1]))
print('CNN-GRU unified     -- MAE: %.5f  RMSE: %.5f' % (
    history_callback_4.history['val_mae'][-1], history_callback_4.history['val_root_mean_squared_error'][-1]))

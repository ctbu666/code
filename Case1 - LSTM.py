import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import scipy.io as scio

from keras import backend as K
from keras import regularizers
from keras.layers import Input, Dense, Dropout
from keras.models import Model,Sequential
from keras.utils.np_utils import to_categorical

from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Flatten
from keras.layers import concatenate
from keras.utils import plot_model
from keras.models import load_model

K.clear_session()

# 读取
Data = scio.loadmat('Dataforpython.mat') # 读取mat文件
Trajectory=Data['SOC']
Trajectory=Trajectory.tolist()
Trajectory=Trajectory[0]


window=20

Trajectory_reorg=list()
RUL_reorg=list()
for tra in Trajectory:
    RUL=np.linspace(start = tra.size-1, stop = 0, num = tra.size)
    RUL=RUL[window-1:]
    tra_reorganize=np.zeros((tra.size-window+1,window))
    for i in range(tra.size-window+1):
        tra_reorganize[i,:]=tra[i:i+window].T
    Trajectory_reorg.append(tra_reorganize)
    RUL_reorg.append(RUL)
    del tra_reorganize, RUL


RUL_reorg=Trajectory_reorg
random.seed(58) #初始seed是1
Index_train=random.sample(range(0, len(Trajectory_reorg)),100)
Index_train.sort()

Trajectory_train=[Trajectory_reorg[i] for i in Index_train]
x_train=np.vstack(Trajectory_train)
RUL_train=[RUL_reorg[i] for i in Index_train]
y_train=np.concatenate(RUL_train)

Index_test = list(set(range(0,len(Trajectory_reorg))) ^ set(Index_train))
Trajectory_test=[Trajectory_reorg[i] for i in Index_test]
x_test=np.vstack(Trajectory_test)
RUL_test=[RUL_reorg[i] for i in Index_test]
y_test=np.concatenate(RUL_test)

y_train=y_train[:,1]
y_test=y_test[:,1]


x_train=x_train.reshape(len(x_train),20,1)
x_test=x_test.reshape(len(x_test),20,1)
y_train=y_train.reshape(len(y_train),1)
y_test=y_test.reshape(len(y_test),1)

# Train LSTM and save results
from keras.layers import LSTM, Masking


model = Sequential()
model.add(Masking(mask_value=0., input_shape=(20, 1)))
model.add(LSTM(40,return_sequences=False))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)
cc=model.predict(x_test)

y_test=y_test*100
cc=cc*100
plt.figure()
plt.plot(y_test,color='black')
plt.plot(cc,color='red')
plt.title('LSTM test')


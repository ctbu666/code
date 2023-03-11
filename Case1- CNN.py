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

# Define structure of 1D CNN
x_input=Input(shape=(window,1))
kernels_length=[4,10,15] # The kernel lengths used for each convolution layer
feature_dim=96 # dimension of extracted features
pool_output=[] # The container used to receive outputs from each convolution layer
for klength in kernels_length: # convolution layer in parallel
    # 1D convolution layer, 32 filters, kernel width is the dimension of word vector and kernel length is kernel_text,
    # strides indicate how many steps the filter move for once, activation function is "relu"
    c=Conv1D(filters=32, kernel_size=klength, strides=1,activation="relu")(x_input)
    p=MaxPooling1D(pool_size=int(c.shape[1]))(c) # 1D max pooling (pool size is the length of the feature map, it is used as a global max pooling)
    pool_output.append(p) # Save the output in pool_output
pool_output=concatenate([p for p in pool_output]) # Concatenate all the features in pool_output (parallel to sequence)
x_feature=Flatten()(pool_output) # sequence to vevtor
output=Dense(1,activation='sigmoid')(x_feature)
mbCNN=Model(x_input,output)
mbCNN.compile(optimizer='adam',loss='mse')
x_train_3d = np.expand_dims(x_train, axis=2)
mbCNN.fit(x_train_3d,y_train,epochs=100,batch_size=60,shuffle=False,validation_split=0.05)

y_train_pre=mbCNN.predict(x_train_3d)
plt.figure()
plt.plot(y_train,color='black')
plt.plot(y_train_pre,color='red')
plt.title('Train')

x_test_3d=np.expand_dims(x_test, axis=2)
y_test_pre=mbCNN.predict(x_test_3d)

mse_cnn=((y_test_pre - y_test) ** 2).mean()
rmse_cnn=np.sqrt(((y_test_pre - y_test) ** 2).mean())
mae_cnn=(np.abs(y_test_pre - y_test)).mean()

y_test=y_test*100
y_test_pre=y_test_pre*100

plt.figure()
plt.plot(y_test,color='black')
plt.plot(y_test_pre,color='red')
plt.title('Test')
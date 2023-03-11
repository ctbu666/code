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

x_train=x_train/1.0104
y_train=y_train/2215

x_test=x_test/1.0104
y_test=y_test/2215

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
#Train 1D CNN
mbCNN.fit(x_train_3d,y_train,epochs=1,batch_size=60,shuffle=False,validation_split=0.05)

y_train_pre=mbCNN.predict(x_train_3d)
plt.figure()
plt.plot(y_train,color='black')
plt.plot(y_train_pre,color='red')
plt.title('Train')


x_test_3d=np.expand_dims(x_test, axis=2)
y_test_pre=mbCNN.predict(x_test_3d)
plt.figure()
plt.plot(y_test,color='black')
plt.plot(y_test_pre,color='red')
plt.title('Test')


CNNfeature=Model(x_input,x_feature)

feature_ESNinput=list()
for tra in Trajectory_train:  
    feature=CNNfeature.predict(np.expand_dims(tra/1.0104, axis=2))
    feature=feature.T
    feature_ESNinput.append(feature)

for tra in Trajectory_test:  
    feature=CNNfeature.predict(np.expand_dims(tra/1.0104, axis=2))
    feature=feature.T
    feature_ESNinput.append(feature)


RUL_ESNoutput=list()
for rul in RUL_train:
    rulnew=rul.reshape((1,rul.size))/2215
    RUL_ESNoutput.append(rulnew)
for rul in RUL_test:
    rulnew=rul.reshape((1,rul.size))/2215
    RUL_ESNoutput.append(rulnew)

# Deep ESN
import sys
import os
#sys.path
sys.path.append(os.path.join(os.getcwd(),'DeepESN'))
sys.path.append(os.path.join(os.getcwd(),'utils'))
from DeepESN import DeepESN
from task import select_indexes
np.random.seed(7)
# Be careful with memory usage
Nu=feature_dim
Nr = 20 # number of recurrent units #初始是50
Nl = 10 # number of recurrent layers #初始是20
reg = 10.0**-2;
transient = 5

class Struct(object): pass

def config_battery(IP_indexes):

    configs = Struct()
    
    configs.rhos = 0.2 # set spectral radius 0.1 for all recurrent layers #初始0.8
    configs.lis = 0.7 # set li 0.7 for all recurrent layers #初始0.7
    configs.iss = 0.1 # set insput scale 0.1 for all recurrent layers
    
    configs.IPconf = Struct()
    configs.IPconf.DeepIP = 1 # activate pre-train
    configs.IPconf.threshold = 0.1 # threshold for gradient descent in pre-train algorithm
    configs.IPconf.eta = 10**-5 # learning rate for IP rule
    configs.IPconf.mu = 0 # mean of target gaussian function
    configs.IPconf.sigma = 0.1 # std of target gaussian function
    configs.IPconf.Nepochs = 10 # maximum number of epochs
    configs.IPconf.indexes = IP_indexes # perform the pre-train on these indexes

    configs.reservoirConf = Struct()
    configs.reservoirConf.connectivity = 1 # connectivity of recurrent matrix
    
    configs.readout = Struct()
    configs.readout.trainMethod = 'NormalEquations' # train with normal equations (faster)
    configs.readout.regularizations = 10.0**np.array(range(-4,-1,1))
    
    return configs

Indexes_train_DESN=list(range(0,100)) #前15个trajectory用于训练DeepESN
Indexes_test_DESN=list(range(100,124)) #后5个trajectory用于测试DeepESN
configs=config_battery(Indexes_train_DESN)

deepESN = DeepESN(Nu, Nr, Nl, configs)
states = deepESN.computeState(feature_ESNinput, deepESN.IPconf.DeepIP)

train_states = select_indexes(states, Indexes_train_DESN, transient)
train_targets = select_indexes(RUL_ESNoutput, Indexes_train_DESN, transient)
test_states = select_indexes(states, Indexes_test_DESN)
test_targets = select_indexes(RUL_ESNoutput, Indexes_test_DESN)

deepESN.trainReadout(train_states, train_targets, reg)

train_states_all = select_indexes(states, Indexes_train_DESN)
train_outputs = deepESN.computeOutput(train_states_all)

test_outputs = deepESN.computeOutput(test_states)

plt.figure()
plt.plot(y_train,color='black')
plt.plot(train_outputs.T,color='red')
plt.title('Train')

plt.figure()
plt.plot(y_test,color='black')
plt.plot(test_outputs.T,color='red')
plt.title('Test')




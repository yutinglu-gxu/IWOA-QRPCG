import tensorflow as tf
print("TensorFlow version is:",tf.__version__)

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import random
import math
from tensorflow.keras.models import Model
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Input,concatenate,add,Activation,Bidirectional,CuDNNGRU,ConvLSTM2D
from keras.layers.convolutional import Conv1D,MaxPooling1D
import time
import keras.backend as K
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

pre_size = 24
test_length = 336
valid_length = 5136 
feature = 1
epochs = 200
window_size = 168       
filters1 = 64         
filters2 = 128      
kernel_size = 5  
pool_size = 2         
strides = 2               
Dense_units1 = 64      
Dense_units2 = 32   
Dropout_rate = 0.3      
batch_size = 64        
Learning_rate = 0.0001  

n_seq = 7
n_steps = 24


pred_list=[]

def get_dataset(window_size):
    df = pd.read_csv('../ISO_2016_2017.csv')
    data=df['RT_Demand'].values.reshape(-1, 1)
    pre_true=data.reshape(-1)[-test_length:] 
    train_data=data.reshape(-1)[:-test_length]
    smax=train_data.max()
    smin=train_data.min()
    open_arr=((data-smin)/(smax-smin)).reshape(-1)
    X = np.zeros(shape=(len(open_arr) - window_size-pre_size+1, window_size))
    label = np.zeros(shape=(len(open_arr) - window_size-pre_size+1,pre_size))
    test_X = np.zeros(shape=(window_size,pre_size))
    for i in range(len(open_arr) - window_size-pre_size+1):
        X[i, :] = open_arr[i:i+window_size]
        label[i] = open_arr[i+window_size:i+window_size+pre_size]
    train_X = X[:(-test_length-valid_length), :]
    train_label = label[:(-test_length-valid_length)]
    
    valid_X = X[(-test_length-valid_length):-test_length, :]
    valid_label = label[(-test_length-valid_length):-test_length, :]
    valid_X = valid_X[pre_size-1:, :]
    valid_label = valid_label[pre_size-1:, :]
    
    test_X = X[-test_length:, :]
    test_label = label[-test_length:]
    test_X = test_X[pre_size-1:, :]
    test_label = test_label[pre_size-1:] 
    
    train_X=train_X.reshape(-1,window_size,1)
    test_X=test_X.reshape(-1,window_size,1)
    valid_X=valid_X.reshape(-1,window_size,1)
    train_label=train_label.reshape(-1,pre_size,1)
    valid_label=valid_label.reshape(-1,pre_size,1)
    test_label=test_label.reshape(-1,pre_size,1)
    
    return train_X, train_label, valid_X, valid_label, test_X, test_label,  smax, smin

def plot(pred, true):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(len(pred)), pred)
    ax.plot(range(len(true)), true)
    plt.show()

# 0.05、0.95、0.025、0.975
def quantile_loss(q, y_true, y_pred):
    e = (y_true-y_pred)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)

def build_model():      
    train_X, train_label, valid_X, valid_label, test_X, test_label,  smax, smin = get_dataset(window_size)

    #打散
    train_Xm = np.zeros(shape=(len(train_X), window_size,feature))
    train_labelm = np.zeros(shape=(len(train_label), pre_size,1))
    train_Xm=train_X 
    train_labelm=train_label
    indices = np.random.permutation(train_Xm.shape[0])
    train_Xm = train_Xm[indices]
    train_labelm = train_labelm[indices]   
    
    train_Xm = train_Xm.reshape((train_Xm.shape[0], n_seq, 1, n_steps, feature))
    valid_X = valid_X.reshape((valid_X.shape[0], n_seq, 1, n_steps, feature))
    test_X = test_X.reshape((test_X.shape[0], n_seq, 1, n_steps, feature))
    

    model = Sequential() #  (samples,time, rows, cols, channels)
    model.add(ConvLSTM2D(filters=filters1, kernel_size=(1,kernel_size), input_shape=(n_seq, 1, n_steps, feature),
                         padding='same',return_sequences=True))
    model.add(ConvLSTM2D(filters=filters2, kernel_size=(1,kernel_size), padding='same',return_sequences=True))
    model.add(Dropout(Dropout_rate))
    model.add(Flatten())
    model.add(Dense(Dense_units1,activation='relu'))
    model.add(Dense(Dense_units2,activation='relu'))
    model.add(Dense(24))
    model.summary()

    quantile=0.025  # 0.05、0.95、0.025、0.975
    model.compile(loss=lambda y_t, y_p: quantile_loss(quantile, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
    
    earlyStop =[
         tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30,restore_best_weights=True)
        ] 
    history=model.fit([train_Xm],train_labelm,callbacks=[earlyStop],epochs=epochs,batch_size=batch_size,validation_data=([valid_X], valid_label),verbose=1,shuffle=False)
    val_loss=np.array(history.history['val_loss'])
    plot(history.history['loss'], history.history['val_loss'])
    
    prediction = model.predict([test_X])
    scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
    
    result=model.evaluate([valid_X], valid_label)
    
    return scaled_prediction,model,result

if __name__ == '__main__':
    pred_list,model,result=build_model()
    model_filename='./ConvLSTM0025.h5'
    print(result)
    model.save_weights(model_filename) 
    pred_value=pd.DataFrame(data=(np.array(pred_list).T))
    pred_value.to_csv("./ConvLSTM0025.csv",encoding='utf-8')
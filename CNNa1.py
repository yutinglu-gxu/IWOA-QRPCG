import tensorflow as tf
print("TensorFlow version is:",tf.__version__)
print(tf.test.is_built_with_cuda())
print(tf.config.list_physical_devices('GPU'))
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import random
import math
from tensorflow.keras.models import Model
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Input,Activation
from keras.layers.convolutional import Conv1D,MaxPooling1D
from tensorflow.compat.v1.keras.layers import CuDNNGRU,CuDNNLSTM
import time
import keras.backend as K
from keras.callbacks import EarlyStopping
from math import sqrt
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error,mean_squared_error,r2_score

print("---------------------CNNa1_10-------------------------")
pre_size = 24
test_length = 168
feature = 5
window_size = 24

pool_size=2
epochs= 150
Learning_rate= 0.0005
batch_size= 32
units1= 16
units2= 32
Dropout_rate= 0.2
kernel_size= 3

def build_model(units1,units2,Dropout_rate,kernel_size):
    inp = Input(shape=(window_size, feature))
    conv = Conv1D(units1, kernel_size,strides=1,padding='same',activation='relu')(inp)
    conv = MaxPooling1D(pool_size=pool_size,strides=1, padding='same')(conv)
    conv = Dropout(Dropout_rate)(conv)
    conv = Conv1D(units2, kernel_size,strides=1,padding='same',activation='relu')(conv)
    conv = MaxPooling1D(pool_size=pool_size,strides=1, padding='same')(conv)
    conv = Dropout(Dropout_rate)(conv)
    conv = Flatten()(conv)
    outp = Dense(24)(conv)
    model = Model(inputs=inp, outputs=outp)
    return model

def huber_loss(threshold,y_true, y_pred):          
    error = y_true - y_pred
    is_small_error = tf.abs(error) <= threshold
    small_error_loss = tf.square(error)/2    
    big_error_loss = threshold * (tf.abs(error) - 0.5 * threshold)
    return K.mean(tf.where(is_small_error,small_error_loss,big_error_loss))

print("---------------------k=0.1,p=1/32--------------------------")
def get_dataset():
    df = pd.read_csv('./data1.csv',usecols=["Z21","T","month","day","hour"])  
    data = np.array(df)
#   数据集总长度
    wi = np.ones(shape=(data.shape[0],feature))
    # 训练总长度
    ab_len=data[:-test_length-window_size,:].shape[0]
    indices = pd.read_csv('./indices11.csv',usecols=["9"])  
    indices =np.array(indices).reshape(-1).astype(int)
#   攻击 k p
    # k随机选10% 20% 30%
    k = 0.1
    p = 1/32
    # 测试集用的前168个数据不能受到攻击
    for i in range(int(data[:(-test_length-window_size),:].shape[0]*k)):
        wi[indices[i],0]= 1+p
    data=data*wi   # 受到攻击        
        
    train_data=data[:-test_length,:]
    smax=train_data.max(axis=0)
    smin=train_data.min(axis=0)
    open_arr=((data-smin)/(smax-smin))
    X = np.zeros(shape=(len(open_arr) - window_size-pre_size+1, window_size,feature))
    label = np.zeros(shape=(len(open_arr) - window_size-pre_size+1,pre_size))
    for i in range(len(open_arr) - window_size-pre_size+1):
        X[i, :] = open_arr[i:i+window_size,:]
        label[i] = open_arr[i+window_size:i+window_size+pre_size,0]
    train_X = X[:(-test_length), :]
    train_label = label[:(-test_length)]

    test_X = X[-test_length:, :]
    test_label = label[-test_length:]
    test_X = test_X[pre_size-1:, :]
    test_label = test_label[pre_size-1:] 
   
    valid_length=int(train_X.shape[0]*0.2)
    valid_X = train_X[-valid_length:, :]
    valid_label = train_label[-valid_length:]
    valid_X = valid_X[pre_size-1:, :]
    valid_label = valid_label[pre_size-1:]
    
    train_X = train_X[:-valid_length, :]
    train_label = train_label[:-valid_length]   
    
    return train_X, train_label, valid_X, valid_label, test_X, test_label,  smax[0], smin[0]

train_X, train_label, valid_X, valid_label, test_X, test_label,  smax, smin = get_dataset() 

#打散
train_Xm = np.zeros(shape=(len(train_X), window_size,feature))
train_labelm = np.zeros(shape=(len(train_label), pre_size,1))
indices = np.random.permutation(train_Xm.shape[0])
train_Xm = train_X[indices]
train_labelm = train_label[indices] 

wr = np.zeros(shape=(len(test_label.reshape(-1)),12))

print("--------------------------0.0005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.0005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,0] = pred_test

print("--------------------------0.001-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.001, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,1] = pred_test

print("--------------------------0.005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,2] = pred_test

print("---------------------------0.01----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.01, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,3] = pred_test

print("-----------------------------0.015----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.015, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)   
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,4] = pred_test

print("---------------------------------0.02-------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.02, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,5] = pred_test

print("------------------------------------0.025---------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.025, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)    
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,6] = pred_test

print("--------------------------------------0.03--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.03, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,7] = pred_test

print("--------------------------------------0.035--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.035, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,8] = pred_test

print("--------------------------------------0.04--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.04, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,9] = pred_test

def mae_loss(y_true, y_pred):       
    return K.mean(tf.abs(y_true - y_pred))
print("--------------------------mae-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mae_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,10] = pred_test

def mse_loss(y_true, y_pred):       
    return K.mean(tf.square(y_true - y_pred))
print("--------------------------mse-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mse_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,11] = pred_test

pred_value=pd.DataFrame(data=(np.array(wr)))
pred_value.to_csv("./ex/cnna1__11_10.csv",encoding='utf-8') 



print("---------------------k=0.1,p=1/16--------------------------")
def get_dataset():
    df = pd.read_csv('./data1.csv',usecols=["Z21","T","month","day","hour"])  
    data = np.array(df)
#   数据集总长度
    wi = np.ones(shape=(data.shape[0],feature))
    # 训练总长度
    ab_len=data[:-test_length-window_size,:].shape[0]
    indices = pd.read_csv('./indices12.csv',usecols=["9"])  
    indices =np.array(indices).reshape(-1).astype(int)
#   攻击 k p
    # k随机选10% 20% 30%
    k = 0.1
    p=1/16
    # 测试集用的前168个数据不能受到攻击
    for i in range(int(data[:(-test_length-window_size),:].shape[0]*k)):
        wi[indices[i],0]= 1+p
    data=data*wi   # 受到攻击        
        
    train_data=data[:-test_length,:]
    smax=train_data.max(axis=0)
    smin=train_data.min(axis=0)
    open_arr=((data-smin)/(smax-smin))
    X = np.zeros(shape=(len(open_arr) - window_size-pre_size+1, window_size,feature))
    label = np.zeros(shape=(len(open_arr) - window_size-pre_size+1,pre_size))
    for i in range(len(open_arr) - window_size-pre_size+1):
        X[i, :] = open_arr[i:i+window_size,:]
        label[i] = open_arr[i+window_size:i+window_size+pre_size,0]
    train_X = X[:(-test_length), :]
    train_label = label[:(-test_length)]

    test_X = X[-test_length:, :]
    test_label = label[-test_length:]
    test_X = test_X[pre_size-1:, :]
    test_label = test_label[pre_size-1:] 
   
    valid_length=int(train_X.shape[0]*0.2)
    valid_X = train_X[-valid_length:, :]
    valid_label = train_label[-valid_length:]
    valid_X = valid_X[pre_size-1:, :]
    valid_label = valid_label[pre_size-1:]
    
    train_X = train_X[:-valid_length, :]
    train_label = train_label[:-valid_length]   
    
    return train_X, train_label, valid_X, valid_label, test_X, test_label,  smax[0], smin[0]

train_X, train_label, valid_X, valid_label, test_X, test_label,  smax, smin = get_dataset() 

#打散
train_Xm = np.zeros(shape=(len(train_X), window_size,feature))
train_labelm = np.zeros(shape=(len(train_label), pre_size,1))
indices = np.random.permutation(train_Xm.shape[0])
train_Xm = train_X[indices]
train_labelm = train_label[indices] 

wr = np.zeros(shape=(len(test_label.reshape(-1)),12))

print("--------------------------0.0005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.0005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,0] = pred_test

print("--------------------------0.001-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.001, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,1] = pred_test

print("--------------------------0.005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,2] = pred_test

print("---------------------------0.01----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.01, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,3] = pred_test

print("-----------------------------0.015----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.015, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)   
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,4] = pred_test

print("---------------------------------0.02-------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.02, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,5] = pred_test

print("------------------------------------0.025---------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.025, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)    
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,6] = pred_test

print("--------------------------------------0.03--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.03, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,7] = pred_test

print("--------------------------------------0.035--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.035, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,8] = pred_test

print("--------------------------------------0.04--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.04, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,9] = pred_test

def mae_loss(y_true, y_pred):       
    return K.mean(tf.abs(y_true - y_pred))
print("--------------------------mae-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mae_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,10] = pred_test

def mse_loss(y_true, y_pred):       
    return K.mean(tf.square(y_true - y_pred))
print("--------------------------mse-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mse_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,11] = pred_test

pred_value=pd.DataFrame(data=(np.array(wr)))
pred_value.to_csv("./ex/cnna1__12_10.csv",encoding='utf-8') 



print("---------------------k=0.1,p=1/8--------------------------")
def get_dataset():
    df = pd.read_csv('./data1.csv',usecols=["Z21","T","month","day","hour"])  
    data = np.array(df)
#   数据集总长度
    wi = np.ones(shape=(data.shape[0],feature))
    # 训练总长度
    ab_len=data[:-test_length-window_size,:].shape[0]
    indices = pd.read_csv('./indices13.csv',usecols=["9"])  
    indices =np.array(indices).reshape(-1).astype(int)
#   攻击 k p
    # k随机选10% 20% 30%
    k = 0.1
    p=1/8
    # 测试集用的前168个数据不能受到攻击
    for i in range(int(data[:(-test_length-window_size),:].shape[0]*k)):
        wi[indices[i],0]= 1+p
    data=data*wi   # 受到攻击        
        
    train_data=data[:-test_length,:]
    smax=train_data.max(axis=0)
    smin=train_data.min(axis=0)
    open_arr=((data-smin)/(smax-smin))
    X = np.zeros(shape=(len(open_arr) - window_size-pre_size+1, window_size,feature))
    label = np.zeros(shape=(len(open_arr) - window_size-pre_size+1,pre_size))
    for i in range(len(open_arr) - window_size-pre_size+1):
        X[i, :] = open_arr[i:i+window_size,:]
        label[i] = open_arr[i+window_size:i+window_size+pre_size,0]
    train_X = X[:(-test_length), :]
    train_label = label[:(-test_length)]

    test_X = X[-test_length:, :]
    test_label = label[-test_length:]
    test_X = test_X[pre_size-1:, :]
    test_label = test_label[pre_size-1:] 
   
    valid_length=int(train_X.shape[0]*0.2)
    valid_X = train_X[-valid_length:, :]
    valid_label = train_label[-valid_length:]
    valid_X = valid_X[pre_size-1:, :]
    valid_label = valid_label[pre_size-1:]
    
    train_X = train_X[:-valid_length, :]
    train_label = train_label[:-valid_length]   
    
    return train_X, train_label, valid_X, valid_label, test_X, test_label,  smax[0], smin[0]

train_X, train_label, valid_X, valid_label, test_X, test_label,  smax, smin = get_dataset() 

#打散
train_Xm = np.zeros(shape=(len(train_X), window_size,feature))
train_labelm = np.zeros(shape=(len(train_label), pre_size,1))
indices = np.random.permutation(train_Xm.shape[0])
train_Xm = train_X[indices]
train_labelm = train_label[indices] 

wr = np.zeros(shape=(len(test_label.reshape(-1)),12))

print("--------------------------0.0005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.0005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,0] = pred_test

print("--------------------------0.001-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.001, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,1] = pred_test

print("--------------------------0.005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,2] = pred_test

print("---------------------------0.01----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.01, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,3] = pred_test

print("-----------------------------0.015----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.015, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)   
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,4] = pred_test

print("---------------------------------0.02-------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.02, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,5] = pred_test

print("------------------------------------0.025---------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.025, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)    
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,6] = pred_test

print("--------------------------------------0.03--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.03, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,7] = pred_test

print("--------------------------------------0.035--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.035, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,8] = pred_test

print("--------------------------------------0.04--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.04, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,9] = pred_test

def mae_loss(y_true, y_pred):       
    return K.mean(tf.abs(y_true - y_pred))
print("--------------------------mae-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mae_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,10] = pred_test

def mse_loss(y_true, y_pred):       
    return K.mean(tf.square(y_true - y_pred))
print("--------------------------mse-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mse_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,11] = pred_test

pred_value=pd.DataFrame(data=(np.array(wr)))
pred_value.to_csv("./ex/cnna1__13_10.csv",encoding='utf-8') 



print("---------------------k=0.1,p=1/4--------------------------")
def get_dataset():
    df = pd.read_csv('./data1.csv',usecols=["Z21","T","month","day","hour"])  
    data = np.array(df)
#   数据集总长度
    wi = np.ones(shape=(data.shape[0],feature))
    # 训练总长度
    ab_len=data[:-test_length-window_size,:].shape[0]
    indices = pd.read_csv('./indices14.csv',usecols=["9"])  
    indices =np.array(indices).reshape(-1).astype(int)
#   攻击 k p
    # k随机选10% 20% 30%
    k = 0.1
    p=1/4
    # 测试集用的前168个数据不能受到攻击
    for i in range(int(data[:(-test_length-window_size),:].shape[0]*k)):
        wi[indices[i],0]= 1+p
    data=data*wi   # 受到攻击        
        
    train_data=data[:-test_length,:]
    smax=train_data.max(axis=0)
    smin=train_data.min(axis=0)
    open_arr=((data-smin)/(smax-smin))
    X = np.zeros(shape=(len(open_arr) - window_size-pre_size+1, window_size,feature))
    label = np.zeros(shape=(len(open_arr) - window_size-pre_size+1,pre_size))
    for i in range(len(open_arr) - window_size-pre_size+1):
        X[i, :] = open_arr[i:i+window_size,:]
        label[i] = open_arr[i+window_size:i+window_size+pre_size,0]
    train_X = X[:(-test_length), :]
    train_label = label[:(-test_length)]

    test_X = X[-test_length:, :]
    test_label = label[-test_length:]
    test_X = test_X[pre_size-1:, :]
    test_label = test_label[pre_size-1:] 
   
    valid_length=int(train_X.shape[0]*0.2)
    valid_X = train_X[-valid_length:, :]
    valid_label = train_label[-valid_length:]
    valid_X = valid_X[pre_size-1:, :]
    valid_label = valid_label[pre_size-1:]
    
    train_X = train_X[:-valid_length, :]
    train_label = train_label[:-valid_length]   
    
    return train_X, train_label, valid_X, valid_label, test_X, test_label,  smax[0], smin[0]

train_X, train_label, valid_X, valid_label, test_X, test_label,  smax, smin = get_dataset() 

#打散
train_Xm = np.zeros(shape=(len(train_X), window_size,feature))
train_labelm = np.zeros(shape=(len(train_label), pre_size,1))
indices = np.random.permutation(train_Xm.shape[0])
train_Xm = train_X[indices]
train_labelm = train_label[indices] 

wr = np.zeros(shape=(len(test_label.reshape(-1)),12))

print("--------------------------0.0005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.0005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,0] = pred_test

print("--------------------------0.001-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.001, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,1] = pred_test

print("--------------------------0.005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,2] = pred_test

print("---------------------------0.01----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.01, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,3] = pred_test

print("-----------------------------0.015----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.015, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)   
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,4] = pred_test

print("---------------------------------0.02-------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.02, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,5] = pred_test

print("------------------------------------0.025---------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.025, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)    
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,6] = pred_test

print("--------------------------------------0.03--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.03, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,7] = pred_test

print("--------------------------------------0.035--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.035, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,8] = pred_test

print("--------------------------------------0.04--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.04, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,9] = pred_test

def mae_loss(y_true, y_pred):       
    return K.mean(tf.abs(y_true - y_pred))
print("--------------------------mae-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mae_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,10] = pred_test

def mse_loss(y_true, y_pred):       
    return K.mean(tf.square(y_true - y_pred))
print("--------------------------mse-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mse_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,11] = pred_test

pred_value=pd.DataFrame(data=(np.array(wr)))
pred_value.to_csv("./ex/cnna1__14_10.csv",encoding='utf-8') 


print("---------------------k=0.1,p=1/2--------------------------")
def get_dataset():
    df = pd.read_csv('./data1.csv',usecols=["Z21","T","month","day","hour"])  
    data = np.array(df)
#   数据集总长度
    wi = np.ones(shape=(data.shape[0],feature))
    # 训练总长度
    ab_len=data[:-test_length-window_size,:].shape[0]
    indices = pd.read_csv('./indices15.csv',usecols=["9"])  
    indices =np.array(indices).reshape(-1).astype(int)
#   攻击 k p
    # k随机选10% 20% 30%
    k = 0.1
    p=1/2
    # 测试集用的前168个数据不能受到攻击
    for i in range(int(data[:(-test_length-window_size),:].shape[0]*k)):
        wi[indices[i],0]= 1+p
    data=data*wi   # 受到攻击        
        
    train_data=data[:-test_length,:]
    smax=train_data.max(axis=0)
    smin=train_data.min(axis=0)
    open_arr=((data-smin)/(smax-smin))
    X = np.zeros(shape=(len(open_arr) - window_size-pre_size+1, window_size,feature))
    label = np.zeros(shape=(len(open_arr) - window_size-pre_size+1,pre_size))
    for i in range(len(open_arr) - window_size-pre_size+1):
        X[i, :] = open_arr[i:i+window_size,:]
        label[i] = open_arr[i+window_size:i+window_size+pre_size,0]
    train_X = X[:(-test_length), :]
    train_label = label[:(-test_length)]

    test_X = X[-test_length:, :]
    test_label = label[-test_length:]
    test_X = test_X[pre_size-1:, :]
    test_label = test_label[pre_size-1:] 
   
    valid_length=int(train_X.shape[0]*0.2)
    valid_X = train_X[-valid_length:, :]
    valid_label = train_label[-valid_length:]
    valid_X = valid_X[pre_size-1:, :]
    valid_label = valid_label[pre_size-1:]
    
    train_X = train_X[:-valid_length, :]
    train_label = train_label[:-valid_length]   
    
    return train_X, train_label, valid_X, valid_label, test_X, test_label,  smax[0], smin[0]

train_X, train_label, valid_X, valid_label, test_X, test_label,  smax, smin = get_dataset() 

#打散
train_Xm = np.zeros(shape=(len(train_X), window_size,feature))
train_labelm = np.zeros(shape=(len(train_label), pre_size,1))
indices = np.random.permutation(train_Xm.shape[0])
train_Xm = train_X[indices]
train_labelm = train_label[indices] 

wr = np.zeros(shape=(len(test_label.reshape(-1)),12))

print("--------------------------0.0005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.0005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,0] = pred_test

print("--------------------------0.001-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.001, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,1] = pred_test

print("--------------------------0.005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,2] = pred_test

print("---------------------------0.01----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.01, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,3] = pred_test

print("-----------------------------0.015----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.015, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)   
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,4] = pred_test

print("---------------------------------0.02-------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.02, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,5] = pred_test

print("------------------------------------0.025---------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.025, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)    
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,6] = pred_test

print("--------------------------------------0.03--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.03, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,7] = pred_test

print("--------------------------------------0.035--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.035, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,8] = pred_test

print("--------------------------------------0.04--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.04, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,9] = pred_test

def mae_loss(y_true, y_pred):       
    return K.mean(tf.abs(y_true - y_pred))
print("--------------------------mae-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mae_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,10] = pred_test

def mse_loss(y_true, y_pred):       
    return K.mean(tf.square(y_true - y_pred))
print("--------------------------mse-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mse_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,11] = pred_test

pred_value=pd.DataFrame(data=(np.array(wr)))
pred_value.to_csv("./ex/cnna1__15_10.csv",encoding='utf-8') 


print("---------------------k=0.1,p=1--------------------------")
def get_dataset():
    df = pd.read_csv('./data1.csv',usecols=["Z21","T","month","day","hour"])  
    data = np.array(df)
#   数据集总长度
    wi = np.ones(shape=(data.shape[0],feature))
    # 训练总长度
    ab_len=data[:-test_length-window_size,:].shape[0]
    indices = pd.read_csv('./indices16.csv',usecols=["9"])  
    indices =np.array(indices).reshape(-1).astype(int)
#   攻击 k p
    # k随机选10% 20% 30%
    k = 0.1
    p=1
    # 测试集用的前168个数据不能受到攻击
    for i in range(int(data[:(-test_length-window_size),:].shape[0]*k)):
        wi[indices[i],0]= 1+p
    data=data*wi   # 受到攻击        
        
    train_data=data[:-test_length,:]
    smax=train_data.max(axis=0)
    smin=train_data.min(axis=0)
    open_arr=((data-smin)/(smax-smin))
    X = np.zeros(shape=(len(open_arr) - window_size-pre_size+1, window_size,feature))
    label = np.zeros(shape=(len(open_arr) - window_size-pre_size+1,pre_size))
    for i in range(len(open_arr) - window_size-pre_size+1):
        X[i, :] = open_arr[i:i+window_size,:]
        label[i] = open_arr[i+window_size:i+window_size+pre_size,0]
    train_X = X[:(-test_length), :]
    train_label = label[:(-test_length)]

    test_X = X[-test_length:, :]
    test_label = label[-test_length:]
    test_X = test_X[pre_size-1:, :]
    test_label = test_label[pre_size-1:] 
   
    valid_length=int(train_X.shape[0]*0.2)
    valid_X = train_X[-valid_length:, :]
    valid_label = train_label[-valid_length:]
    valid_X = valid_X[pre_size-1:, :]
    valid_label = valid_label[pre_size-1:]
    
    train_X = train_X[:-valid_length, :]
    train_label = train_label[:-valid_length]   
    
    return train_X, train_label, valid_X, valid_label, test_X, test_label,  smax[0], smin[0]

train_X, train_label, valid_X, valid_label, test_X, test_label,  smax, smin = get_dataset() 

#打散
train_Xm = np.zeros(shape=(len(train_X), window_size,feature))
train_labelm = np.zeros(shape=(len(train_label), pre_size,1))
indices = np.random.permutation(train_Xm.shape[0])
train_Xm = train_X[indices]
train_labelm = train_label[indices] 

wr = np.zeros(shape=(len(test_label.reshape(-1)),12))

print("--------------------------0.0005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.0005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,0] = pred_test

print("--------------------------0.001-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.001, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,1] = pred_test

print("--------------------------0.005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,2] = pred_test

print("---------------------------0.01----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.01, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,3] = pred_test

print("-----------------------------0.015----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.015, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)   
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,4] = pred_test

print("---------------------------------0.02-------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.02, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,5] = pred_test

print("------------------------------------0.025---------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.025, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)    
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,6] = pred_test

print("--------------------------------------0.03--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.03, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,7] = pred_test

print("--------------------------------------0.035--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.035, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,8] = pred_test

print("--------------------------------------0.04--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.04, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,9] = pred_test

def mae_loss(y_true, y_pred):       
    return K.mean(tf.abs(y_true - y_pred))
print("--------------------------mae-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mae_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,10] = pred_test

def mse_loss(y_true, y_pred):       
    return K.mean(tf.square(y_true - y_pred))
print("--------------------------mse-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mse_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,11] = pred_test

pred_value=pd.DataFrame(data=(np.array(wr)))
pred_value.to_csv("./ex/cnna1__16_10.csv",encoding='utf-8') 



print("---------------------k=0.2,p=1/32--------------------------")
def get_dataset():
    df = pd.read_csv('./data1.csv',usecols=["Z21","T","month","day","hour"])  
    data = np.array(df)
#   数据集总长度
    wi = np.ones(shape=(data.shape[0],feature))
    # 训练总长度
    ab_len=data[:-test_length-window_size,:].shape[0]
    indices = pd.read_csv('./indices21.csv',usecols=["9"])  
    indices =np.array(indices).reshape(-1).astype(int)
#   攻击 k p
    # k随机选10% 20% 30%
    k = 0.2
    p=1/32
    # 测试集用的前168个数据不能受到攻击
    for i in range(int(data[:(-test_length-window_size),:].shape[0]*k)):
        wi[indices[i],0]= 1+p
    data=data*wi   # 受到攻击        
        
    train_data=data[:-test_length,:]
    smax=train_data.max(axis=0)
    smin=train_data.min(axis=0)
    open_arr=((data-smin)/(smax-smin))
    X = np.zeros(shape=(len(open_arr) - window_size-pre_size+1, window_size,feature))
    label = np.zeros(shape=(len(open_arr) - window_size-pre_size+1,pre_size))
    for i in range(len(open_arr) - window_size-pre_size+1):
        X[i, :] = open_arr[i:i+window_size,:]
        label[i] = open_arr[i+window_size:i+window_size+pre_size,0]
    train_X = X[:(-test_length), :]
    train_label = label[:(-test_length)]

    test_X = X[-test_length:, :]
    test_label = label[-test_length:]
    test_X = test_X[pre_size-1:, :]
    test_label = test_label[pre_size-1:] 
   
    valid_length=int(train_X.shape[0]*0.2)
    valid_X = train_X[-valid_length:, :]
    valid_label = train_label[-valid_length:]
    valid_X = valid_X[pre_size-1:, :]
    valid_label = valid_label[pre_size-1:]
    
    train_X = train_X[:-valid_length, :]
    train_label = train_label[:-valid_length]   
    
    return train_X, train_label, valid_X, valid_label, test_X, test_label,  smax[0], smin[0]

train_X, train_label, valid_X, valid_label, test_X, test_label,  smax, smin = get_dataset() 

#打散
train_Xm = np.zeros(shape=(len(train_X), window_size,feature))
train_labelm = np.zeros(shape=(len(train_label), pre_size,1))
indices = np.random.permutation(train_Xm.shape[0])
train_Xm = train_X[indices]
train_labelm = train_label[indices] 

wr = np.zeros(shape=(len(test_label.reshape(-1)),12))

print("--------------------------0.0005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.0005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,0] = pred_test

print("--------------------------0.001-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.001, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,1] = pred_test

print("--------------------------0.005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,2] = pred_test

print("---------------------------0.01----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.01, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,3] = pred_test

print("-----------------------------0.015----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.015, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)   
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,4] = pred_test

print("---------------------------------0.02-------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.02, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,5] = pred_test

print("------------------------------------0.025---------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.025, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)    
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,6] = pred_test

print("--------------------------------------0.03--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.03, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,7] = pred_test

print("--------------------------------------0.035--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.035, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,8] = pred_test

print("--------------------------------------0.04--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.04, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,9] = pred_test

def mae_loss(y_true, y_pred):       
    return K.mean(tf.abs(y_true - y_pred))
print("--------------------------mae-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mae_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,10] = pred_test

def mse_loss(y_true, y_pred):       
    return K.mean(tf.square(y_true - y_pred))
print("--------------------------mse-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mse_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,11] = pred_test

pred_value=pd.DataFrame(data=(np.array(wr)))
pred_value.to_csv("./ex/cnna1__21_10.csv",encoding='utf-8') 


print("---------------------k=0.2,p=1/16--------------------------")
def get_dataset():
    df = pd.read_csv('./data1.csv',usecols=["Z21","T","month","day","hour"])  
    data = np.array(df)
#   数据集总长度
    wi = np.ones(shape=(data.shape[0],feature))
    # 训练总长度
    ab_len=data[:-test_length-window_size,:].shape[0]
    indices = pd.read_csv('./indices22.csv',usecols=["9"])  
    indices =np.array(indices).reshape(-1).astype(int)
#   攻击 k p
    # k随机选10% 20% 30%
    k = 0.2
    p=1/16
    # 测试集用的前168个数据不能受到攻击
    for i in range(int(data[:(-test_length-window_size),:].shape[0]*k)):
        wi[indices[i],0]= 1+p
    data=data*wi   # 受到攻击        
        
    train_data=data[:-test_length,:]
    smax=train_data.max(axis=0)
    smin=train_data.min(axis=0)
    open_arr=((data-smin)/(smax-smin))
    X = np.zeros(shape=(len(open_arr) - window_size-pre_size+1, window_size,feature))
    label = np.zeros(shape=(len(open_arr) - window_size-pre_size+1,pre_size))
    for i in range(len(open_arr) - window_size-pre_size+1):
        X[i, :] = open_arr[i:i+window_size,:]
        label[i] = open_arr[i+window_size:i+window_size+pre_size,0]
    train_X = X[:(-test_length), :]
    train_label = label[:(-test_length)]

    test_X = X[-test_length:, :]
    test_label = label[-test_length:]
    test_X = test_X[pre_size-1:, :]
    test_label = test_label[pre_size-1:] 
   
    valid_length=int(train_X.shape[0]*0.2)
    valid_X = train_X[-valid_length:, :]
    valid_label = train_label[-valid_length:]
    valid_X = valid_X[pre_size-1:, :]
    valid_label = valid_label[pre_size-1:]
    
    train_X = train_X[:-valid_length, :]
    train_label = train_label[:-valid_length]   
    
    return train_X, train_label, valid_X, valid_label, test_X, test_label,  smax[0], smin[0]

train_X, train_label, valid_X, valid_label, test_X, test_label,  smax, smin = get_dataset() 

#打散
train_Xm = np.zeros(shape=(len(train_X), window_size,feature))
train_labelm = np.zeros(shape=(len(train_label), pre_size,1))
indices = np.random.permutation(train_Xm.shape[0])
train_Xm = train_X[indices]
train_labelm = train_label[indices] 

wr = np.zeros(shape=(len(test_label.reshape(-1)),12))

print("--------------------------0.0005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.0005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,0] = pred_test

print("--------------------------0.001-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.001, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,1] = pred_test

print("--------------------------0.005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,2] = pred_test

print("---------------------------0.01----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.01, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,3] = pred_test

print("-----------------------------0.015----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.015, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)   
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,4] = pred_test

print("---------------------------------0.02-------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.02, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,5] = pred_test

print("------------------------------------0.025---------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.025, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)    
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,6] = pred_test

print("--------------------------------------0.03--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.03, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,7] = pred_test

print("--------------------------------------0.035--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.035, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,8] = pred_test

print("--------------------------------------0.04--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.04, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,9] = pred_test

def mae_loss(y_true, y_pred):       
    return K.mean(tf.abs(y_true - y_pred))
print("--------------------------mae-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mae_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,10] = pred_test

def mse_loss(y_true, y_pred):       
    return K.mean(tf.square(y_true - y_pred))
print("--------------------------mse-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mse_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,11] = pred_test

pred_value=pd.DataFrame(data=(np.array(wr)))
pred_value.to_csv("./ex/cnna1__22_10.csv",encoding='utf-8') 


print("---------------------k=0.2,p=1/8--------------------------")
def get_dataset():
    df = pd.read_csv('./data1.csv',usecols=["Z21","T","month","day","hour"])  
    data = np.array(df)
#   数据集总长度
    wi = np.ones(shape=(data.shape[0],feature))
    # 训练总长度
    ab_len=data[:-test_length-window_size,:].shape[0]
    indices = pd.read_csv('./indices23.csv',usecols=["9"])  
    indices =np.array(indices).reshape(-1).astype(int)
#   攻击 k p
    # k随机选10% 20% 30%
    k = 0.2
    p=1/8
    # 测试集用的前168个数据不能受到攻击
    for i in range(int(data[:(-test_length-window_size),:].shape[0]*k)):
        wi[indices[i],0]= 1+p
    data=data*wi   # 受到攻击        
        
    train_data=data[:-test_length,:]
    smax=train_data.max(axis=0)
    smin=train_data.min(axis=0)
    open_arr=((data-smin)/(smax-smin))
    X = np.zeros(shape=(len(open_arr) - window_size-pre_size+1, window_size,feature))
    label = np.zeros(shape=(len(open_arr) - window_size-pre_size+1,pre_size))
    for i in range(len(open_arr) - window_size-pre_size+1):
        X[i, :] = open_arr[i:i+window_size,:]
        label[i] = open_arr[i+window_size:i+window_size+pre_size,0]
    train_X = X[:(-test_length), :]
    train_label = label[:(-test_length)]

    test_X = X[-test_length:, :]
    test_label = label[-test_length:]
    test_X = test_X[pre_size-1:, :]
    test_label = test_label[pre_size-1:] 
   
    valid_length=int(train_X.shape[0]*0.2)
    valid_X = train_X[-valid_length:, :]
    valid_label = train_label[-valid_length:]
    valid_X = valid_X[pre_size-1:, :]
    valid_label = valid_label[pre_size-1:]
    
    train_X = train_X[:-valid_length, :]
    train_label = train_label[:-valid_length]   
    
    return train_X, train_label, valid_X, valid_label, test_X, test_label,  smax[0], smin[0]

train_X, train_label, valid_X, valid_label, test_X, test_label,  smax, smin = get_dataset() 

#打散
train_Xm = np.zeros(shape=(len(train_X), window_size,feature))
train_labelm = np.zeros(shape=(len(train_label), pre_size,1))
indices = np.random.permutation(train_Xm.shape[0])
train_Xm = train_X[indices]
train_labelm = train_label[indices] 

wr = np.zeros(shape=(len(test_label.reshape(-1)),12))

print("--------------------------0.0005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.0005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,0] = pred_test

print("--------------------------0.001-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.001, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,1] = pred_test

print("--------------------------0.005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,2] = pred_test

print("---------------------------0.01----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.01, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,3] = pred_test

print("-----------------------------0.015----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.015, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)   
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,4] = pred_test

print("---------------------------------0.02-------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.02, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,5] = pred_test

print("------------------------------------0.025---------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.025, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)    
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,6] = pred_test

print("--------------------------------------0.03--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.03, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,7] = pred_test

print("--------------------------------------0.035--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.035, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,8] = pred_test

print("--------------------------------------0.04--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.04, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,9] = pred_test

def mae_loss(y_true, y_pred):       
    return K.mean(tf.abs(y_true - y_pred))
print("--------------------------mae-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mae_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,10] = pred_test

def mse_loss(y_true, y_pred):       
    return K.mean(tf.square(y_true - y_pred))
print("--------------------------mse-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mse_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,11] = pred_test

pred_value=pd.DataFrame(data=(np.array(wr)))
pred_value.to_csv("./ex/cnna1__23_10.csv",encoding='utf-8') 



print("---------------------k=0.2,p=1/4--------------------------")
def get_dataset():
    df = pd.read_csv('./data1.csv',usecols=["Z21","T","month","day","hour"])  
    data = np.array(df)
#   数据集总长度
    wi = np.ones(shape=(data.shape[0],feature))
    # 训练总长度
    ab_len=data[:-test_length-window_size,:].shape[0]
    indices = pd.read_csv('./indices24.csv',usecols=["9"])  
    indices =np.array(indices).reshape(-1).astype(int)
#   攻击 k p
    # k随机选10% 20% 30%
    k = 0.2
    p=1/4
    # 测试集用的前168个数据不能受到攻击
    for i in range(int(data[:(-test_length-window_size),:].shape[0]*k)):
        wi[indices[i],0]= 1+p
    data=data*wi   # 受到攻击        
        
    train_data=data[:-test_length,:]
    smax=train_data.max(axis=0)
    smin=train_data.min(axis=0)
    open_arr=((data-smin)/(smax-smin))
    X = np.zeros(shape=(len(open_arr) - window_size-pre_size+1, window_size,feature))
    label = np.zeros(shape=(len(open_arr) - window_size-pre_size+1,pre_size))
    for i in range(len(open_arr) - window_size-pre_size+1):
        X[i, :] = open_arr[i:i+window_size,:]
        label[i] = open_arr[i+window_size:i+window_size+pre_size,0]
    train_X = X[:(-test_length), :]
    train_label = label[:(-test_length)]

    test_X = X[-test_length:, :]
    test_label = label[-test_length:]
    test_X = test_X[pre_size-1:, :]
    test_label = test_label[pre_size-1:] 
   
    valid_length=int(train_X.shape[0]*0.2)
    valid_X = train_X[-valid_length:, :]
    valid_label = train_label[-valid_length:]
    valid_X = valid_X[pre_size-1:, :]
    valid_label = valid_label[pre_size-1:]
    
    train_X = train_X[:-valid_length, :]
    train_label = train_label[:-valid_length]   
    
    return train_X, train_label, valid_X, valid_label, test_X, test_label,  smax[0], smin[0]

train_X, train_label, valid_X, valid_label, test_X, test_label,  smax, smin = get_dataset() 

#打散
train_Xm = np.zeros(shape=(len(train_X), window_size,feature))
train_labelm = np.zeros(shape=(len(train_label), pre_size,1))
indices = np.random.permutation(train_Xm.shape[0])
train_Xm = train_X[indices]
train_labelm = train_label[indices] 

wr = np.zeros(shape=(len(test_label.reshape(-1)),12))

print("--------------------------0.0005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.0005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,0] = pred_test

print("--------------------------0.001-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.001, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,1] = pred_test

print("--------------------------0.005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,2] = pred_test

print("---------------------------0.01----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.01, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,3] = pred_test

print("-----------------------------0.015----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.015, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)   
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,4] = pred_test

print("---------------------------------0.02-------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.02, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,5] = pred_test

print("------------------------------------0.025---------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.025, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)    
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,6] = pred_test

print("--------------------------------------0.03--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.03, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,7] = pred_test

print("--------------------------------------0.035--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.035, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,8] = pred_test

print("--------------------------------------0.04--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.04, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,9] = pred_test

def mae_loss(y_true, y_pred):       
    return K.mean(tf.abs(y_true - y_pred))
print("--------------------------mae-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mae_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,10] = pred_test

def mse_loss(y_true, y_pred):       
    return K.mean(tf.square(y_true - y_pred))
print("--------------------------mse-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mse_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,11] = pred_test

pred_value=pd.DataFrame(data=(np.array(wr)))
pred_value.to_csv("./ex/cnna1__24_10.csv",encoding='utf-8') 



print("---------------------k=0.2,p=1/2--------------------------")
def get_dataset():
    df = pd.read_csv('./data1.csv',usecols=["Z21","T","month","day","hour"])  
    data = np.array(df)
#   数据集总长度
    wi = np.ones(shape=(data.shape[0],feature))
    # 训练总长度
    ab_len=data[:-test_length-window_size,:].shape[0]
    indices = pd.read_csv('./indices25.csv',usecols=["9"])  
    indices =np.array(indices).reshape(-1).astype(int)
#   攻击 k p
    # k随机选10% 20% 30%
    k = 0.2
    p=1/2
    # 测试集用的前168个数据不能受到攻击
    for i in range(int(data[:(-test_length-window_size),:].shape[0]*k)):
        wi[indices[i],0]= 1+p
    data=data*wi   # 受到攻击        
        
    train_data=data[:-test_length,:]
    smax=train_data.max(axis=0)
    smin=train_data.min(axis=0)
    open_arr=((data-smin)/(smax-smin))
    X = np.zeros(shape=(len(open_arr) - window_size-pre_size+1, window_size,feature))
    label = np.zeros(shape=(len(open_arr) - window_size-pre_size+1,pre_size))
    for i in range(len(open_arr) - window_size-pre_size+1):
        X[i, :] = open_arr[i:i+window_size,:]
        label[i] = open_arr[i+window_size:i+window_size+pre_size,0]
    train_X = X[:(-test_length), :]
    train_label = label[:(-test_length)]

    test_X = X[-test_length:, :]
    test_label = label[-test_length:]
    test_X = test_X[pre_size-1:, :]
    test_label = test_label[pre_size-1:] 
   
    valid_length=int(train_X.shape[0]*0.2)
    valid_X = train_X[-valid_length:, :]
    valid_label = train_label[-valid_length:]
    valid_X = valid_X[pre_size-1:, :]
    valid_label = valid_label[pre_size-1:]
    
    train_X = train_X[:-valid_length, :]
    train_label = train_label[:-valid_length]   
    
    return train_X, train_label, valid_X, valid_label, test_X, test_label,  smax[0], smin[0]

train_X, train_label, valid_X, valid_label, test_X, test_label,  smax, smin = get_dataset() 

#打散
train_Xm = np.zeros(shape=(len(train_X), window_size,feature))
train_labelm = np.zeros(shape=(len(train_label), pre_size,1))
indices = np.random.permutation(train_Xm.shape[0])
train_Xm = train_X[indices]
train_labelm = train_label[indices] 

wr = np.zeros(shape=(len(test_label.reshape(-1)),12))

print("--------------------------0.0005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.0005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,0] = pred_test

print("--------------------------0.001-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.001, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,1] = pred_test

print("--------------------------0.005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,2] = pred_test

print("---------------------------0.01----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.01, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,3] = pred_test

print("-----------------------------0.015----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.015, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)   
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,4] = pred_test

print("---------------------------------0.02-------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.02, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,5] = pred_test

print("------------------------------------0.025---------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.025, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)    
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,6] = pred_test

print("--------------------------------------0.03--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.03, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,7] = pred_test

print("--------------------------------------0.035--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.035, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,8] = pred_test

print("--------------------------------------0.04--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.04, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,9] = pred_test

def mae_loss(y_true, y_pred):       
    return K.mean(tf.abs(y_true - y_pred))
print("--------------------------mae-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mae_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,10] = pred_test

def mse_loss(y_true, y_pred):       
    return K.mean(tf.square(y_true - y_pred))
print("--------------------------mse-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mse_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,11] = pred_test

pred_value=pd.DataFrame(data=(np.array(wr)))
pred_value.to_csv("./ex/cnna1__25_10.csv",encoding='utf-8') 


print("---------------------k=0.2,p=1--------------------------")
def get_dataset():
    df = pd.read_csv('./data1.csv',usecols=["Z21","T","month","day","hour"])  
    data = np.array(df)
#   数据集总长度
    wi = np.ones(shape=(data.shape[0],feature))
    # 训练总长度
    ab_len=data[:-test_length-window_size,:].shape[0]
    indices = pd.read_csv('./indices26.csv',usecols=["9"])  
    indices =np.array(indices).reshape(-1).astype(int)
#   攻击 k p
    # k随机选10% 20% 30%
    k = 0.2
    p=1
    # 测试集用的前168个数据不能受到攻击
    for i in range(int(data[:(-test_length-window_size),:].shape[0]*k)):
        wi[indices[i],0]= 1+p
    data=data*wi   # 受到攻击        
        
    train_data=data[:-test_length,:]
    smax=train_data.max(axis=0)
    smin=train_data.min(axis=0)
    open_arr=((data-smin)/(smax-smin))
    X = np.zeros(shape=(len(open_arr) - window_size-pre_size+1, window_size,feature))
    label = np.zeros(shape=(len(open_arr) - window_size-pre_size+1,pre_size))
    for i in range(len(open_arr) - window_size-pre_size+1):
        X[i, :] = open_arr[i:i+window_size,:]
        label[i] = open_arr[i+window_size:i+window_size+pre_size,0]
    train_X = X[:(-test_length), :]
    train_label = label[:(-test_length)]

    test_X = X[-test_length:, :]
    test_label = label[-test_length:]
    test_X = test_X[pre_size-1:, :]
    test_label = test_label[pre_size-1:] 
   
    valid_length=int(train_X.shape[0]*0.2)
    valid_X = train_X[-valid_length:, :]
    valid_label = train_label[-valid_length:]
    valid_X = valid_X[pre_size-1:, :]
    valid_label = valid_label[pre_size-1:]
    
    train_X = train_X[:-valid_length, :]
    train_label = train_label[:-valid_length]   
    
    return train_X, train_label, valid_X, valid_label, test_X, test_label,  smax[0], smin[0]

train_X, train_label, valid_X, valid_label, test_X, test_label,  smax, smin = get_dataset() 

#打散
train_Xm = np.zeros(shape=(len(train_X), window_size,feature))
train_labelm = np.zeros(shape=(len(train_label), pre_size,1))
indices = np.random.permutation(train_Xm.shape[0])
train_Xm = train_X[indices]
train_labelm = train_label[indices] 

wr = np.zeros(shape=(len(test_label.reshape(-1)),12))

print("--------------------------0.0005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.0005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,0] = pred_test

print("--------------------------0.001-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.001, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,1] = pred_test

print("--------------------------0.005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,2] = pred_test

print("---------------------------0.01----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.01, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,3] = pred_test

print("-----------------------------0.015----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.015, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)   
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,4] = pred_test

print("---------------------------------0.02-------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.02, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,5] = pred_test

print("------------------------------------0.025---------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.025, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)    
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,6] = pred_test

print("--------------------------------------0.03--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.03, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,7] = pred_test

print("--------------------------------------0.035--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.035, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,8] = pred_test

print("--------------------------------------0.04--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.04, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,9] = pred_test

def mae_loss(y_true, y_pred):       
    return K.mean(tf.abs(y_true - y_pred))
print("--------------------------mae-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mae_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,10] = pred_test

def mse_loss(y_true, y_pred):       
    return K.mean(tf.square(y_true - y_pred))
print("--------------------------mse-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mse_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,11] = pred_test

pred_value=pd.DataFrame(data=(np.array(wr)))
pred_value.to_csv("./ex/cnna1__26_10.csv",encoding='utf-8') 


print("---------------------k=0.3,p=1/32--------------------------")
def get_dataset():
    df = pd.read_csv('./data1.csv',usecols=["Z21","T","month","day","hour"])  
    data = np.array(df)
#   数据集总长度
    wi = np.ones(shape=(data.shape[0],feature))
    # 训练总长度
    ab_len=data[:-test_length-window_size,:].shape[0]
    indices = pd.read_csv('./indices31.csv',usecols=["9"])  
    indices =np.array(indices).reshape(-1).astype(int)
#   攻击 k p
    # k随机选10% 20% 30%
    k = 0.3
    p=1/32
    # 测试集用的前168个数据不能受到攻击
    for i in range(int(data[:(-test_length-window_size),:].shape[0]*k)):
        wi[indices[i],0]= 1+p
    data=data*wi   # 受到攻击        
        
    train_data=data[:-test_length,:]
    smax=train_data.max(axis=0)
    smin=train_data.min(axis=0)
    open_arr=((data-smin)/(smax-smin))
    X = np.zeros(shape=(len(open_arr) - window_size-pre_size+1, window_size,feature))
    label = np.zeros(shape=(len(open_arr) - window_size-pre_size+1,pre_size))
    for i in range(len(open_arr) - window_size-pre_size+1):
        X[i, :] = open_arr[i:i+window_size,:]
        label[i] = open_arr[i+window_size:i+window_size+pre_size,0]
    train_X = X[:(-test_length), :]
    train_label = label[:(-test_length)]

    test_X = X[-test_length:, :]
    test_label = label[-test_length:]
    test_X = test_X[pre_size-1:, :]
    test_label = test_label[pre_size-1:] 
   
    valid_length=int(train_X.shape[0]*0.2)
    valid_X = train_X[-valid_length:, :]
    valid_label = train_label[-valid_length:]
    valid_X = valid_X[pre_size-1:, :]
    valid_label = valid_label[pre_size-1:]
    
    train_X = train_X[:-valid_length, :]
    train_label = train_label[:-valid_length]   
    
    return train_X, train_label, valid_X, valid_label, test_X, test_label,  smax[0], smin[0]

train_X, train_label, valid_X, valid_label, test_X, test_label,  smax, smin = get_dataset() 

#打散
train_Xm = np.zeros(shape=(len(train_X), window_size,feature))
train_labelm = np.zeros(shape=(len(train_label), pre_size,1))
indices = np.random.permutation(train_Xm.shape[0])
train_Xm = train_X[indices]
train_labelm = train_label[indices] 

wr = np.zeros(shape=(len(test_label.reshape(-1)),12))

print("--------------------------0.0005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.0005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,0] = pred_test

print("--------------------------0.001-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.001, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,1] = pred_test

print("--------------------------0.005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,2] = pred_test

print("---------------------------0.01----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.01, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,3] = pred_test

print("-----------------------------0.015----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.015, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)   
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,4] = pred_test

print("---------------------------------0.02-------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.02, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,5] = pred_test

print("------------------------------------0.025---------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.025, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)    
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,6] = pred_test

print("--------------------------------------0.03--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.03, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,7] = pred_test

print("--------------------------------------0.035--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.035, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,8] = pred_test

print("--------------------------------------0.04--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.04, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,9] = pred_test

def mae_loss(y_true, y_pred):       
    return K.mean(tf.abs(y_true - y_pred))
print("--------------------------mae-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mae_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,10] = pred_test

def mse_loss(y_true, y_pred):       
    return K.mean(tf.square(y_true - y_pred))
print("--------------------------mse-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mse_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,11] = pred_test

pred_value=pd.DataFrame(data=(np.array(wr)))
pred_value.to_csv("./ex/cnna1__31_10.csv",encoding='utf-8') 


print("---------------------k=0.3,p=1/16--------------------------")
def get_dataset():
    df = pd.read_csv('./data1.csv',usecols=["Z21","T","month","day","hour"])  
    data = np.array(df)
#   数据集总长度
    wi = np.ones(shape=(data.shape[0],feature))
    # 训练总长度
    ab_len=data[:-test_length-window_size,:].shape[0]
    indices = pd.read_csv('./indices32.csv',usecols=["9"])  
    indices =np.array(indices).reshape(-1).astype(int)
#   攻击 k p
    # k随机选10% 20% 30%
    k = 0.3
    p=1/16
    # 测试集用的前168个数据不能受到攻击
    for i in range(int(data[:(-test_length-window_size),:].shape[0]*k)):
        wi[indices[i],0]= 1+p
    data=data*wi   # 受到攻击        
        
    train_data=data[:-test_length,:]
    smax=train_data.max(axis=0)
    smin=train_data.min(axis=0)
    open_arr=((data-smin)/(smax-smin))
    X = np.zeros(shape=(len(open_arr) - window_size-pre_size+1, window_size,feature))
    label = np.zeros(shape=(len(open_arr) - window_size-pre_size+1,pre_size))
    for i in range(len(open_arr) - window_size-pre_size+1):
        X[i, :] = open_arr[i:i+window_size,:]
        label[i] = open_arr[i+window_size:i+window_size+pre_size,0]
    train_X = X[:(-test_length), :]
    train_label = label[:(-test_length)]

    test_X = X[-test_length:, :]
    test_label = label[-test_length:]
    test_X = test_X[pre_size-1:, :]
    test_label = test_label[pre_size-1:] 
   
    valid_length=int(train_X.shape[0]*0.2)
    valid_X = train_X[-valid_length:, :]
    valid_label = train_label[-valid_length:]
    valid_X = valid_X[pre_size-1:, :]
    valid_label = valid_label[pre_size-1:]
    
    train_X = train_X[:-valid_length, :]
    train_label = train_label[:-valid_length]   
    
    return train_X, train_label, valid_X, valid_label, test_X, test_label,  smax[0], smin[0]

train_X, train_label, valid_X, valid_label, test_X, test_label,  smax, smin = get_dataset() 

#打散
train_Xm = np.zeros(shape=(len(train_X), window_size,feature))
train_labelm = np.zeros(shape=(len(train_label), pre_size,1))
indices = np.random.permutation(train_Xm.shape[0])
train_Xm = train_X[indices]
train_labelm = train_label[indices] 

wr = np.zeros(shape=(len(test_label.reshape(-1)),12))

print("--------------------------0.0005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.0005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,0] = pred_test

print("--------------------------0.001-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.001, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,1] = pred_test

print("--------------------------0.005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,2] = pred_test

print("---------------------------0.01----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.01, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,3] = pred_test

print("-----------------------------0.015----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.015, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)   
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,4] = pred_test

print("---------------------------------0.02-------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.02, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,5] = pred_test

print("------------------------------------0.025---------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.025, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)    
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,6] = pred_test

print("--------------------------------------0.03--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.03, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,7] = pred_test

print("--------------------------------------0.035--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.035, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,8] = pred_test

print("--------------------------------------0.04--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.04, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,9] = pred_test

def mae_loss(y_true, y_pred):       
    return K.mean(tf.abs(y_true - y_pred))
print("--------------------------mae-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mae_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,10] = pred_test

def mse_loss(y_true, y_pred):       
    return K.mean(tf.square(y_true - y_pred))
print("--------------------------mse-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mse_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,11] = pred_test

pred_value=pd.DataFrame(data=(np.array(wr)))
pred_value.to_csv("./ex/cnna1__32_10.csv",encoding='utf-8') 


print("---------------------k=0.3,p=1/8--------------------------")
def get_dataset():
    df = pd.read_csv('./data1.csv',usecols=["Z21","T","month","day","hour"])  
    data = np.array(df)
#   数据集总长度
    wi = np.ones(shape=(data.shape[0],feature))
    # 训练总长度
    ab_len=data[:-test_length-window_size,:].shape[0]
    indices = pd.read_csv('./indices33.csv',usecols=["9"])  
    indices =np.array(indices).reshape(-1).astype(int)
#   攻击 k p
    # k随机选10% 20% 30%
    k = 0.3
    p=1/8
    # 测试集用的前168个数据不能受到攻击
    for i in range(int(data[:(-test_length-window_size),:].shape[0]*k)):
        wi[indices[i],0]= 1+p
    data=data*wi   # 受到攻击        
        
    train_data=data[:-test_length,:]
    smax=train_data.max(axis=0)
    smin=train_data.min(axis=0)
    open_arr=((data-smin)/(smax-smin))
    X = np.zeros(shape=(len(open_arr) - window_size-pre_size+1, window_size,feature))
    label = np.zeros(shape=(len(open_arr) - window_size-pre_size+1,pre_size))
    for i in range(len(open_arr) - window_size-pre_size+1):
        X[i, :] = open_arr[i:i+window_size,:]
        label[i] = open_arr[i+window_size:i+window_size+pre_size,0]
    train_X = X[:(-test_length), :]
    train_label = label[:(-test_length)]

    test_X = X[-test_length:, :]
    test_label = label[-test_length:]
    test_X = test_X[pre_size-1:, :]
    test_label = test_label[pre_size-1:] 
   
    valid_length=int(train_X.shape[0]*0.2)
    valid_X = train_X[-valid_length:, :]
    valid_label = train_label[-valid_length:]
    valid_X = valid_X[pre_size-1:, :]
    valid_label = valid_label[pre_size-1:]
    
    train_X = train_X[:-valid_length, :]
    train_label = train_label[:-valid_length]   
    
    return train_X, train_label, valid_X, valid_label, test_X, test_label,  smax[0], smin[0]

train_X, train_label, valid_X, valid_label, test_X, test_label,  smax, smin = get_dataset() 

#打散
train_Xm = np.zeros(shape=(len(train_X), window_size,feature))
train_labelm = np.zeros(shape=(len(train_label), pre_size,1))
indices = np.random.permutation(train_Xm.shape[0])
train_Xm = train_X[indices]
train_labelm = train_label[indices] 

wr = np.zeros(shape=(len(test_label.reshape(-1)),12))

print("--------------------------0.0005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.0005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,0] = pred_test

print("--------------------------0.001-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.001, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,1] = pred_test

print("--------------------------0.005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,2] = pred_test

print("---------------------------0.01----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.01, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,3] = pred_test

print("-----------------------------0.015----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.015, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)   
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,4] = pred_test

print("---------------------------------0.02-------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.02, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,5] = pred_test

print("------------------------------------0.025---------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.025, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)    
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,6] = pred_test

print("--------------------------------------0.03--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.03, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,7] = pred_test

print("--------------------------------------0.035--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.035, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,8] = pred_test

print("--------------------------------------0.04--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.04, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,9] = pred_test

def mae_loss(y_true, y_pred):       
    return K.mean(tf.abs(y_true - y_pred))
print("--------------------------mae-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mae_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,10] = pred_test

def mse_loss(y_true, y_pred):       
    return K.mean(tf.square(y_true - y_pred))
print("--------------------------mse-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mse_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,11] = pred_test

pred_value=pd.DataFrame(data=(np.array(wr)))
pred_value.to_csv("./ex/cnna1__33_10.csv",encoding='utf-8') 


print("---------------------k=0.3,p=1/4--------------------------")
def get_dataset():
    df = pd.read_csv('./data1.csv',usecols=["Z21","T","month","day","hour"])  
    data = np.array(df)
#   数据集总长度
    wi = np.ones(shape=(data.shape[0],feature))
    # 训练总长度
    ab_len=data[:-test_length-window_size,:].shape[0]
    indices = pd.read_csv('./indices34.csv',usecols=["9"])  
    indices =np.array(indices).reshape(-1).astype(int)
#   攻击 k p
    # k随机选10% 20% 30%
    k = 0.3
    p=1/4
    # 测试集用的前168个数据不能受到攻击
    for i in range(int(data[:(-test_length-window_size),:].shape[0]*k)):
        wi[indices[i],0]= 1+p
    data=data*wi   # 受到攻击        
        
    train_data=data[:-test_length,:]
    smax=train_data.max(axis=0)
    smin=train_data.min(axis=0)
    open_arr=((data-smin)/(smax-smin))
    X = np.zeros(shape=(len(open_arr) - window_size-pre_size+1, window_size,feature))
    label = np.zeros(shape=(len(open_arr) - window_size-pre_size+1,pre_size))
    for i in range(len(open_arr) - window_size-pre_size+1):
        X[i, :] = open_arr[i:i+window_size,:]
        label[i] = open_arr[i+window_size:i+window_size+pre_size,0]
    train_X = X[:(-test_length), :]
    train_label = label[:(-test_length)]

    test_X = X[-test_length:, :]
    test_label = label[-test_length:]
    test_X = test_X[pre_size-1:, :]
    test_label = test_label[pre_size-1:] 
   
    valid_length=int(train_X.shape[0]*0.2)
    valid_X = train_X[-valid_length:, :]
    valid_label = train_label[-valid_length:]
    valid_X = valid_X[pre_size-1:, :]
    valid_label = valid_label[pre_size-1:]
    
    train_X = train_X[:-valid_length, :]
    train_label = train_label[:-valid_length]   
    
    return train_X, train_label, valid_X, valid_label, test_X, test_label,  smax[0], smin[0]

train_X, train_label, valid_X, valid_label, test_X, test_label,  smax, smin = get_dataset() 

#打散
train_Xm = np.zeros(shape=(len(train_X), window_size,feature))
train_labelm = np.zeros(shape=(len(train_label), pre_size,1))
indices = np.random.permutation(train_Xm.shape[0])
train_Xm = train_X[indices]
train_labelm = train_label[indices] 

wr = np.zeros(shape=(len(test_label.reshape(-1)),12))

print("--------------------------0.0005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.0005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,0] = pred_test

print("--------------------------0.001-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.001, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,1] = pred_test

print("--------------------------0.005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,2] = pred_test

print("---------------------------0.01----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.01, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,3] = pred_test

print("-----------------------------0.015----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.015, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)   
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,4] = pred_test

print("---------------------------------0.02-------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.02, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,5] = pred_test

print("------------------------------------0.025---------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.025, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)    
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,6] = pred_test

print("--------------------------------------0.03--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.03, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,7] = pred_test

print("--------------------------------------0.035--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.035, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,8] = pred_test

print("--------------------------------------0.04--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.04, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,9] = pred_test

def mae_loss(y_true, y_pred):       
    return K.mean(tf.abs(y_true - y_pred))
print("--------------------------mae-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mae_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,10] = pred_test

def mse_loss(y_true, y_pred):       
    return K.mean(tf.square(y_true - y_pred))
print("--------------------------mse-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mse_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,11] = pred_test

pred_value=pd.DataFrame(data=(np.array(wr)))
pred_value.to_csv("./ex/cnna1__34_10.csv",encoding='utf-8') 



print("---------------------k=0.3,p=1/2--------------------------")
def get_dataset():
    df = pd.read_csv('./data1.csv',usecols=["Z21","T","month","day","hour"])  
    data = np.array(df)
#   数据集总长度
    wi = np.ones(shape=(data.shape[0],feature))
    # 训练总长度
    ab_len=data[:-test_length-window_size,:].shape[0]
    indices = pd.read_csv('./indices35.csv',usecols=["9"])  
    indices =np.array(indices).reshape(-1).astype(int)
#   攻击 k p
    # k随机选10% 20% 30%
    k = 0.3
    p=1/2
    # 测试集用的前168个数据不能受到攻击
    for i in range(int(data[:(-test_length-window_size),:].shape[0]*k)):
        wi[indices[i],0]= 1+p
    data=data*wi   # 受到攻击        
        
    train_data=data[:-test_length,:]
    smax=train_data.max(axis=0)
    smin=train_data.min(axis=0)
    open_arr=((data-smin)/(smax-smin))
    X = np.zeros(shape=(len(open_arr) - window_size-pre_size+1, window_size,feature))
    label = np.zeros(shape=(len(open_arr) - window_size-pre_size+1,pre_size))
    for i in range(len(open_arr) - window_size-pre_size+1):
        X[i, :] = open_arr[i:i+window_size,:]
        label[i] = open_arr[i+window_size:i+window_size+pre_size,0]
    train_X = X[:(-test_length), :]
    train_label = label[:(-test_length)]

    test_X = X[-test_length:, :]
    test_label = label[-test_length:]
    test_X = test_X[pre_size-1:, :]
    test_label = test_label[pre_size-1:] 
   
    valid_length=int(train_X.shape[0]*0.2)
    valid_X = train_X[-valid_length:, :]
    valid_label = train_label[-valid_length:]
    valid_X = valid_X[pre_size-1:, :]
    valid_label = valid_label[pre_size-1:]
    
    train_X = train_X[:-valid_length, :]
    train_label = train_label[:-valid_length]   
    
    return train_X, train_label, valid_X, valid_label, test_X, test_label,  smax[0], smin[0]

train_X, train_label, valid_X, valid_label, test_X, test_label,  smax, smin = get_dataset() 

#打散
train_Xm = np.zeros(shape=(len(train_X), window_size,feature))
train_labelm = np.zeros(shape=(len(train_label), pre_size,1))
indices = np.random.permutation(train_Xm.shape[0])
train_Xm = train_X[indices]
train_labelm = train_label[indices] 

wr = np.zeros(shape=(len(test_label.reshape(-1)),12))

print("--------------------------0.0005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.0005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,0] = pred_test

print("--------------------------0.001-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.001, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,1] = pred_test

print("--------------------------0.005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,2] = pred_test

print("---------------------------0.01----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.01, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,3] = pred_test

print("-----------------------------0.015----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.015, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)   
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,4] = pred_test

print("---------------------------------0.02-------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.02, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,5] = pred_test

print("------------------------------------0.025---------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.025, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)    
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,6] = pred_test

print("--------------------------------------0.03--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.03, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,7] = pred_test

print("--------------------------------------0.035--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.035, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,8] = pred_test

print("--------------------------------------0.04--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.04, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,9] = pred_test

def mae_loss(y_true, y_pred):       
    return K.mean(tf.abs(y_true - y_pred))
print("--------------------------mae-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mae_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,10] = pred_test

def mse_loss(y_true, y_pred):       
    return K.mean(tf.square(y_true - y_pred))
print("--------------------------mse-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mse_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,11] = pred_test

pred_value=pd.DataFrame(data=(np.array(wr)))
pred_value.to_csv("./ex/cnna1__35_10.csv",encoding='utf-8') 


print("---------------------k=0.3,p=1--------------------------")
def get_dataset():
    df = pd.read_csv('./data1.csv',usecols=["Z21","T","month","day","hour"])  
    data = np.array(df)
#   数据集总长度
    wi = np.ones(shape=(data.shape[0],feature))
    # 训练总长度
    ab_len=data[:-test_length-window_size,:].shape[0]
    indices = pd.read_csv('./indices36.csv',usecols=["9"])  
    indices =np.array(indices).reshape(-1).astype(int)
#   攻击 k p
    # k随机选10% 20% 30%
    k = 0.3
    p=1
    # 测试集用的前168个数据不能受到攻击
    for i in range(int(data[:(-test_length-window_size),:].shape[0]*k)):
        wi[indices[i],0]= 1+p
    data=data*wi   # 受到攻击        
        
    train_data=data[:-test_length,:]
    smax=train_data.max(axis=0)
    smin=train_data.min(axis=0)
    open_arr=((data-smin)/(smax-smin))
    X = np.zeros(shape=(len(open_arr) - window_size-pre_size+1, window_size,feature))
    label = np.zeros(shape=(len(open_arr) - window_size-pre_size+1,pre_size))
    for i in range(len(open_arr) - window_size-pre_size+1):
        X[i, :] = open_arr[i:i+window_size,:]
        label[i] = open_arr[i+window_size:i+window_size+pre_size,0]
    train_X = X[:(-test_length), :]
    train_label = label[:(-test_length)]

    test_X = X[-test_length:, :]
    test_label = label[-test_length:]
    test_X = test_X[pre_size-1:, :]
    test_label = test_label[pre_size-1:] 
   
    valid_length=int(train_X.shape[0]*0.2)
    valid_X = train_X[-valid_length:, :]
    valid_label = train_label[-valid_length:]
    valid_X = valid_X[pre_size-1:, :]
    valid_label = valid_label[pre_size-1:]
    
    train_X = train_X[:-valid_length, :]
    train_label = train_label[:-valid_length]   
    
    return train_X, train_label, valid_X, valid_label, test_X, test_label,  smax[0], smin[0]

train_X, train_label, valid_X, valid_label, test_X, test_label,  smax, smin = get_dataset() 

#打散
train_Xm = np.zeros(shape=(len(train_X), window_size,feature))
train_labelm = np.zeros(shape=(len(train_label), pre_size,1))
indices = np.random.permutation(train_Xm.shape[0])
train_Xm = train_X[indices]
train_labelm = train_label[indices] 

wr = np.zeros(shape=(len(test_label.reshape(-1)),12))

print("--------------------------0.0005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.0005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,0] = pred_test

print("--------------------------0.001-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.001, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,1] = pred_test

print("--------------------------0.005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,2] = pred_test

print("---------------------------0.01----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.01, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,3] = pred_test

print("-----------------------------0.015----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.015, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)   
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,4] = pred_test

print("---------------------------------0.02-------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.02, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,5] = pred_test

print("------------------------------------0.025---------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.025, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)    
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,6] = pred_test

print("--------------------------------------0.03--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.03, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,7] = pred_test

print("--------------------------------------0.035--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.035, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,8] = pred_test

print("--------------------------------------0.04--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.04, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,9] = pred_test

def mae_loss(y_true, y_pred):       
    return K.mean(tf.abs(y_true - y_pred))
print("--------------------------mae-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mae_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,10] = pred_test

def mse_loss(y_true, y_pred):       
    return K.mean(tf.square(y_true - y_pred))
print("--------------------------mse-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mse_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,11] = pred_test

pred_value=pd.DataFrame(data=(np.array(wr)))
pred_value.to_csv("./ex/cnna1__36_10.csv",encoding='utf-8') 


print("---------------------k=0.4,p=1/32--------------------------")
def get_dataset():
    df = pd.read_csv('./data1.csv',usecols=["Z21","T","month","day","hour"])  
    data = np.array(df)
#   数据集总长度
    wi = np.ones(shape=(data.shape[0],feature))
    # 训练总长度
    ab_len=data[:-test_length-window_size,:].shape[0]
    indices = pd.read_csv('./indices41.csv',usecols=["9"])  
    indices =np.array(indices).reshape(-1).astype(int)
#   攻击 k p
    # k随机选10% 20% 30%
    k = 0.4
    p=1/32
    # 测试集用的前168个数据不能受到攻击
    for i in range(int(data[:(-test_length-window_size),:].shape[0]*k)):
        wi[indices[i],0]= 1+p
    data=data*wi   # 受到攻击        
        
    train_data=data[:-test_length,:]
    smax=train_data.max(axis=0)
    smin=train_data.min(axis=0)
    open_arr=((data-smin)/(smax-smin))
    X = np.zeros(shape=(len(open_arr) - window_size-pre_size+1, window_size,feature))
    label = np.zeros(shape=(len(open_arr) - window_size-pre_size+1,pre_size))
    for i in range(len(open_arr) - window_size-pre_size+1):
        X[i, :] = open_arr[i:i+window_size,:]
        label[i] = open_arr[i+window_size:i+window_size+pre_size,0]
    train_X = X[:(-test_length), :]
    train_label = label[:(-test_length)]

    test_X = X[-test_length:, :]
    test_label = label[-test_length:]
    test_X = test_X[pre_size-1:, :]
    test_label = test_label[pre_size-1:] 
   
    valid_length=int(train_X.shape[0]*0.2)
    valid_X = train_X[-valid_length:, :]
    valid_label = train_label[-valid_length:]
    valid_X = valid_X[pre_size-1:, :]
    valid_label = valid_label[pre_size-1:]
    
    train_X = train_X[:-valid_length, :]
    train_label = train_label[:-valid_length]   
    
    return train_X, train_label, valid_X, valid_label, test_X, test_label,  smax[0], smin[0]

train_X, train_label, valid_X, valid_label, test_X, test_label,  smax, smin = get_dataset() 

#打散
train_Xm = np.zeros(shape=(len(train_X), window_size,feature))
train_labelm = np.zeros(shape=(len(train_label), pre_size,1))
indices = np.random.permutation(train_Xm.shape[0])
train_Xm = train_X[indices]
train_labelm = train_label[indices] 

wr = np.zeros(shape=(len(test_label.reshape(-1)),12))

print("--------------------------0.0005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.0005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,0] = pred_test

print("--------------------------0.001-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.001, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,1] = pred_test

print("--------------------------0.005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,2] = pred_test

print("---------------------------0.01----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.01, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,3] = pred_test

print("-----------------------------0.015----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.015, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)   
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,4] = pred_test

print("---------------------------------0.02-------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.02, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,5] = pred_test

print("------------------------------------0.025---------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.025, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)    
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,6] = pred_test

print("--------------------------------------0.03--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.03, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,7] = pred_test

print("--------------------------------------0.035--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.035, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,8] = pred_test

print("--------------------------------------0.04--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.04, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,9] = pred_test

def mae_loss(y_true, y_pred):       
    return K.mean(tf.abs(y_true - y_pred))
print("--------------------------mae-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mae_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,10] = pred_test

def mse_loss(y_true, y_pred):       
    return K.mean(tf.square(y_true - y_pred))
print("--------------------------mse-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mse_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,11] = pred_test

pred_value=pd.DataFrame(data=(np.array(wr)))
pred_value.to_csv("./ex/cnna1__41_10.csv",encoding='utf-8') 


print("---------------------k=0.4,p=1/16--------------------------")
def get_dataset():
    df = pd.read_csv('./data1.csv',usecols=["Z21","T","month","day","hour"])  
    data = np.array(df)
#   数据集总长度
    wi = np.ones(shape=(data.shape[0],feature))
    # 训练总长度
    ab_len=data[:-test_length-window_size,:].shape[0]
    indices = pd.read_csv('./indices42.csv',usecols=["9"])  
    indices =np.array(indices).reshape(-1).astype(int)
#   攻击 k p
    # k随机选10% 20% 30%
    k = 0.4
    p=1/16
    # 测试集用的前168个数据不能受到攻击
    for i in range(int(data[:(-test_length-window_size),:].shape[0]*k)):
        wi[indices[i],0]= 1+p
    data=data*wi   # 受到攻击        
        
    train_data=data[:-test_length,:]
    smax=train_data.max(axis=0)
    smin=train_data.min(axis=0)
    open_arr=((data-smin)/(smax-smin))
    X = np.zeros(shape=(len(open_arr) - window_size-pre_size+1, window_size,feature))
    label = np.zeros(shape=(len(open_arr) - window_size-pre_size+1,pre_size))
    for i in range(len(open_arr) - window_size-pre_size+1):
        X[i, :] = open_arr[i:i+window_size,:]
        label[i] = open_arr[i+window_size:i+window_size+pre_size,0]
    train_X = X[:(-test_length), :]
    train_label = label[:(-test_length)]

    test_X = X[-test_length:, :]
    test_label = label[-test_length:]
    test_X = test_X[pre_size-1:, :]
    test_label = test_label[pre_size-1:] 
   
    valid_length=int(train_X.shape[0]*0.2)
    valid_X = train_X[-valid_length:, :]
    valid_label = train_label[-valid_length:]
    valid_X = valid_X[pre_size-1:, :]
    valid_label = valid_label[pre_size-1:]
    
    train_X = train_X[:-valid_length, :]
    train_label = train_label[:-valid_length]   
    
    return train_X, train_label, valid_X, valid_label, test_X, test_label,  smax[0], smin[0]

train_X, train_label, valid_X, valid_label, test_X, test_label,  smax, smin = get_dataset() 

#打散
train_Xm = np.zeros(shape=(len(train_X), window_size,feature))
train_labelm = np.zeros(shape=(len(train_label), pre_size,1))
indices = np.random.permutation(train_Xm.shape[0])
train_Xm = train_X[indices]
train_labelm = train_label[indices] 

wr = np.zeros(shape=(len(test_label.reshape(-1)),12))

print("--------------------------0.0005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.0005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,0] = pred_test

print("--------------------------0.001-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.001, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,1] = pred_test

print("--------------------------0.005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,2] = pred_test

print("---------------------------0.01----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.01, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,3] = pred_test

print("-----------------------------0.015----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.015, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)   
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,4] = pred_test

print("---------------------------------0.02-------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.02, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,5] = pred_test

print("------------------------------------0.025---------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.025, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)    
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,6] = pred_test

print("--------------------------------------0.03--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.03, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,7] = pred_test

print("--------------------------------------0.035--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.035, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,8] = pred_test

print("--------------------------------------0.04--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.04, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,9] = pred_test

def mae_loss(y_true, y_pred):       
    return K.mean(tf.abs(y_true - y_pred))
print("--------------------------mae-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mae_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,10] = pred_test

def mse_loss(y_true, y_pred):       
    return K.mean(tf.square(y_true - y_pred))
print("--------------------------mse-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mse_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,11] = pred_test

pred_value=pd.DataFrame(data=(np.array(wr)))
pred_value.to_csv("./ex/cnna1__42_10.csv",encoding='utf-8') 


print("---------------------k=0.4,p=1/8--------------------------")
def get_dataset():
    df = pd.read_csv('./data1.csv',usecols=["Z21","T","month","day","hour"])  
    data = np.array(df)
#   数据集总长度
    wi = np.ones(shape=(data.shape[0],feature))
    # 训练总长度
    ab_len=data[:-test_length-window_size,:].shape[0]
    indices = pd.read_csv('./indices43.csv',usecols=["9"])  
    indices =np.array(indices).reshape(-1).astype(int)
#   攻击 k p
    # k随机选10% 20% 30%
    k = 0.4
    p=1/8
    # 测试集用的前168个数据不能受到攻击
    for i in range(int(data[:(-test_length-window_size),:].shape[0]*k)):
        wi[indices[i],0]= 1+p
    data=data*wi   # 受到攻击        
        
    train_data=data[:-test_length,:]
    smax=train_data.max(axis=0)
    smin=train_data.min(axis=0)
    open_arr=((data-smin)/(smax-smin))
    X = np.zeros(shape=(len(open_arr) - window_size-pre_size+1, window_size,feature))
    label = np.zeros(shape=(len(open_arr) - window_size-pre_size+1,pre_size))
    for i in range(len(open_arr) - window_size-pre_size+1):
        X[i, :] = open_arr[i:i+window_size,:]
        label[i] = open_arr[i+window_size:i+window_size+pre_size,0]
    train_X = X[:(-test_length), :]
    train_label = label[:(-test_length)]

    test_X = X[-test_length:, :]
    test_label = label[-test_length:]
    test_X = test_X[pre_size-1:, :]
    test_label = test_label[pre_size-1:] 
   
    valid_length=int(train_X.shape[0]*0.2)
    valid_X = train_X[-valid_length:, :]
    valid_label = train_label[-valid_length:]
    valid_X = valid_X[pre_size-1:, :]
    valid_label = valid_label[pre_size-1:]
    
    train_X = train_X[:-valid_length, :]
    train_label = train_label[:-valid_length]   
    
    return train_X, train_label, valid_X, valid_label, test_X, test_label,  smax[0], smin[0]

train_X, train_label, valid_X, valid_label, test_X, test_label,  smax, smin = get_dataset() 

#打散
train_Xm = np.zeros(shape=(len(train_X), window_size,feature))
train_labelm = np.zeros(shape=(len(train_label), pre_size,1))
indices = np.random.permutation(train_Xm.shape[0])
train_Xm = train_X[indices]
train_labelm = train_label[indices] 

wr = np.zeros(shape=(len(test_label.reshape(-1)),12))

print("--------------------------0.0005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.0005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,0] = pred_test

print("--------------------------0.001-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.001, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,1] = pred_test

print("--------------------------0.005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,2] = pred_test

print("---------------------------0.01----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.01, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,3] = pred_test

print("-----------------------------0.015----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.015, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)   
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,4] = pred_test

print("---------------------------------0.02-------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.02, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,5] = pred_test

print("------------------------------------0.025---------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.025, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)    
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,6] = pred_test

print("--------------------------------------0.03--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.03, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,7] = pred_test

print("--------------------------------------0.035--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.035, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,8] = pred_test

print("--------------------------------------0.04--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.04, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,9] = pred_test

def mae_loss(y_true, y_pred):       
    return K.mean(tf.abs(y_true - y_pred))
print("--------------------------mae-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mae_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,10] = pred_test

def mse_loss(y_true, y_pred):       
    return K.mean(tf.square(y_true - y_pred))
print("--------------------------mse-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mse_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,11] = pred_test

pred_value=pd.DataFrame(data=(np.array(wr)))
pred_value.to_csv("./ex/cnna1__43_10.csv",encoding='utf-8') 


print("---------------------k=0.4,p=1/4--------------------------")
def get_dataset():
    df = pd.read_csv('./data1.csv',usecols=["Z21","T","month","day","hour"])  
    data = np.array(df)
#   数据集总长度
    wi = np.ones(shape=(data.shape[0],feature))
    # 训练总长度
    ab_len=data[:-test_length-window_size,:].shape[0]
    indices = pd.read_csv('./indices44.csv',usecols=["9"])  
    indices =np.array(indices).reshape(-1).astype(int)
#   攻击 k p
    # k随机选10% 20% 30%
    k = 0.4
    p=1/4
    # 测试集用的前168个数据不能受到攻击
    for i in range(int(data[:(-test_length-window_size),:].shape[0]*k)):
        wi[indices[i],0]= 1+p
    data=data*wi   # 受到攻击        
        
    train_data=data[:-test_length,:]
    smax=train_data.max(axis=0)
    smin=train_data.min(axis=0)
    open_arr=((data-smin)/(smax-smin))
    X = np.zeros(shape=(len(open_arr) - window_size-pre_size+1, window_size,feature))
    label = np.zeros(shape=(len(open_arr) - window_size-pre_size+1,pre_size))
    for i in range(len(open_arr) - window_size-pre_size+1):
        X[i, :] = open_arr[i:i+window_size,:]
        label[i] = open_arr[i+window_size:i+window_size+pre_size,0]
    train_X = X[:(-test_length), :]
    train_label = label[:(-test_length)]

    test_X = X[-test_length:, :]
    test_label = label[-test_length:]
    test_X = test_X[pre_size-1:, :]
    test_label = test_label[pre_size-1:] 
   
    valid_length=int(train_X.shape[0]*0.2)
    valid_X = train_X[-valid_length:, :]
    valid_label = train_label[-valid_length:]
    valid_X = valid_X[pre_size-1:, :]
    valid_label = valid_label[pre_size-1:]
    
    train_X = train_X[:-valid_length, :]
    train_label = train_label[:-valid_length]   
    
    return train_X, train_label, valid_X, valid_label, test_X, test_label,  smax[0], smin[0]

train_X, train_label, valid_X, valid_label, test_X, test_label,  smax, smin = get_dataset() 

#打散
train_Xm = np.zeros(shape=(len(train_X), window_size,feature))
train_labelm = np.zeros(shape=(len(train_label), pre_size,1))
indices = np.random.permutation(train_Xm.shape[0])
train_Xm = train_X[indices]
train_labelm = train_label[indices] 

wr = np.zeros(shape=(len(test_label.reshape(-1)),12))

print("--------------------------0.0005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.0005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,0] = pred_test

print("--------------------------0.001-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.001, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,1] = pred_test

print("--------------------------0.005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,2] = pred_test

print("---------------------------0.01----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.01, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,3] = pred_test

print("-----------------------------0.015----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.015, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)   
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,4] = pred_test

print("---------------------------------0.02-------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.02, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,5] = pred_test

print("------------------------------------0.025---------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.025, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)    
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,6] = pred_test

print("--------------------------------------0.03--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.03, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,7] = pred_test

print("--------------------------------------0.035--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.035, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,8] = pred_test

print("--------------------------------------0.04--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.04, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,9] = pred_test

def mae_loss(y_true, y_pred):       
    return K.mean(tf.abs(y_true - y_pred))
print("--------------------------mae-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mae_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,10] = pred_test

def mse_loss(y_true, y_pred):       
    return K.mean(tf.square(y_true - y_pred))
print("--------------------------mse-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mse_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,11] = pred_test

pred_value=pd.DataFrame(data=(np.array(wr)))
pred_value.to_csv("./ex/cnna1__44_10.csv",encoding='utf-8') 



print("---------------------k=0.4,p=1/2--------------------------")
def get_dataset():
    df = pd.read_csv('./data1.csv',usecols=["Z21","T","month","day","hour"])  
    data = np.array(df)
#   数据集总长度
    wi = np.ones(shape=(data.shape[0],feature))
    # 训练总长度
    ab_len=data[:-test_length-window_size,:].shape[0]
    indices = pd.read_csv('./indices45.csv',usecols=["9"])  
    indices =np.array(indices).reshape(-1).astype(int)
#   攻击 k p
    # k随机选10% 20% 30%
    k = 0.4
    p=1/2
    # 测试集用的前168个数据不能受到攻击
    for i in range(int(data[:(-test_length-window_size),:].shape[0]*k)):
        wi[indices[i],0]= 1+p
    data=data*wi   # 受到攻击        
        
    train_data=data[:-test_length,:]
    smax=train_data.max(axis=0)
    smin=train_data.min(axis=0)
    open_arr=((data-smin)/(smax-smin))
    X = np.zeros(shape=(len(open_arr) - window_size-pre_size+1, window_size,feature))
    label = np.zeros(shape=(len(open_arr) - window_size-pre_size+1,pre_size))
    for i in range(len(open_arr) - window_size-pre_size+1):
        X[i, :] = open_arr[i:i+window_size,:]
        label[i] = open_arr[i+window_size:i+window_size+pre_size,0]
    train_X = X[:(-test_length), :]
    train_label = label[:(-test_length)]

    test_X = X[-test_length:, :]
    test_label = label[-test_length:]
    test_X = test_X[pre_size-1:, :]
    test_label = test_label[pre_size-1:] 
   
    valid_length=int(train_X.shape[0]*0.2)
    valid_X = train_X[-valid_length:, :]
    valid_label = train_label[-valid_length:]
    valid_X = valid_X[pre_size-1:, :]
    valid_label = valid_label[pre_size-1:]
    
    train_X = train_X[:-valid_length, :]
    train_label = train_label[:-valid_length]   
    
    return train_X, train_label, valid_X, valid_label, test_X, test_label,  smax[0], smin[0]

train_X, train_label, valid_X, valid_label, test_X, test_label,  smax, smin = get_dataset() 

#打散
train_Xm = np.zeros(shape=(len(train_X), window_size,feature))
train_labelm = np.zeros(shape=(len(train_label), pre_size,1))
indices = np.random.permutation(train_Xm.shape[0])
train_Xm = train_X[indices]
train_labelm = train_label[indices] 

wr = np.zeros(shape=(len(test_label.reshape(-1)),12))

print("--------------------------0.0005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.0005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,0] = pred_test

print("--------------------------0.001-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.001, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,1] = pred_test

print("--------------------------0.005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,2] = pred_test

print("---------------------------0.01----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.01, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,3] = pred_test

print("-----------------------------0.015----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.015, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)   
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,4] = pred_test

print("---------------------------------0.02-------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.02, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,5] = pred_test

print("------------------------------------0.025---------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.025, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)    
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,6] = pred_test

print("--------------------------------------0.03--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.03, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,7] = pred_test

print("--------------------------------------0.035--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.035, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,8] = pred_test

print("--------------------------------------0.04--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.04, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,9] = pred_test

def mae_loss(y_true, y_pred):       
    return K.mean(tf.abs(y_true - y_pred))
print("--------------------------mae-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mae_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,10] = pred_test

def mse_loss(y_true, y_pred):       
    return K.mean(tf.square(y_true - y_pred))
print("--------------------------mse-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mse_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,11] = pred_test

pred_value=pd.DataFrame(data=(np.array(wr)))
pred_value.to_csv("./ex/cnna1__45_10.csv",encoding='utf-8') 


print("---------------------k=0.4,p=1--------------------------")
def get_dataset():
    df = pd.read_csv('./data1.csv',usecols=["Z21","T","month","day","hour"])  
    data = np.array(df)
#   数据集总长度
    wi = np.ones(shape=(data.shape[0],feature))
    # 训练总长度
    ab_len=data[:-test_length-window_size,:].shape[0]
    indices = pd.read_csv('./indices46.csv',usecols=["9"])  
    indices =np.array(indices).reshape(-1).astype(int)
#   攻击 k p
    # k随机选10% 20% 30%
    k = 0.4
    p=1
    # 测试集用的前168个数据不能受到攻击
    for i in range(int(data[:(-test_length-window_size),:].shape[0]*k)):
        wi[indices[i],0]= 1+p
    data=data*wi   # 受到攻击        
        
    train_data=data[:-test_length,:]
    smax=train_data.max(axis=0)
    smin=train_data.min(axis=0)
    open_arr=((data-smin)/(smax-smin))
    X = np.zeros(shape=(len(open_arr) - window_size-pre_size+1, window_size,feature))
    label = np.zeros(shape=(len(open_arr) - window_size-pre_size+1,pre_size))
    for i in range(len(open_arr) - window_size-pre_size+1):
        X[i, :] = open_arr[i:i+window_size,:]
        label[i] = open_arr[i+window_size:i+window_size+pre_size,0]
    train_X = X[:(-test_length), :]
    train_label = label[:(-test_length)]

    test_X = X[-test_length:, :]
    test_label = label[-test_length:]
    test_X = test_X[pre_size-1:, :]
    test_label = test_label[pre_size-1:] 
   
    valid_length=int(train_X.shape[0]*0.2)
    valid_X = train_X[-valid_length:, :]
    valid_label = train_label[-valid_length:]
    valid_X = valid_X[pre_size-1:, :]
    valid_label = valid_label[pre_size-1:]
    
    train_X = train_X[:-valid_length, :]
    train_label = train_label[:-valid_length]   
    
    return train_X, train_label, valid_X, valid_label, test_X, test_label,  smax[0], smin[0]

train_X, train_label, valid_X, valid_label, test_X, test_label,  smax, smin = get_dataset() 

#打散
train_Xm = np.zeros(shape=(len(train_X), window_size,feature))
train_labelm = np.zeros(shape=(len(train_label), pre_size,1))
indices = np.random.permutation(train_Xm.shape[0])
train_Xm = train_X[indices]
train_labelm = train_label[indices] 

wr = np.zeros(shape=(len(test_label.reshape(-1)),12))

print("--------------------------0.0005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.0005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,0] = pred_test

print("--------------------------0.001-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.001, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,1] = pred_test

print("--------------------------0.005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,2] = pred_test

print("---------------------------0.01----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.01, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,3] = pred_test

print("-----------------------------0.015----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.015, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)   
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,4] = pred_test

print("---------------------------------0.02-------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.02, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,5] = pred_test

print("------------------------------------0.025---------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.025, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)    
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,6] = pred_test

print("--------------------------------------0.03--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.03, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,7] = pred_test

print("--------------------------------------0.035--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.035, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,8] = pred_test

print("--------------------------------------0.04--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.04, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,9] = pred_test

def mae_loss(y_true, y_pred):       
    return K.mean(tf.abs(y_true - y_pred))
print("--------------------------mae-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mae_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,10] = pred_test

def mse_loss(y_true, y_pred):       
    return K.mean(tf.square(y_true - y_pred))
print("--------------------------mse-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mse_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,11] = pred_test

pred_value=pd.DataFrame(data=(np.array(wr)))
pred_value.to_csv("./ex/cnna1__46_10.csv",encoding='utf-8') 


print("---------------------k=0.5,p=1/32--------------------------")
def get_dataset():
    df = pd.read_csv('./data1.csv',usecols=["Z21","T","month","day","hour"])  
    data = np.array(df)
#   数据集总长度
    wi = np.ones(shape=(data.shape[0],feature))
    # 训练总长度
    ab_len=data[:-test_length-window_size,:].shape[0]
    indices = pd.read_csv('./indices51.csv',usecols=["9"])  
    indices =np.array(indices).reshape(-1).astype(int)
#   攻击 k p
    # k随机选10% 20% 30%
    k = 0.5
    p=1/32
    # 测试集用的前168个数据不能受到攻击
    for i in range(int(data[:(-test_length-window_size),:].shape[0]*k)):
        wi[indices[i],0]= 1+p
    data=data*wi   # 受到攻击        
        
    train_data=data[:-test_length,:]
    smax=train_data.max(axis=0)
    smin=train_data.min(axis=0)
    open_arr=((data-smin)/(smax-smin))
    X = np.zeros(shape=(len(open_arr) - window_size-pre_size+1, window_size,feature))
    label = np.zeros(shape=(len(open_arr) - window_size-pre_size+1,pre_size))
    for i in range(len(open_arr) - window_size-pre_size+1):
        X[i, :] = open_arr[i:i+window_size,:]
        label[i] = open_arr[i+window_size:i+window_size+pre_size,0]
    train_X = X[:(-test_length), :]
    train_label = label[:(-test_length)]

    test_X = X[-test_length:, :]
    test_label = label[-test_length:]
    test_X = test_X[pre_size-1:, :]
    test_label = test_label[pre_size-1:] 
   
    valid_length=int(train_X.shape[0]*0.2)
    valid_X = train_X[-valid_length:, :]
    valid_label = train_label[-valid_length:]
    valid_X = valid_X[pre_size-1:, :]
    valid_label = valid_label[pre_size-1:]
    
    train_X = train_X[:-valid_length, :]
    train_label = train_label[:-valid_length]   
    
    return train_X, train_label, valid_X, valid_label, test_X, test_label,  smax[0], smin[0]

train_X, train_label, valid_X, valid_label, test_X, test_label,  smax, smin = get_dataset() 

#打散
train_Xm = np.zeros(shape=(len(train_X), window_size,feature))
train_labelm = np.zeros(shape=(len(train_label), pre_size,1))
indices = np.random.permutation(train_Xm.shape[0])
train_Xm = train_X[indices]
train_labelm = train_label[indices] 

wr = np.zeros(shape=(len(test_label.reshape(-1)),12))

print("--------------------------0.0005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.0005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,0] = pred_test

print("--------------------------0.001-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.001, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,1] = pred_test

print("--------------------------0.005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,2] = pred_test

print("---------------------------0.01----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.01, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,3] = pred_test

print("-----------------------------0.015----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.015, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)   
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,4] = pred_test

print("---------------------------------0.02-------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.02, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,5] = pred_test

print("------------------------------------0.025---------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.025, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)    
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,6] = pred_test

print("--------------------------------------0.03--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.03, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,7] = pred_test

print("--------------------------------------0.035--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.035, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,8] = pred_test

print("--------------------------------------0.04--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.04, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,9] = pred_test

def mae_loss(y_true, y_pred):       
    return K.mean(tf.abs(y_true - y_pred))
print("--------------------------mae-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mae_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,10] = pred_test

def mse_loss(y_true, y_pred):       
    return K.mean(tf.square(y_true - y_pred))
print("--------------------------mse-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mse_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,11] = pred_test

pred_value=pd.DataFrame(data=(np.array(wr)))
pred_value.to_csv("./ex/cnna1__51_10.csv",encoding='utf-8') 


print("---------------------k=0.5,p=1/16--------------------------")
def get_dataset():
    df = pd.read_csv('./data1.csv',usecols=["Z21","T","month","day","hour"])  
    data = np.array(df)
#   数据集总长度
    wi = np.ones(shape=(data.shape[0],feature))
    # 训练总长度
    ab_len=data[:-test_length-window_size,:].shape[0]
    indices = pd.read_csv('./indices52.csv',usecols=["9"])  
    indices =np.array(indices).reshape(-1).astype(int)
#   攻击 k p
    # k随机选10% 20% 30%
    k = 0.5
    p=1/16
    # 测试集用的前168个数据不能受到攻击
    for i in range(int(data[:(-test_length-window_size),:].shape[0]*k)):
        wi[indices[i],0]= 1+p
    data=data*wi   # 受到攻击        
        
    train_data=data[:-test_length,:]
    smax=train_data.max(axis=0)
    smin=train_data.min(axis=0)
    open_arr=((data-smin)/(smax-smin))
    X = np.zeros(shape=(len(open_arr) - window_size-pre_size+1, window_size,feature))
    label = np.zeros(shape=(len(open_arr) - window_size-pre_size+1,pre_size))
    for i in range(len(open_arr) - window_size-pre_size+1):
        X[i, :] = open_arr[i:i+window_size,:]
        label[i] = open_arr[i+window_size:i+window_size+pre_size,0]
    train_X = X[:(-test_length), :]
    train_label = label[:(-test_length)]

    test_X = X[-test_length:, :]
    test_label = label[-test_length:]
    test_X = test_X[pre_size-1:, :]
    test_label = test_label[pre_size-1:] 
   
    valid_length=int(train_X.shape[0]*0.2)
    valid_X = train_X[-valid_length:, :]
    valid_label = train_label[-valid_length:]
    valid_X = valid_X[pre_size-1:, :]
    valid_label = valid_label[pre_size-1:]
    
    train_X = train_X[:-valid_length, :]
    train_label = train_label[:-valid_length]   
    
    return train_X, train_label, valid_X, valid_label, test_X, test_label,  smax[0], smin[0]

train_X, train_label, valid_X, valid_label, test_X, test_label,  smax, smin = get_dataset() 

#打散
train_Xm = np.zeros(shape=(len(train_X), window_size,feature))
train_labelm = np.zeros(shape=(len(train_label), pre_size,1))
indices = np.random.permutation(train_Xm.shape[0])
train_Xm = train_X[indices]
train_labelm = train_label[indices] 

wr = np.zeros(shape=(len(test_label.reshape(-1)),12))

print("--------------------------0.0005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.0005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,0] = pred_test

print("--------------------------0.001-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.001, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,1] = pred_test

print("--------------------------0.005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,2] = pred_test

print("---------------------------0.01----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.01, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,3] = pred_test

print("-----------------------------0.015----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.015, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)   
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,4] = pred_test

print("---------------------------------0.02-------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.02, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,5] = pred_test

print("------------------------------------0.025---------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.025, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)    
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,6] = pred_test

print("--------------------------------------0.03--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.03, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,7] = pred_test

print("--------------------------------------0.035--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.035, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,8] = pred_test

print("--------------------------------------0.04--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.04, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,9] = pred_test

def mae_loss(y_true, y_pred):       
    return K.mean(tf.abs(y_true - y_pred))
print("--------------------------mae-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mae_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,10] = pred_test

def mse_loss(y_true, y_pred):       
    return K.mean(tf.square(y_true - y_pred))
print("--------------------------mse-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mse_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,11] = pred_test

pred_value=pd.DataFrame(data=(np.array(wr)))
pred_value.to_csv("./ex/cnna1__52_10.csv",encoding='utf-8') 


print("---------------------k=0.5,p=1/8--------------------------")
def get_dataset():
    df = pd.read_csv('./data1.csv',usecols=["Z21","T","month","day","hour"])  
    data = np.array(df)
#   数据集总长度
    wi = np.ones(shape=(data.shape[0],feature))
    # 训练总长度
    ab_len=data[:-test_length-window_size,:].shape[0]
    indices = pd.read_csv('./indices53.csv',usecols=["9"])  
    indices =np.array(indices).reshape(-1).astype(int)
#   攻击 k p
    # k随机选10% 20% 30%
    k = 0.5
    p=1/8
    # 测试集用的前168个数据不能受到攻击
    for i in range(int(data[:(-test_length-window_size),:].shape[0]*k)):
        wi[indices[i],0]= 1+p
    data=data*wi   # 受到攻击        
        
    train_data=data[:-test_length,:]
    smax=train_data.max(axis=0)
    smin=train_data.min(axis=0)
    open_arr=((data-smin)/(smax-smin))
    X = np.zeros(shape=(len(open_arr) - window_size-pre_size+1, window_size,feature))
    label = np.zeros(shape=(len(open_arr) - window_size-pre_size+1,pre_size))
    for i in range(len(open_arr) - window_size-pre_size+1):
        X[i, :] = open_arr[i:i+window_size,:]
        label[i] = open_arr[i+window_size:i+window_size+pre_size,0]
    train_X = X[:(-test_length), :]
    train_label = label[:(-test_length)]

    test_X = X[-test_length:, :]
    test_label = label[-test_length:]
    test_X = test_X[pre_size-1:, :]
    test_label = test_label[pre_size-1:] 
   
    valid_length=int(train_X.shape[0]*0.2)
    valid_X = train_X[-valid_length:, :]
    valid_label = train_label[-valid_length:]
    valid_X = valid_X[pre_size-1:, :]
    valid_label = valid_label[pre_size-1:]
    
    train_X = train_X[:-valid_length, :]
    train_label = train_label[:-valid_length]   
    
    return train_X, train_label, valid_X, valid_label, test_X, test_label,  smax[0], smin[0]

train_X, train_label, valid_X, valid_label, test_X, test_label,  smax, smin = get_dataset() 

#打散
train_Xm = np.zeros(shape=(len(train_X), window_size,feature))
train_labelm = np.zeros(shape=(len(train_label), pre_size,1))
indices = np.random.permutation(train_Xm.shape[0])
train_Xm = train_X[indices]
train_labelm = train_label[indices] 

wr = np.zeros(shape=(len(test_label.reshape(-1)),12))

print("--------------------------0.0005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.0005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,0] = pred_test

print("--------------------------0.001-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.001, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,1] = pred_test

print("--------------------------0.005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,2] = pred_test

print("---------------------------0.01----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.01, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,3] = pred_test

print("-----------------------------0.015----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.015, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)   
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,4] = pred_test

print("---------------------------------0.02-------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.02, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,5] = pred_test

print("------------------------------------0.025---------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.025, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)    
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,6] = pred_test

print("--------------------------------------0.03--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.03, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,7] = pred_test

print("--------------------------------------0.035--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.035, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,8] = pred_test

print("--------------------------------------0.04--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.04, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,9] = pred_test

def mae_loss(y_true, y_pred):       
    return K.mean(tf.abs(y_true - y_pred))
print("--------------------------mae-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mae_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,10] = pred_test

def mse_loss(y_true, y_pred):       
    return K.mean(tf.square(y_true - y_pred))
print("--------------------------mse-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mse_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,11] = pred_test

pred_value=pd.DataFrame(data=(np.array(wr)))
pred_value.to_csv("./ex/cnna1__53_10.csv",encoding='utf-8') 


print("---------------------k=0.5,p=1/4--------------------------")
def get_dataset():
    df = pd.read_csv('./data1.csv',usecols=["Z21","T","month","day","hour"])  
    data = np.array(df)
#   数据集总长度
    wi = np.ones(shape=(data.shape[0],feature))
    # 训练总长度
    ab_len=data[:-test_length-window_size,:].shape[0]
    indices = pd.read_csv('./indices54.csv',usecols=["9"])  
    indices =np.array(indices).reshape(-1).astype(int)
#   攻击 k p
    # k随机选10% 20% 30%
    k = 0.5
    p=1/4
    # 测试集用的前168个数据不能受到攻击
    for i in range(int(data[:(-test_length-window_size),:].shape[0]*k)):
        wi[indices[i],0]= 1+p
    data=data*wi   # 受到攻击        
        
    train_data=data[:-test_length,:]
    smax=train_data.max(axis=0)
    smin=train_data.min(axis=0)
    open_arr=((data-smin)/(smax-smin))
    X = np.zeros(shape=(len(open_arr) - window_size-pre_size+1, window_size,feature))
    label = np.zeros(shape=(len(open_arr) - window_size-pre_size+1,pre_size))
    for i in range(len(open_arr) - window_size-pre_size+1):
        X[i, :] = open_arr[i:i+window_size,:]
        label[i] = open_arr[i+window_size:i+window_size+pre_size,0]
    train_X = X[:(-test_length), :]
    train_label = label[:(-test_length)]

    test_X = X[-test_length:, :]
    test_label = label[-test_length:]
    test_X = test_X[pre_size-1:, :]
    test_label = test_label[pre_size-1:] 
   
    valid_length=int(train_X.shape[0]*0.2)
    valid_X = train_X[-valid_length:, :]
    valid_label = train_label[-valid_length:]
    valid_X = valid_X[pre_size-1:, :]
    valid_label = valid_label[pre_size-1:]
    
    train_X = train_X[:-valid_length, :]
    train_label = train_label[:-valid_length]   
    
    return train_X, train_label, valid_X, valid_label, test_X, test_label,  smax[0], smin[0]

train_X, train_label, valid_X, valid_label, test_X, test_label,  smax, smin = get_dataset() 

#打散
train_Xm = np.zeros(shape=(len(train_X), window_size,feature))
train_labelm = np.zeros(shape=(len(train_label), pre_size,1))
indices = np.random.permutation(train_Xm.shape[0])
train_Xm = train_X[indices]
train_labelm = train_label[indices] 

wr = np.zeros(shape=(len(test_label.reshape(-1)),12))

print("--------------------------0.0005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.0005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,0] = pred_test

print("--------------------------0.001-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.001, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,1] = pred_test

print("--------------------------0.005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,2] = pred_test

print("---------------------------0.01----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.01, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,3] = pred_test

print("-----------------------------0.015----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.015, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)   
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,4] = pred_test

print("---------------------------------0.02-------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.02, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,5] = pred_test

print("------------------------------------0.025---------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.025, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)    
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,6] = pred_test

print("--------------------------------------0.03--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.03, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,7] = pred_test

print("--------------------------------------0.035--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.035, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,8] = pred_test

print("--------------------------------------0.04--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.04, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,9] = pred_test

def mae_loss(y_true, y_pred):       
    return K.mean(tf.abs(y_true - y_pred))
print("--------------------------mae-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mae_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,10] = pred_test

def mse_loss(y_true, y_pred):       
    return K.mean(tf.square(y_true - y_pred))
print("--------------------------mse-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mse_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,11] = pred_test

pred_value=pd.DataFrame(data=(np.array(wr)))
pred_value.to_csv("./ex/cnna1__54_10.csv",encoding='utf-8') 



print("---------------------k=0.5,p=1/2--------------------------")
def get_dataset():
    df = pd.read_csv('./data1.csv',usecols=["Z21","T","month","day","hour"])  
    data = np.array(df)
#   数据集总长度
    wi = np.ones(shape=(data.shape[0],feature))
    # 训练总长度
    ab_len=data[:-test_length-window_size,:].shape[0]
    indices = pd.read_csv('./indices55.csv',usecols=["9"])  
    indices =np.array(indices).reshape(-1).astype(int)
#   攻击 k p
    # k随机选10% 20% 30%
    k = 0.5
    p=1/2
    # 测试集用的前168个数据不能受到攻击
    for i in range(int(data[:(-test_length-window_size),:].shape[0]*k)):
        wi[indices[i],0]= 1+p
    data=data*wi   # 受到攻击        
        
    train_data=data[:-test_length,:]
    smax=train_data.max(axis=0)
    smin=train_data.min(axis=0)
    open_arr=((data-smin)/(smax-smin))
    X = np.zeros(shape=(len(open_arr) - window_size-pre_size+1, window_size,feature))
    label = np.zeros(shape=(len(open_arr) - window_size-pre_size+1,pre_size))
    for i in range(len(open_arr) - window_size-pre_size+1):
        X[i, :] = open_arr[i:i+window_size,:]
        label[i] = open_arr[i+window_size:i+window_size+pre_size,0]
    train_X = X[:(-test_length), :]
    train_label = label[:(-test_length)]

    test_X = X[-test_length:, :]
    test_label = label[-test_length:]
    test_X = test_X[pre_size-1:, :]
    test_label = test_label[pre_size-1:] 
   
    valid_length=int(train_X.shape[0]*0.2)
    valid_X = train_X[-valid_length:, :]
    valid_label = train_label[-valid_length:]
    valid_X = valid_X[pre_size-1:, :]
    valid_label = valid_label[pre_size-1:]
    
    train_X = train_X[:-valid_length, :]
    train_label = train_label[:-valid_length]   
    
    return train_X, train_label, valid_X, valid_label, test_X, test_label,  smax[0], smin[0]

train_X, train_label, valid_X, valid_label, test_X, test_label,  smax, smin = get_dataset() 

#打散
train_Xm = np.zeros(shape=(len(train_X), window_size,feature))
train_labelm = np.zeros(shape=(len(train_label), pre_size,1))
indices = np.random.permutation(train_Xm.shape[0])
train_Xm = train_X[indices]
train_labelm = train_label[indices] 

wr = np.zeros(shape=(len(test_label.reshape(-1)),12))

print("--------------------------0.0005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.0005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,0] = pred_test

print("--------------------------0.001-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.001, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,1] = pred_test

print("--------------------------0.005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,2] = pred_test

print("---------------------------0.01----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.01, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,3] = pred_test

print("-----------------------------0.015----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.015, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)   
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,4] = pred_test

print("---------------------------------0.02-------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.02, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,5] = pred_test

print("------------------------------------0.025---------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.025, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)    
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,6] = pred_test

print("--------------------------------------0.03--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.03, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,7] = pred_test

print("--------------------------------------0.035--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.035, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,8] = pred_test

print("--------------------------------------0.04--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.04, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,9] = pred_test

def mae_loss(y_true, y_pred):       
    return K.mean(tf.abs(y_true - y_pred))
print("--------------------------mae-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mae_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,10] = pred_test

def mse_loss(y_true, y_pred):       
    return K.mean(tf.square(y_true - y_pred))
print("--------------------------mse-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mse_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,11] = pred_test

pred_value=pd.DataFrame(data=(np.array(wr)))
pred_value.to_csv("./ex/cnna1__55_10.csv",encoding='utf-8') 


print("---------------------k=0.5,p=1--------------------------")
def get_dataset():
    df = pd.read_csv('./data1.csv',usecols=["Z21","T","month","day","hour"])  
    data = np.array(df)
#   数据集总长度
    wi = np.ones(shape=(data.shape[0],feature))
    # 训练总长度
    ab_len=data[:-test_length-window_size,:].shape[0]
    indices = pd.read_csv('./indices56.csv',usecols=["9"])  
    indices =np.array(indices).reshape(-1).astype(int)
#   攻击 k p
    # k随机选10% 20% 30%
    k = 0.5
    p=1
    # 测试集用的前168个数据不能受到攻击
    for i in range(int(data[:(-test_length-window_size),:].shape[0]*k)):
        wi[indices[i],0]= 1+p
    data=data*wi   # 受到攻击        
        
    train_data=data[:-test_length,:]
    smax=train_data.max(axis=0)
    smin=train_data.min(axis=0)
    open_arr=((data-smin)/(smax-smin))
    X = np.zeros(shape=(len(open_arr) - window_size-pre_size+1, window_size,feature))
    label = np.zeros(shape=(len(open_arr) - window_size-pre_size+1,pre_size))
    for i in range(len(open_arr) - window_size-pre_size+1):
        X[i, :] = open_arr[i:i+window_size,:]
        label[i] = open_arr[i+window_size:i+window_size+pre_size,0]
    train_X = X[:(-test_length), :]
    train_label = label[:(-test_length)]

    test_X = X[-test_length:, :]
    test_label = label[-test_length:]
    test_X = test_X[pre_size-1:, :]
    test_label = test_label[pre_size-1:] 
   
    valid_length=int(train_X.shape[0]*0.2)
    valid_X = train_X[-valid_length:, :]
    valid_label = train_label[-valid_length:]
    valid_X = valid_X[pre_size-1:, :]
    valid_label = valid_label[pre_size-1:]
    
    train_X = train_X[:-valid_length, :]
    train_label = train_label[:-valid_length]   
    
    return train_X, train_label, valid_X, valid_label, test_X, test_label,  smax[0], smin[0]

train_X, train_label, valid_X, valid_label, test_X, test_label,  smax, smin = get_dataset() 

#打散
train_Xm = np.zeros(shape=(len(train_X), window_size,feature))
train_labelm = np.zeros(shape=(len(train_label), pre_size,1))
indices = np.random.permutation(train_Xm.shape[0])
train_Xm = train_X[indices]
train_labelm = train_label[indices] 

wr = np.zeros(shape=(len(test_label.reshape(-1)),12))

print("--------------------------0.0005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.0005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,0] = pred_test

print("--------------------------0.001-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.001, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,1] = pred_test

print("--------------------------0.005-------------------------------")    
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.005, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,2] = pred_test

print("---------------------------0.01----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.01, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,3] = pred_test

print("-----------------------------0.015----------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.015, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)   
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,4] = pred_test

print("---------------------------------0.02-------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.02, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,5] = pred_test

print("------------------------------------0.025---------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.025, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)   
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)    
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,6] = pred_test

print("--------------------------------------0.03--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.03, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False) 
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,7] = pred_test

print("--------------------------------------0.035--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.035, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)  
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,8] = pred_test

print("--------------------------------------0.04--------------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(loss=lambda y_t, y_p: huber_loss(0.04, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)  
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
pred_test = (model.predict(test_X)*(smax-smin)+smin).reshape(-1)
test_mape = mean_absolute_percentage_error(pre_true,pred_test)
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("test_mape:",test_mape,"valid_mape:",valid_mape)
wr[:,9] = pred_test

def mae_loss(y_true, y_pred):       
    return K.mean(tf.abs(y_true - y_pred))
print("--------------------------mae-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mae_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,10] = pred_test

def mse_loss(y_true, y_pred):       
    return K.mean(tf.square(y_true - y_pred))
print("--------------------------mse-------------------------------")
model = build_model(units1,units2,Dropout_rate,kernel_size)
model.compile(optimizer=tf.keras.optimizers.Adam(Learning_rate), loss=mse_loss)
history=model.fit(train_Xm,train_labelm,epochs=epochs,batch_size=batch_size,verbose=0,shuffle=False)
prediction = model.predict(test_X)
scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
pre_true = (test_label*(smax-smin)+smin).reshape(-1)
scaled_prediction=pd.DataFrame(data=(np.array(scaled_prediction).T))
print("mean_squared_error:", mean_squared_error(pre_true,scaled_prediction))
print("rmse:", sqrt(mean_squared_error(pre_true,scaled_prediction)))
print("mean_absolute_error:", mean_absolute_error(pre_true,scaled_prediction))
print("mape:", mean_absolute_percentage_error(pre_true,scaled_prediction))
print("r2 score:", r2_score(pre_true,scaled_prediction)) 
valid_mape = mean_absolute_percentage_error(valid_label.reshape(-1),model.predict(valid_X).reshape(-1))
print("valid_mape:", valid_mape)
wr[:,11] = pred_test

pred_value=pd.DataFrame(data=(np.array(wr)))
pred_value.to_csv("./ex/cnna1__56_10.csv",encoding='utf-8') 


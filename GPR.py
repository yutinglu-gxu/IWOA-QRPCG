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
import time


pre_size = 24
test_length = 336
valid_length = 5136 # From May
feature = 1
window_size = 168

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
    
    train_X=train_X.reshape(-1,window_size)
    test_X=test_X.reshape(-1,window_size)
    valid_X=valid_X.reshape(-1,window_size)
    train_label=train_label.reshape(-1,pre_size)
    valid_label=valid_label.reshape(-1,pre_size)
    test_label=test_label.reshape(-1,pre_size)
    
    return train_X, train_label, valid_X, valid_label, test_X, test_label,  smax, smin

train_X, train_label, valid_X, valid_label, test_X, test_label,  smax, smin = get_dataset(window_size)



from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Exponentiation,RationalQuadratic,RBF,ConstantKernel,DotProduct,WhiteKernel,Matern,PairwiseKernel
from scipy.special import erfinv
import keras.backend as K

kernels=[
           Exponentiation(RationalQuadratic(), exponent=2),
           RBF() + ConstantKernel(constant_value=2),
           DotProduct() + WhiteKernel(noise_level=0.5),
           1.0 * RBF(1.0),
           1.0 * Matern(length_scale=1.0, nu=1.5),
           RationalQuadratic(length_scale=1.0, alpha=1.5),
           PairwiseKernel(metric='rbf')
        ]


df = pd.read_csv('../true.csv')
data=df['0'].values.reshape(-1, 1)

for kernel in kernels:
    gpr = GaussianProcessRegressor(kernel=kernel).fit(train_X, train_label)
    
    print(kernel)  
    print("------------------------")
    quantile = 0.025  
    print("quantile",quantile)
    # Validation set
    y_pred, sigma = gpr.predict(valid_X, return_std=True)
    quantile_values = []
    for i in range(sigma.shape[0]):
        y_mean=y_pred[i,:]
        y_std = sigma[i,:]
        # CDF
        quantile_values.append(y_mean + np.sqrt(2) * y_std * erfinv(2 * quantile - 1))   
     
    qv=np.array(quantile_values).reshape(-1, 1)
    vl=valid_label.reshape(-1, 1)
    score=np.array(K.mean(K.maximum(quantile*(vl-qv),(quantile-1)*(vl-qv))))
    print(score)  
    model=gpr
    
    # Test set
    y_pred, sigma = model.predict(test_X, return_std=True)
    quantile_values = []
    for i in range(sigma.shape[0]):
        y_mean=y_pred[i,:]
        y_std = sigma[i,:]
        quantile_values.append(y_mean + np.sqrt(2) * y_std * erfinv(2 * quantile - 1))      
    
    quantile_values=np.array(quantile_values) *(smax-smin)+smin       
    qv=quantile_values.reshape(-1, 1)
    print(np.array(K.mean(K.maximum(quantile*(data-qv),(quantile-1)*(data-qv)))))
    pred_value=pd.DataFrame(data=(np.array(quantile_values.reshape(-1))))
    pred_value.to_csv("./ISO0025.csv",encoding='utf-8')
    
    print("------------------------")
    quantile = 0.05  
    print("quantile",quantile)
    # Validation set
    y_pred, sigma = gpr.predict(valid_X, return_std=True)
    quantile_values = []
    for i in range(sigma.shape[0]):
        y_mean=y_pred[i,:]
        y_std = sigma[i,:]
        quantile_values.append(y_mean + np.sqrt(2) * y_std * erfinv(2 * quantile - 1))   
     
    qv=np.array(quantile_values).reshape(-1, 1)
    vl=valid_label.reshape(-1, 1)
    score=np.array(K.mean(K.maximum(quantile*(vl-qv),(quantile-1)*(vl-qv))))
    print(score)  
    model=gpr

    # Test set
    y_pred, sigma = model.predict(test_X, return_std=True)
    quantile_values = []
    for i in range(sigma.shape[0]):
        y_mean=y_pred[i,:]
        y_std = sigma[i,:]
        quantile_values.append(y_mean + np.sqrt(2) * y_std * erfinv(2 * quantile - 1))      
    
    quantile_values=np.array(quantile_values) *(smax-smin)+smin       
    qv=quantile_values.reshape(-1, 1)
    print(np.array(K.mean(K.maximum(quantile*(data-qv),(quantile-1)*(data-qv)))))
    pred_value=pd.DataFrame(data=(np.array(quantile_values.reshape(-1))))
    pred_value.to_csv("./ISO005.csv",encoding='utf-8')
    
    print("------------------------")
    quantile = 0.95  
    print("quantile",quantile)
    # Validation set
    y_pred, sigma = gpr.predict(valid_X, return_std=True)
    quantile_values = []
    for i in range(sigma.shape[0]):
        y_mean=y_pred[i,:]
        y_std = sigma[i,:]
        quantile_values.append(y_mean + np.sqrt(2) * y_std * erfinv(2 * quantile - 1))   
     
    qv=np.array(quantile_values).reshape(-1, 1)
    vl=valid_label.reshape(-1, 1)
    score=np.array(K.mean(K.maximum(quantile*(vl-qv),(quantile-1)*(vl-qv))))
    print(score)  
    model=gpr
    
    # Test set
    y_pred, sigma = model.predict(test_X, return_std=True)
    quantile_values = []
    for i in range(sigma.shape[0]):
        y_mean=y_pred[i,:]
        y_std = sigma[i,:]
        quantile_values.append(y_mean + np.sqrt(2) * y_std * erfinv(2 * quantile - 1))      
    
    quantile_values=np.array(quantile_values) *(smax-smin)+smin       
    qv=quantile_values.reshape(-1, 1)
    print(np.array(K.mean(K.maximum(quantile*(data-qv),(quantile-1)*(data-qv)))))
    pred_value=pd.DataFrame(data=(np.array(quantile_values.reshape(-1))))
    pred_value.to_csv("./ISO095.csv",encoding='utf-8')    
    
    print("------------------------")
    quantile = 0.975  
    print("quantile",quantile)
    # Validation set
    y_pred, sigma = gpr.predict(valid_X, return_std=True)
    quantile_values = []
    for i in range(sigma.shape[0]):
        y_mean=y_pred[i,:]
        y_std = sigma[i,:]
        quantile_values.append(y_mean + np.sqrt(2) * y_std * erfinv(2 * quantile - 1))   
     
    qv=np.array(quantile_values).reshape(-1, 1)
    vl=valid_label.reshape(-1, 1)
    score=np.array(K.mean(K.maximum(quantile*(vl-qv),(quantile-1)*(vl-qv))))
    print(score)  
    model=gpr
    
    # Test set
    y_pred, sigma = model.predict(test_X, return_std=True)
    quantile_values = []
    for i in range(sigma.shape[0]):
        y_mean=y_pred[i,:]
        y_std = sigma[i,:]
        quantile_values.append(y_mean + np.sqrt(2) * y_std * erfinv(2 * quantile - 1))      
    
    quantile_values=np.array(quantile_values) *(smax-smin)+smin       
    qv=quantile_values.reshape(-1, 1)
    print(np.array(K.mean(K.maximum(quantile*(data-qv),(quantile-1)*(data-qv)))))
    pred_value=pd.DataFrame(data=(np.array(quantile_values.reshape(-1))))
    pred_value.to_csv("./ISO0975.csv",encoding='utf-8')
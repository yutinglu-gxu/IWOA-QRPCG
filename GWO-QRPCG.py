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
from keras.layers import Dense,Dropout,Flatten,Input,concatenate,add,Activation,Bidirectional
from keras.layers.convolutional import Conv1D,MaxPooling1D
import time
import keras.backend as K
from tensorflow.compat.v1.keras.layers import CuDNNGRU
from keras.callbacks import EarlyStopping


pre_size = 24
test_length = 336
valid_length = 5136 # From May
feature = 1
epochs = 200 
window_size = 0   
window_size2 = 0    
filters1 = 0      
filters2 = 0        
kernel_size = 0   
pool_size = 0     
strides = 0        
GRU_units1 = 0     
GRU_units2 = 0    
Dense_units1 = 0   
Dense_units2 = 0   
Dropout_rate = 0   
batch_size = 0  
Learning_rate = 0  

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

def build_model(Pos):
    window_size = 168*(int(Pos[0]))       # 168 336--168*x,1 2
    window_size2 = 24*(int(Pos[1]))       # 24 48 72--24*x,1 2 3
    filters1 = 2**(int(Pos[2])+3)         # 16 32 64 128--2**(x+3),1,2,3,4
    filters2 = 2**(int(Pos[3])+3)         # 16 32 64 128--2**(x+3),1,2,3,4
    kernel_size = (int(Pos[4]))+1         # 2 3 4 5 6 7 8 9 --x+1,1,2,3,4,5,6,7,8
    pool_size = int(Pos[5]+1)             # 2 3 4 5 6--x+1,1,2,3,4,5
    strides = int(Pos[6]+1)               # 2 3 4--x+1,1,2,3且小于pool_size
    if strides>pool_size:
        strides = pool_size+0
    GRU_units1 = 2**(int(Pos[7])+3)       # 16 32 64 128--2**(x+3),x=1,2,3,4
    GRU_units2 = 2**(int(Pos[8])+3)       # 16 32 64 128--2**(x+3),x=1,2,3,4
    Dense_units1 = 2**(int(Pos[9])+3)     # 16 32 64 128--2**(x+3),x=1,2,3,4
    Dense_units2 = 2**(int(Pos[10])+3)    # 16 32 64 128--2**(x+3),x=1,2,3,4
    Dropout_rate = 0.1*(int(Pos[11])+0)   # 0.1 0.2 0.3 0.4 0.5--0.1*x,x=1,2,3,4,5
    batch_size = 2**(int(Pos[12])+4)      # 32 64 128--2**(x+4),x=1,2,3
    Learning_rate = 0.00005*math.pow(2,int((int(Pos[13])+1)/2))*math.pow(5,int(int(Pos[13])/2))# 0.0001,0.0005,0.001,0.005    
    
    print(window_size,window_size2,filters1,filters2,kernel_size,pool_size,strides,GRU_units1,GRU_units2,Dense_units1,Dense_units2,Dropout_rate,batch_size,Learning_rate)
    
    train_X, train_label, valid_X, valid_label, test_X, test_label,  smax, smin = get_dataset(window_size)
    train_X2, train_label2, valid_X2, valid_label2,test_X2, test_label2, smax2, smin2 = get_dataset(window_size2)
    train_X2=train_X2[(window_size-window_size2):,]
    train_label2=train_label2[(window_size-window_size2):,]

    # shuffle
    train_Xm = np.zeros(shape=(len(train_X), window_size,feature))
    train_labelm = np.zeros(shape=(len(train_label), pre_size,1))
    train_Xm=train_X 
    train_labelm=train_label
    indices = np.random.permutation(train_Xm.shape[0])
    train_Xm = train_Xm[indices]
    train_labelm = train_labelm[indices]   

    train_Xm2 = np.zeros(shape=(len(train_X2), window_size2,feature))
    train_Xm2 = train_X2
    train_Xm2 = train_Xm2[indices] 
    
    inp = Input(shape=(window_size, feature))
    conv = Conv1D(filters1, kernel_size,padding='same',activation='relu')(inp)
    conv = Conv1D(filters2, kernel_size,padding='same',activation='relu')(conv)
    conv = MaxPooling1D(pool_size=pool_size,strides=strides,padding='same')(conv)
    conv = Dropout(Dropout_rate)(conv)
    conv = Flatten()(conv)

    inp2 = Input(shape=(window_size2, feature))
    lstm = Bidirectional(CuDNNGRU(GRU_units1, return_sequences=True))(inp2)
    lstm = Bidirectional(CuDNNGRU(GRU_units2, return_sequences=True))(lstm)
    lstm = Dropout(Dropout_rate)(lstm)
    lstm = Flatten()(lstm)

    conc = concatenate([conv, lstm])
    conc = Dense(Dense_units1,activation='relu')(conc)
    conc = Dense(Dense_units2,activation='relu')(conc)
    outp = Dense(24)(conc)
    model = Model(inputs=[inp,inp2], outputs=outp)

    quantile=0.05  # 0.05、0.95、0.025、0.975
    model.compile(loss=lambda y_t, y_p: quantile_loss(quantile, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
    
    earlyStop =[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30,restore_best_weights=True,verbose=0) ] 
    history=model.fit([train_Xm,train_Xm2],train_labelm,callbacks=[earlyStop],epochs=epochs,batch_size=batch_size,validation_data=([valid_X,valid_X2], valid_label),verbose=0,shuffle=False)
    
    prediction = model.predict([test_X,test_X2])
    scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
    result=model.evaluate([valid_X,valid_X2], valid_label)
    
    return result,scaled_prediction,model

model_filename='./ex/models_GWO005.h5'
def GWO(objf,lb,ub,dim,SearchAgents_no,Max_iter):
    Positions = np.zeros([SearchAgents_no, dim])
    Alpha_pos = np.zeros([1, dim])
    Alpha_score = float("inf")
    Beta_pos = np.zeros([1, dim])
    Beta_score = float("inf")
    Delta_pos = np.zeros([1, dim])
    Delta_score = float("inf")
    
    fitness = np.zeros(SearchAgents_no)

    t=0
    convergence_curve=np.zeros(Max_iter)
    
    Positions=np.random.uniform(0,1,(SearchAgents_no,dim))*(ub-lb)+lb
            
    startTime1 = time.time()    
    while t < Max_iter:
        print('The number of iterations:',t+1)
        startTime0 = time.time()
    
        for i in range(0,SearchAgents_no):
            Positions[i,:]=np.clip(Positions[i,:],lb,ub)
            fitness[i],prediction,model=objf(Positions[i,:])
            if fitness[i]<Alpha_score:
                Alpha_score=fitness[i]+0
                Alpha_pos=Positions[i,:]+0
                print('-------------------------Fit_GWO----------------------------')
                print('The number of iterations:',t+1)
                print('Alpha_pos:',Alpha_pos)
                print('prediction:',prediction)
                pred_list.append(prediction)
                model.save_weights(model_filename)
                
            if (fitness[i]>Alpha_score and fitness[i]<Beta_score): 
                Beta_score=fitness[i]+0
                Beta_pos=Positions[i,:]+0
                
            if (fitness[i]>Alpha_score and fitness[i]>Beta_score and fitness[i]<Delta_score): 
                Delta_score=fitness[i]+0
                Delta_pos=Positions[i,:]+0             
                
        a=2-t*((2)/Max_iter) #a~[2,0]    
        
        for i in range(0,SearchAgents_no):                    
            for j in range(0,dim):   
                r1=random.random() # r1 is a random number in [0,1]
                r2=random.random() # r2 is a random number in [0,1]
                A1=2*a*r1-a   # Equation (3.3)
                C1=2*r2       # Equation (3.4)          
                D_alpha=abs(C1*Alpha_pos[j]-Positions[i,j]); # Equation (3.5)-part 1
                X1=Alpha_pos[j]-A1*D_alpha;    # Equation (3.6)-part 1
                
                r1=random.random()
                r2=random.random()
                A2=2*a*r1-a
                C2=2*r2
                D_beta=abs(C2*Beta_pos[j]-Positions[i,j]); # Equation (3.5)-part 2
                X2=Beta_pos[j]-A2*D_beta;  # Equation (3.6)-part 2
            
                r1=random.random()
                r2=random.random()
                A3=2*a*r1-a
                C3=2*r2
                D_delta=abs(C3*Delta_pos[j]-Positions[i,j]); # Equation (3.5)-part 3
                X3=Delta_pos[j]-A3*D_delta; # Equation (3.5)-part 3             
            
                Positions[i,j]=(X1+X2+X3)/3;# Equation (3.7)
                                
        convergence_curve[t]=Alpha_score
        t0 = time.time() - startTime0
        print('End iteration:',t+1,'Iteration time:', t0)
        t=t+1
    t1 = time.time() - startTime1
    print('Total time of GWO:', t1)
    return Alpha_score,Alpha_pos,convergence_curve,pred_list


if __name__ == '__main__':
    dim = 14
    lb=np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1])
    ub=np.array([2.99,3.99,4.99,4.99,8.99,5.99,3.99,4.99,4.99,4.99,4.99,5.99,3.99,4.99])
    SearchAgents_no=15
    Max_iter=10
    print('-------------GWO005--------------')
    # GWO( )  
    pred_value=pd.DataFrame(data=(np.array(pred_list).T))
    pred_value.to_csv("./ex/GWO005.csv",encoding='utf-8')
    
    print(Leader_score)
    print(Leader_pos)     
    print(convergence_curve)


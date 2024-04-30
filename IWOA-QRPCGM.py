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
feature = 5
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
    df = pd.read_csv('../ISO_2016_2017.csv',usecols=["RT_Demand","DA_Demand","RT_LMP","Dry_Bulb","Dew_Point"])   
    data=np.array(df)
    train_data=data[:-test_length,:]
    smax=train_data.max(axis=0)
    smin=train_data.min(axis=0)
    open_arr=((data-smin)/(smax-smin))
    X = np.zeros(shape=(len(open_arr) - window_size-pre_size+1, window_size,feature))
    label = np.zeros(shape=(len(open_arr) - window_size-pre_size+1,pre_size))
    test_X = np.zeros(shape=(window_size,pre_size))
    for i in range(len(open_arr) - window_size-pre_size+1):
        X[i, :] = open_arr[i:i+window_size,:]
        label[i] = open_arr[i+window_size:i+window_size+pre_size,0]
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
    
    train_label=train_label.reshape(-1,pre_size,1)
    valid_label=valid_label.reshape(-1,pre_size,1)
    test_label=test_label.reshape(-1,pre_size,1)
    
    return train_X, train_label, valid_X, valid_label, test_X, test_label,  smax[0], smin[0]


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
    filters1 = 2**(int(Pos[2])+3)              # 16 32 64 128--2**(x+3),1,2,3,4
    filters2 = 2**(int(Pos[3])+3)              # 16 32 64 128--2**(x+3),1,2,3,4
    kernel_size = (int(Pos[4]))+1            # 2 3 4 5 6 7 8 9 --x+1,1,2,3,4,5,6,7,8
    pool_size = int(Pos[5]+1)                 # 2 3 4 5 6--x+1,1,2,3,4,5
    strides = int(Pos[6]+1)                     # 2 3 4--x+1,1,2,3 and <=pool_size
    if strides>pool_size:
        strides = pool_size+0
    GRU_units1 = 2**(int(Pos[7])+3)       # 16 32 64 128--2**(x+3),x=1,2,3,4
    GRU_units2 = 2**(int(Pos[8])+3)       # 16 32 64 128--2**(x+3),x=1,2,3,4
    Dense_units1 = 2**(int(Pos[9])+3)     # 16 32 64 128--2**(x+3),x=1,2,3,4
    Dense_units2 = 2**(int(Pos[10])+3)    # 16 32 64 128--2**(x+3),x=1,2,3,4
    Dropout_rate = 0.1*(int(Pos[11])+0)   # 0.1 0.2 0.3 0.4 0.5--0.1*x,x=1,2,3,4,5
    batch_size = 2**(int(Pos[12])+4)         # 32 64 128--2**(x+4),x=1,2,3
    Learning_rate = 0.00005*math.pow(2,int((int(Pos[13])+1)/2))*math.pow(5,int(int(Pos[13])/2)) # 0.0001,0.0005,0.001,0.005    
    
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

#     model.summary()  
    quantile=0.025  # 0.05、0.95、0.025、0.975
    model.compile(loss=lambda y_t, y_p: quantile_loss(quantile, y_t, y_p), optimizer=tf.keras.optimizers.Adam(Learning_rate))
    
    earlyStop =[
         tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30,restore_best_weights=True,verbose=0),
         ] 
    history=model.fit([train_Xm,train_Xm2],train_labelm,callbacks=[earlyStop],epochs=epochs,batch_size=batch_size,validation_data=([valid_X,valid_X2], valid_label),verbose=0,shuffle=False)
    
    prediction = model.predict([test_X,test_X2])
    scaled_prediction = (prediction*(smax-smin)+smin).reshape(-1)
    result=model.evaluate([valid_X,valid_X2], valid_label)
    
    return result,scaled_prediction,model


model_filename='./ex/models_MWOA0025.h5'
def MWOA(objf,lb,ub,dim,SearchAgents_no,Max_iter): 
    Positions = np.zeros([SearchAgents_no, dim])
    X_WOA = np.zeros([SearchAgents_no, dim])
    X_DLH = np.zeros([SearchAgents_no, dim])
    fitness = np.zeros(SearchAgents_no)
    Fit_WOA = np.zeros(SearchAgents_no)
    Fit_DLH = np.zeros(SearchAgents_no)
    Leader_pos= np.zeros(dim)
    Leader_score=float("inf")
    neighbor = np.zeros([SearchAgents_no, SearchAgents_no])
    radius = np.zeros([SearchAgents_no,SearchAgents_no])
    dist_Position = np.zeros([SearchAgents_no,SearchAgents_no])
    t=0

    convergence_curve=np.zeros(Max_iter)
   
    Z = np.zeros([SearchAgents_no, dim])
    Z[0, :]=np.random.uniform(0,1,(1,dim))

    for i in range(1,SearchAgents_no):
        Z[i,:]=4.0*Z[i-1,:]*(1-Z[i-1,:])

    for i in range(0,SearchAgents_no):
        Positions[i,:] = lb + (ub-lb)*Z[i,:]   
        Positions[i,:]=np.clip(Positions[i,:],lb,ub)
        fitness[i],prediction,model=objf(Positions[i,:])
        if fitness[i]<Leader_score:
            Leader_score=fitness[i]+0
            Leader_pos=Positions[i,:]+0
            print('----------------------No iteration-------------------------------')
            print('Leader_pos:',Leader_pos)
            print('prediction:',prediction)
            pred_list.append(prediction)
            model.save_weights(model_filename)
            
    startTime1 = time.time()    
    while t < Max_iter:
        print('The number of iterations:',t+1)
        startTime0 = time.time()
        
        a=2-t*((2)/Max_iter) #a~[2,0]    
        a2=-1+t*((-1)/Max_iter)  #a2~[-1,-2]
    
        for i in range(0,SearchAgents_no):
            r1=random.random()
            r2=random.random()
        
            A=2*a*r1-a
            C=2*r2
        
            b=1
            l=(a2-1)*random.random()+1
        
            p=random.random()
        
            for j in range(0,dim):            
                if p<0.5:
                    if abs(A)>=1:
                        rand_leader_index=math.floor(SearchAgents_no*random.random())
                        X_rand=Positions[rand_leader_index,:]
                        D_X_rand=abs(C*X_rand[j]-Positions[i,j])
                        X_WOA[i,j]=X_rand[j]-A*D_X_rand                    
                    elif abs(A)<1:
                        D_Leader=abs(C*Leader_pos[j]-Positions[i,j])
                        X_WOA[i,j]=Leader_pos[j]-A*D_Leader                    
                elif p>=0.5:
                    distance2Leader=abs(Leader_pos[j]-Positions[i,j])
                    X_WOA[i,j]=distance2Leader*math.exp(b*l)*math.cos(l*2*math.pi)+Leader_pos[j]
                    
            X_WOA[i,:]=np.clip(X_WOA[i,:],lb,ub)      
            
        ## Calculate the candiadate position Xi-DLH
        for i in range(0,SearchAgents_no):   
            for j in range(0,SearchAgents_no):   
                radius[i,j]=np.sqrt(np.sum(np.square(Positions[i,:]-X_WOA[j,:])))
       
        for i in range(0,len(X_WOA)):   
            for j in range(0,len(Positions)):   
                dist_Position[i,j]=np.sqrt(np.sum(np.square(Positions[i,:]-Positions[j,:])))

        r1_DLH=np.random.permutation(SearchAgents_no)
        
        for i in range(0,SearchAgents_no):
            neighbor[i,:] = (dist_Position[i,:]<=radius[i,i])  
            Idx = np.array(np.where(neighbor[i,:]==1)).reshape(-1)          
            random_Idx_neighbor=np.array(np.random.randint(len(Idx),size=[1,dim])).reshape(-1) 
            for j in range(0,dim):
                X_DLH[i,j] = Positions[i,j] + random.random()*(Positions[Idx[random_Idx_neighbor[j]],j]- Positions[r1_DLH[i],j]) 

            X_DLH[i,:]=np.clip(X_DLH[i,:],lb,ub) 
            
        for i in range(0,SearchAgents_no):
            Fit_WOA[i],prediction,model=objf(X_WOA[i,:]) 
            if Fit_WOA[i]< fitness[i]:
                fitness[i]= Fit_WOA[i]+0
                Positions[i]= X_WOA[i,:]+0              
                if fitness[i]<Leader_score:
                    Leader_score=fitness[i]+0
                    Leader_pos=Positions[i,:]+0
                    print('-----------------------Fit_WOA------------------------------')
                    print('Leader_pos:',Leader_pos)
                    print('prediction:',prediction)
                    print('The number of iterations:',t+1)
                    pred_list.append(prediction)
                    model.save_weights(model_filename)
                    
            Fit_DLH[i],prediction,model = objf(X_DLH[i,:])                              
            if Fit_DLH[i]<fitness[i]:
                fitness[i]= Fit_DLH[i]+0
                Positions[i]= X_DLH[i,:]+0 
                if fitness[i]<Leader_score:
                    Leader_score=fitness[i]+0
                    Leader_pos=Positions[i,:]+0
                    print('-------------------------Fit_DLH----------------------------')
                    print('Leader_pos:',Leader_pos) 
                    print('prediction:',prediction)
                    print('The number of iterations:',t+1)
                    pred_list.append(prediction)
                    model.save_weights(model_filename)
                                         
        convergence_curve[t]=Leader_score
    
        t0 = time.time() - startTime0
        print('End iteration:',t+1,'Iteration time:', t0)
        
        t=t+1
        
    t1 = time.time() - startTime1
    print('Total time of MWOA:', t1)
    return Leader_score,Leader_pos,convergence_curve,pred_list



if __name__ == '__main__':
    dim = 14
    lb=np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1])
    ub=np.array([2.99,3.99,4.99,4.99,8.99,5.99,3.99,4.99,4.99,4.99,4.99,5.99,3.99,4.99])
    SearchAgents_no=15
    Max_iter=10
    print('-------------MWOA0025--------------')
    # MWOA( )  
    pred_value=pd.DataFrame(data=(np.array(pred_list).T))
    pred_value.to_csv("./ex/MWOA0025.csv",encoding='utf-8')
    
    print(Leader_score)
    print(Leader_pos)     
    print(convergence_curve)


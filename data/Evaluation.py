import tensorflow as tf
print("TensorFlow version is:",tf.__version__)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy
import random
import math
import keras.backend as K


# In[ ]:
# ISO

df = pd.read_csv('./IWOA0025_0.csv')
p_low=df['5'].values.reshape(-1, 1)
df = pd.read_csv('../true.csv')
data=df['0'].values.reshape(-1, 1)
q1=0.025
print(K.mean(K.maximum(q1*(data-p_low),(q1-1)*(data-p_low))))
avg0025 = numpy.zeros([len(data), 11])
avg0025[:,0]=p_low.reshape(-1)


# In[ ]:


df = pd.read_csv('./IWOA0025_1.csv')
p_low=df['4'].values.reshape(-1, 1)
df = pd.read_csv('../true.csv')
data=df['0'].values.reshape(-1, 1)
q1=0.025
print(K.mean(K.maximum(q1*(data-p_low),(q1-1)*(data-p_low))))
avg0025[:,1]=p_low.reshape(-1)


# In[ ]:


df = pd.read_csv('./IWOA0025_2.csv')
p_low=df['6'].values.reshape(-1, 1)
df = pd.read_csv('../true.csv')
data=df['0'].values.reshape(-1, 1)
q1=0.025
print(K.mean(K.maximum(q1*(data-p_low),(q1-1)*(data-p_low))))
avg0025[:,2]=p_low.reshape(-1)


# In[ ]:


df = pd.read_csv('./IWOA0025_3.csv')
p_low=df['13'].values.reshape(-1, 1)
df = pd.read_csv('../true.csv')
data=df['0'].values.reshape(-1, 1)
q1=0.025
print(K.mean(K.maximum(q1*(data-p_low),(q1-1)*(data-p_low))))
avg0025[:,3]=p_low.reshape(-1)


# In[ ]:


df = pd.read_csv('./IWOA0025_4.csv')
p_low=df['4'].values.reshape(-1, 1)
df = pd.read_csv('../true.csv')
data=df['0'].values.reshape(-1, 1)
q1=0.025
print(K.mean(K.maximum(q1*(data-p_low),(q1-1)*(data-p_low))))
avg0025[:,4]=p_low.reshape(-1)


# In[ ]:


df = pd.read_csv('./IWOA0025_5.csv')
p_low=df['11'].values.reshape(-1, 1)
df = pd.read_csv('../true.csv')
data=df['0'].values.reshape(-1, 1)
q1=0.025
print(K.mean(K.maximum(q1*(data-p_low),(q1-1)*(data-p_low))))
avg0025[:,5]=p_low.reshape(-1)


# In[ ]:


df = pd.read_csv('./IWOA0025_6.csv')
p_low=df['5'].values.reshape(-1, 1)
df = pd.read_csv('../true.csv')
data=df['0'].values.reshape(-1, 1)
q1=0.025
print(K.mean(K.maximum(q1*(data-p_low),(q1-1)*(data-p_low))))
avg0025[:,6]=p_low.reshape(-1)


# In[ ]:


df = pd.read_csv('./IWOA0025_7.csv')
p_low=df['12'].values.reshape(-1, 1)
df = pd.read_csv('../true.csv')
data=df['0'].values.reshape(-1, 1)
q1=0.025
print(K.mean(K.maximum(q1*(data-p_low),(q1-1)*(data-p_low))))
avg0025[:,7]=p_low.reshape(-1)


# In[ ]:


df = pd.read_csv('./IWOA0025_8.csv')
p_low=df['9'].values.reshape(-1, 1)
df = pd.read_csv('../true.csv')
data=df['0'].values.reshape(-1, 1)
q1=0.025
print(K.mean(K.maximum(q1*(data-p_low),(q1-1)*(data-p_low))))
avg0025[:,8]=p_low.reshape(-1)


# In[ ]:


df = pd.read_csv('./IWOA0025_9.csv')
p_low=df['3'].values.reshape(-1, 1)
df = pd.read_csv('../true.csv')
data=df['0'].values.reshape(-1, 1)
q1=0.025
print(K.mean(K.maximum(q1*(data-p_low),(q1-1)*(data-p_low))))
avg0025[:,9]=p_low.reshape(-1)
avg0025[:,10]=numpy.sum(avg0025, axis=1)/10
pred_value=pd.DataFrame(data=(np.array(avg0025)))
pred_value.to_csv("./IWOA0025_avg.csv",encoding='utf-8')


# In[ ]:





# In[ ]:


df = pd.read_csv('./IWOA0975_0.csv')
p_low=df['8'].values.reshape(-1, 1)
df = pd.read_csv('../true.csv')
data=df['0'].values.reshape(-1, 1)
q1=0.975
print(K.mean(K.maximum(q1*(data-p_low),(q1-1)*(data-p_low))))
avg0975 = numpy.zeros([len(data), 11])
avg0975[:,0]=p_low.reshape(-1)


# In[ ]:


df = pd.read_csv('./IWOA0975_1.csv')
p_low=df['4'].values.reshape(-1, 1)
df = pd.read_csv('../true.csv')
data=df['0'].values.reshape(-1, 1)
q1=0.975
print(K.mean(K.maximum(q1*(data-p_low),(q1-1)*(data-p_low))))
avg0975[:,1]=p_low.reshape(-1)


# In[ ]:


df = pd.read_csv('./IWOA0975_2.csv')
p_low=df['4'].values.reshape(-1, 1)
df = pd.read_csv('../true.csv')
data=df['0'].values.reshape(-1, 1)
q1=0.975
print(K.mean(K.maximum(q1*(data-p_low),(q1-1)*(data-p_low))))
avg0975[:,2]=p_low.reshape(-1)


# In[ ]:


df = pd.read_csv('./IWOA0975_3.csv')
p_low=df['6'].values.reshape(-1, 1)
df = pd.read_csv('../true.csv')
data=df['0'].values.reshape(-1, 1)
q1=0.975
print(K.mean(K.maximum(q1*(data-p_low),(q1-1)*(data-p_low))))
avg0975[:,3]=p_low.reshape(-1)


# In[ ]:


df = pd.read_csv('./IWOA0975_4.csv')
p_low=df['5'].values.reshape(-1, 1)
df = pd.read_csv('../true.csv')
data=df['0'].values.reshape(-1, 1)
q1=0.975
print(K.mean(K.maximum(q1*(data-p_low),(q1-1)*(data-p_low))))
avg0975[:,4]=p_low.reshape(-1)


# In[ ]:


df = pd.read_csv('./IWOA0975_5.csv')
p_low=df['10'].values.reshape(-1, 1)
df = pd.read_csv('../true.csv')
data=df['0'].values.reshape(-1, 1)
q1=0.975
print(K.mean(K.maximum(q1*(data-p_low),(q1-1)*(data-p_low))))
avg0975[:,5]=p_low.reshape(-1)


# In[ ]:


df = pd.read_csv('./IWOA0975_6.csv')
p_low=df['7'].values.reshape(-1, 1)
df = pd.read_csv('../true.csv')
data=df['0'].values.reshape(-1, 1)
q1=0.975
print(K.mean(K.maximum(q1*(data-p_low),(q1-1)*(data-p_low))))
avg0975[:,6]=p_low.reshape(-1)


# In[ ]:


df = pd.read_csv('./IWOA0975_7.csv')
p_low=df['6'].values.reshape(-1, 1)
df = pd.read_csv('../true.csv')
data=df['0'].values.reshape(-1, 1)
q1=0.975
print(K.mean(K.maximum(q1*(data-p_low),(q1-1)*(data-p_low))))
avg0975[:,7]=p_low.reshape(-1)


# In[ ]:


df = pd.read_csv('./IWOA0975_8.csv')
p_low=df['6'].values.reshape(-1, 1)
df = pd.read_csv('../true.csv')
data=df['0'].values.reshape(-1, 1)
q1=0.975
print(K.mean(K.maximum(q1*(data-p_low),(q1-1)*(data-p_low))))
avg0975[:,8]=p_low.reshape(-1)


# In[ ]:


df = pd.read_csv('./IWOA0975_9.csv')
p_low=df['9'].values.reshape(-1, 1)
df = pd.read_csv('../true.csv')
data=df['0'].values.reshape(-1, 1)
q1=0.975
print(K.mean(K.maximum(q1*(data-p_low),(q1-1)*(data-p_low))))
avg0975[:,9]=p_low.reshape(-1)
avg0975[:,10]=numpy.sum(avg0975, axis=1)/10
pred_value=pd.DataFrame(data=(np.array(avg0975)))
pred_value.to_csv("./IWOA0975_avg.csv",encoding='utf-8')


# In[ ]:


df = pd.read_csv('./IWOA0025_avg.csv')
p_low=df['10'].values.reshape(-1, 1)

df = pd.read_csv('./IWOA0975_avg.csv')
p_up=df['10'].values.reshape(-1, 1)

df = pd.read_csv('../true.csv')
data=df['0'].values.reshape(-1, 1)

count=0

for i in range(len(p_low)):
    if data[i]>=p_low[i] and data[i]<=p_up[i]:
        count=count+1

PICP = count/len(p_low)
print("PICP",PICP)

max0=np.max(data[:])
min0=np.min(data[:])
sum0=list(map(lambda x: (x[1]-x[0]) , zip(p_low,p_up)))
sum1=np.sum(sum0)/len(sum0)
PINAW = 1/(max0-min0)*sum1
print("PINAW",PINAW)

AIScount=0
ai=0.05
for i in range(len(p_low)):
    if data[i]<p_low[i]:
        AIScount=AIScount-2*ai*(p_up[i]-p_low[i])-4*(p_low[i]-data[i])
    elif data[i]>p_up[i]:
        AIScount=AIScount-2*ai*(p_up[i]-p_low[i])-4*(data[i]-p_up[i])
    else:
        AIScount=AIScount-2*ai*(p_up[i]-p_low[i])           
AIS = AIScount/len(p_low)
print("AIS",AIS)

q1=0.025
print(K.mean(K.maximum(q1*(data-p_low),(q1-1)*(data-p_low))))
q2=0.975
print(K.mean(K.maximum(q2*(data-p_up),(q2-1)*(data-p_up))))

# In[ ]:


import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Embedding, Conv1D, MaxPooling1D, GRU
from keras import regularizers
from keras.optimizers import SGD
from keras.optimizers import Adam, RMSprop
from keras.layers import LSTM, Bidirectional
import numpy as np
import csv
import pandas as pd
import hashlib
import random
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from keras_self_attention import SeqSelfAttention
import os, sys


# In[ ]:


seed = 7
np.random.seed(seed)
path= os.getcwd()
output_path = os.path.join(path,'output_results_save/')


# In[ ]:


df=pd.read_csv('sample_file_lstm.csv') # read training csv
#print(len(df))


# In[ ]:


max(df.astype('str').applymap(lambda x: len(x)).max()) # get max length of sequence_final to know how much to pad


# In[ ]:


maxlen=500 # 
X=df['sequence_final'].values # extract sequence_final as numpy array
#print(X.shape)


# In[ ]:


import pickle
# load mapping dict
with open(output_path+'mapping_dict_f.pkl', 'rb') as handle:
    map_dict = pickle.load(handle)


# In[ ]:


size=3


# In[ ]:


# convert raw data to numerical form and pad
X_train=[]
for val in X: # for val in all sequences
  sublist=[]
  chars=[val[i:i+size] for i in range(0, len(val), size)]
  for char in chars: # for group of characters in a sequence
    if len(char)==size:
      sublist.append(map_dict[char]) # convert chars to integer e.g. converts ACG UUG TTT to 1 2 3
  X_train.append(np.array(sublist))


X = pad_sequences(X_train, maxlen=int(maxlen/size), padding = 'post') # 

y=df['value'].values
#print(y.shape) # new data generated from sequences to array of integers
#print(X.shape)


# In[ ]:


model = Sequential()
model.add(Embedding(int(maxlen/size), 128, input_length=int(maxlen/size)))
model.add(LSTM(128, return_sequences = True))

model.add(SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                        attention_width=20,
                        attention_activation='tanh',
                       kernel_regularizer=keras.regularizers.l2(1e-6),
                       bias_regularizer=keras.regularizers.l1(1e-6),
                       attention_regularizer_weight=1e-6,
                       use_attention_bias=True,
                       name='Attention'))

model.add(LSTM(128))

model.add(Dense(200, activation='tanh'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer=Adam(0.001))
model.summary()


# In[ ]:


def unison_shuffled_copies(a, b):
  "shuffles 2 arrays together"
  assert len(a) == len(b)
  p = np.random.permutation(len(a))
  return a[p], b[p]


# In[ ]:


from sklearn.model_selection import KFold
cv = KFold(n_splits=5, random_state=1, shuffle=True) # make 5 fold object
i=0
models=[]
iters=10
loss=[]
dfs=[]



model_details={}
model_details['r2']=[]
model_details['rmse']=[]
model_details['model number']=[]
model_details['model loss']=[]
  
for i in range(iters):
  X,y=unison_shuffled_copies(X,y)
  preds={}
  preds['Actual']=[]
  preds['Prediction']=[]
  preds['Fold Number']=[]
  preds['Run_Iteration']=[]

  
  
  print("---------- Iteration: ", i)
  k=0
  for train_index, test_index in cv.split(X): # create splits of data in 5 folds and do training and validation
    X_train, X_val = X[train_index], X[test_index]
    y_train, y_val = y[train_index], y[test_index]
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,restore_best_weights=True)


    history = model.fit(
      X_train, y_train, batch_size=10, epochs=20,validation_data=(X_val,y_val),callbacks = [callback])
    loss.append(np.mean(history.history['val_loss']))

    models.append(model) # can later choose whichever model to use for testing i.e. models[2]=3rd model
    
    model.save_weights(output_path+'saved_models/model_number'+str(i)+'fold_'+str(k)+'.h5') # save model on disk

    pred=model.predict(X_val)
    for j,pr in enumerate(pred):
      preds['Actual'].append(y_val[j])
      preds['Prediction'].append(pr[0])
      preds['Fold Number'].append(k)
      preds['Run_Iteration'].append(i)
    model_details['r2'].append(r2_score(y_val, pred))
    model_details['rmse'].append(mean_squared_error(y_val, pred))
    model_details['model number'].append(i)
    model_details['model loss'].append(np.mean(history.history['val_loss']))
  
    k+=1
  df=pd.DataFrame(preds)
  df.to_csv(output_path + "Results_Iteration_"+str(i)+".csv")
  dfs.append(df)


df1=pd.DataFrame(model_details)
df1.to_csv(output_path + 'Model_Stats.csv')

df2=pd.concat(dfs)
df2.to_csv(output_path +'Model_Results.csv')


# In[ ]:


r2_score=df1["r2"].mean()
r2_score_stdev=round (df1["r2"].std(),2)
rmse_score=df1["rmse"].mean()
rmse_score_stdev=round (df1["rmse"].std(),2)


# In[ ]:


# final performance
print("metric\tavg\tstdev")
print("r2_score\t%.2f\t"% r2_score, r2_score_stdev)
print("rmse\t%.2f\t"% rmse_score, rmse_score_stdev)


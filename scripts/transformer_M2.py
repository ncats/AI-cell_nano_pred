#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling1D,LayerNormalization
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import itertools
from itertools import combinations, permutations
from sklearn.metrics import r2_score, mean_squared_error
import pickle
import os, sys
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


seed = 7
np.random.seed(seed)
path= os.getcwd()
output_path = os.path.join(path,'output_results_save/')


# In[ ]:


dataset_train="sample_file_transformer_m2.csv"
df=pd.read_csv(dataset_train) # read training csv


# In[ ]:


#file with physicochemical properties (numerical descriptors  transformed as categories or bins)
part_features = pd.read_csv('sample_file_transformer_m2_1.csv')


# In[ ]:


def rSubset(arr, r):
    return list(permutations(arr,r))


# In[ ]:


def creatPermutationFile(inputfile):
    combDic = {}
    with open(inputfile) as af:
        next(af)
        count = 0
        for line in af:
            # count = count +1
            if line!= "\n":
                splits = line.split(",")
                particalKey = splits[2].strip()
                seq = splits[1].strip()
                strandID = splits[0].strip()
                valuetoUse = splits[3].strip()
                # print(float(valuetoUse))
                # print(count)
                if particalKey not in combDic:
                    combDic[particalKey] = [[seq],[strandID],[valuetoUse],[line.strip()], [particalKey.strip()]]
                else:
                    combDic[particalKey][0].append(seq)
                    combDic[particalKey][1].append(strandID)
                    combDic[particalKey][2].append(valuetoUse)
                    combDic[particalKey][3].append(line.strip())
                    combDic[particalKey][4].append(particalKey.strip())

    return combDic


# In[ ]:


def Train_case(trainFile=dataset_train):

    
    seqlist = []
    strandList = []
    valueList = []
    partList = []
    combDic = creatPermutationFile(inputfile=trainFile)

    c = 0
    for evryParticle in combDic:
        c = c + 1
        seqTuple = rSubset(combDic[evryParticle][0], len(combDic[evryParticle][0]))
        strandIDTuple = rSubset(combDic[evryParticle][1], len(combDic[evryParticle][0]))

        new_seqTuple = ["".join(tups) for tups in seqTuple]
        seqlist = seqlist + new_seqTuple

        new_strandIDTuple = ["_".join(tup) for tup in strandIDTuple]
        strandList = strandList + new_strandIDTuple


        listofValue = [float(combDic[evryParticle][2][0])] * len(new_strandIDTuple)
        valueList = valueList + listofValue
        
        
        part = [combDic[evryParticle][4][0]] * len(new_strandIDTuple)
        partList = partList + part

    mergedList_df = pd.DataFrame({"mergedSeq": seqlist, "combinations": strandList, "valuestosue": valueList, "particleKey": partList})
    return (mergedList_df)


# In[ ]:


mergedList_df_train = Train_case()


# In[ ]:


maxlen=500 # 
X=mergedList_df_train['mergedSeq'].values # extract sequence_final as numpy array


# In[ ]:


import pickle
# load mapping dict
with open(output_path+'mapping_dict_f.pkl', 'rb') as handle:
    map_dict = pickle.load(handle)


# In[ ]:


size=3


# In[ ]:


mergedList_df_train = mergedList_df_train.merge(part_features,left_on='particleKey',right_on='Particle')


# In[ ]:


features_as_sequence = True # false if using the 2 input network

# convert raw data to numerical form and pad
X_train=[]
desc_cols = [x for x in mergedList_df_train.columns if 'desc' in x]

for val,descs in zip(mergedList_df_train['mergedSeq'],mergedList_df_train[desc_cols].values): # for val in all sequences
  sublist=[]
  chars=[val[i:i+size] for i in range(0, len(val), size)]
  for char in chars: # for group of characters in a sequence
    if len(char)==size:
      sublist.append(map_dict[char]) # convert chars to integer e.g. converts ACG UUG TTT to 1 2 3
  
# First option for adding features to the sequence
  if features_as_sequence:
    for desc in descs:
      sublist.append(map_dict[desc])
    X_train.append(np.array(sublist))
    
X = pad_sequences(X_train, maxlen=int(maxlen/size), padding = 'post') # 

y=mergedList_df_train['valuestosue'].values
Z=mergedList_df_train['particleKey'].values
W=mergedList_df_train['combinations'].values


# In[ ]:


# transformer blocks
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        
        
    def get_config(self):
        "has to override this as the model is custom"
        cfg = super().get_config()
        return cfg
    

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# In[ ]:


# embedding layers

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        
        
        
    def get_config(self):
        "has to override this as the model is custom"
        cfg = super().get_config()
        return cfg

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


# In[ ]:


vocab_size=len(map_dict)+1 # size of unique chars in data +1
embed_dim = 32  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

inputs = Input(shape=(int(maxlen/size),)) # input layer, size=max length
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim) # embedding layer
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim) # make transformer block
x = transformer_block(x)
x = GlobalAveragePooling1D()(x) # global pooling to select relevant features
x = Dropout(0.1)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.1)(x)
outputs = Dense(1, activation="linear")(x) # 1 output unit to predict value

model = Model(inputs=inputs, outputs=outputs)
model.compile("adam", "mean_absolute_error")

model.summary()


# In[ ]:


def unison_shuffled_copies(a, b):
  "shuffles 2 arrays together"
  assert len(a) == len(b)
  p = np.random.permutation(len(a))
  return a[p], b[p]


# In[ ]:


# This adds the values of particleKey as an aditional column to the X matrix.
X = np.append(X, W.reshape(-1,1), axis=1)
X = np.append(X, Z.reshape(-1,1), axis=1)


# In[ ]:


# Creates df from particle features
features_particle = mergedList_df_train[desc_cols].copy()
features_particle = features_particle.replace({col: map_dict for col in desc_cols}).values


# In[ ]:


from sklearn.model_selection import KFold
cv = KFold(n_splits=5, random_state=1, shuffle=True) # make 5 fold object
i=0
models=[]
iters=10
EPOCHS =20
loss=[]
dfs=[]
dfs_ungrouped = []



model_details={}
model_details['r2']=[]
model_details['rmse']=[]
model_details['model number']=[]
model_details['model loss']=[]
model_details['Fold number']=[]
  
for i in range(iters):
  X,y=unison_shuffled_copies(X,y)
  preds={}
  preds['Actual']=[]
  preds['Prediction']=[]
  preds['Fold Number']=[]
  preds['Run_Iteration']=[]
  preds['particleKey']=[]
  preds['combinations']=[]

  
  
  print("---------- Iteration: ", i)
  k=0
  for train_index, test_index in cv.split(X): # create splits of data in 5 folds and do training and validation
    partiKey = X[test_index,-1]
    w_vals_shuffled = X[test_index,-2]
    X_train, X_val = np.asarray(X[train_index,:-2]).astype(int), np.asarray(X[test_index,:-2]).astype(int)
    y_train, y_val = y[train_index], y[test_index]
    z_val,w_val = Z[test_index], W[test_index]
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,restore_best_weights=True)
    
    history = model.fit(
      X_train, y_train, batch_size=256, epochs=EPOCHS,validation_data=(X_val,y_val),callbacks = [callback])
    loss.append(np.mean(history.history['val_loss']))

    models.append(model) # can later choose whichever model to use for testing i.e. models[2]=3rd model
    
    model.save_weights(output_path+'saved_models/model_number'+str(i)+'fold_'+str(k)+'.h5') # save model on disk

    pred=model.predict(X_val)
    for j,pr in enumerate(pred):
      preds['Actual'].append(y_val[j])
      preds['Prediction'].append(pr[0])     
      preds['Fold Number'].append(k)
      preds['Run_Iteration'].append(i)
      preds['particleKey'].append(partiKey[j])
      preds['combinations'].append(w_vals_shuffled[j])

    #--We should be doing a groupby before calculating model stats
    preds1 = pd.DataFrame(preds)
    preds1 = preds1[(preds1['Run_Iteration']==i) & (preds1['Fold Number']==k)].groupby(['particleKey']).agg({'Actual':'mean','Prediction':'mean'}) 

    model_details['r2'].append(r2_score(preds1['Actual'], preds1['Prediction']))
    model_details['rmse'].append(mean_squared_error(preds1['Actual'], preds1['Prediction']))
    model_details['model number'].append(i)
    model_details['Fold number'].append(k)
    model_details['model loss'].append(np.mean(history.history['val_loss']))
  
    k+=1
    
  df=pd.DataFrame(preds).groupby(['particleKey','Fold Number','Run_Iteration']).agg({'Actual':'mean','Prediction':'mean'})
  df = df.reset_index().sort_values('Fold Number')
  df.to_csv(output_path + "Results_Iteration_"+str(i)+".csv")
  dfs.append(df)
    
  df_ungrouped=pd.DataFrame(preds)
  df_ungrouped.to_csv(output_path + "Results_Iteration_ungrouped"+str(i)+".csv")
  dfs_ungrouped.append(df_ungrouped)


df1=pd.DataFrame(model_details)
df1.to_csv(output_path + 'Model_Stats.csv')

df2=pd.concat(dfs)
df2.to_csv(output_path +'Model_Results.csv')


# In[ ]:


r2_score_df=df1["r2"].mean()
r2_score_stdev=round (df1["r2"].std(),2)
rmse_score=df1["rmse"].mean()
rmse_score_stdev=round (df1["rmse"].std(),2)


# In[ ]:


# final performance
print("metric\tavg\tstdev")
print("r2_score\t%.2f\t"% r2_score_df, r2_score_stdev)
print("rmse\t%.2f\t"% rmse_score, rmse_score_stdev)


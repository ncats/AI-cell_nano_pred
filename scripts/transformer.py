# In[ ]:


import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling1D,LayerNormalization
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import os, sys


# In[ ]:


seed = 7
np.random.seed(seed)
path= os.getcwd()
output_path = os.path.join(path,'output_results_save/')


# In[ ]:


df=pd.read_csv('sample_file_transformer.csv') # read training csv
#print(len(df))


# In[ ]:


max(df.astype('str').applymap(lambda x: len(x)).max()) # get max length of sequence_final to know how much to pad


# In[ ]:


maxlen=500 # 
X=df['sequence_final'].values # extract sequence_final as numpy array


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

    history = model.fit(
      X_train, y_train, batch_size=10, epochs=50,validation_data=(X_val,y_val))
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


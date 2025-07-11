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
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt
import itertools
from itertools import combinations, permutations
import pickle
import os, sys
from sklearn.model_selection import KFold


# In[ ]:


seed = 7
np.random.seed(seed)
path= os.getcwd()
output_path = os.path.join(path,'output_results_save/')


# In[ ]:


df = pd.read_csv('sample_file_rf.csv')
# Removes columns with NAN values
df = df.dropna().reset_index(drop=True)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

desc_cols = [x for x in df if 'desc' in str(x)]
nb_trees = 100

nb_iters = 10
nb_splits = 5

cv = KFold(n_splits=nb_splits, random_state=1, shuffle=True) # make n fold object

rmse_list = np.zeros((nb_iters, nb_splits))
r2_list = np.zeros((nb_iters, nb_splits))
std_list = np.zeros((nb_iters, nb_splits))

results_df = pd.DataFrame(columns=list(df.columns)+['iteration','cv_nb'])

details_df = pd.DataFrame(columns=['iteration','cv_nb','r2','rmse','std'])

for it in range(nb_iters):
    it_df = pd.DataFrame(columns=list(df.columns)+['iteration','cv_nb'])
    
    for cv_nb, (train_index, test_index) in enumerate(cv.split(df)):
        X_train, y_train = df.loc[train_index,desc_cols], df.loc[train_index,'Value']
        X_test, y_test = df.loc[test_index,desc_cols], df.loc[test_index,'Value']
        
        rfc = RandomForestRegressor()
        rfc.fit(X_train, y_train)
        pred = rfc.predict(X_test)
        
        rmse = mean_squared_error(y_test, pred, squared=False)
        r2 = r2_score(y_test, pred)
        std = np.std(pred)
        
        details_df = details_df.append({'iteration':it,'cv_nb':cv_nb,'r2':r2,'rmse':rmse,'std':std},ignore_index=True)
        
        temp_df = df.loc[test_index].copy()
        temp_df.loc[:,'pred'] = pred
        temp_df['iteration'] = it
        temp_df['cv_nb'] = cv_nb
        
        it_df = it_df.append(temp_df)
    
    results_df = results_df.append(it_df)
    
    it_df.to_csv(output_path + "Results_Iteration_"+str(it)+".csv",index=False)

    
    grouped_df = it_df.groupby(list(df.columns)).agg({'pred':'mean'})
    
results_df.to_csv(output_path + 'Model_results.csv',index=False)
details_df.to_csv(output_path + 'Model_stats.csv',index=False)


# In[ ]:


r2_score_df=details_df["r2"].mean()
r2_score_stdev=round (details_df["r2"].std(),2)
rmse_score=details_df["rmse"].mean()
rmse_score_stdev=round (details_df["rmse"].std(),2)


# In[ ]:


# final performance
print("metric\tavg\tstdev")
print("r2_score\t%.2f\t"% r2_score_df, r2_score_stdev)
print("rmse\t%.2f\t"% rmse_score, rmse_score_stdev)


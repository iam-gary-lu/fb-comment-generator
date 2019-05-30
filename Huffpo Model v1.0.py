#!/usr/bin/env python
# coding: utf-8

# In[11]:


import json
import os
import time
import pandas as pd
import random

import numpy as np
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding


from keras.backend import clear_session
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard


# In[2]:


m1 = time.time()
with open("huffpo_train<20.json") as f:
    data=json.load(f)
print('num samples: ',len(data))
print('json load time: ',time.time()-m1)

#cant be too large, will trigger a memory error later on. Failed at 1 Mill
np.random.shuffle(data)
data = data[:1000000]

print('total time: ',time.time()-m1)


# Need to optimize this later, use np arrays instead of python lists
# 

# In[3]:


"""
Need to optimize this later, replace use np arrays 
instead of python lists
"""
top_k = 10000

t = Tokenizer(num_words=top_k, oov_token='<unk>')
t.fit_on_texts(data)



start = time.time()
seqs = [t.texts_to_sequences(x) for x in data]
print('elaspted text->int seqs: ',time.time() - start)

seq_data = list()
for seq in seqs:
    flat = [num for sublist in seq for num in sublist]
    seq_data.append(flat)
    
print('total time: ',time.time() - start)


# For reference, seqs took 57s to go through 1 mill samples

# remnants of failed attemp
# 
# """
# seqs_np = np.array(seqs)
# 
# def flat(seq):
#     
#     return np.array(num for sublist in seq for num in sublist)
# 
# 
# a = seqs_np[0]
# b = np.apply_along_axis(flat, 0,a)
# """

# In[4]:


max_len = max([len(seq) for seq in seq_data])

sequences = pad_sequences(seq_data, maxlen=max_len, padding='pre')

print('Max Seq Len: %d' % max_len)

sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
vocab_sz = len(t.word_index)+1
print('vocab sz: ',vocab_sz)

#y_cat = to_categorical(y, num_classes=vocab_sz) #1-hot encoding 


# In[5]:


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, random_state=7)

#NN definition
model = Sequential()
model.add(Embedding(vocab_sz, 300, input_length=max_len-1))
model.add(LSTM(50))
model.add(Dense(vocab_sz, activation='softmax'))
print(model.summary())


# In[6]:


model.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

#save model structure as json

#model_json = model.to_json()
#with open("models/huffpo-model-v1.json", "w") as json_file:
#    json_file.write(model_json)


# In[8]:


"""

# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

"""


fpath = "weights/huffpo-v1/best-weights.hdf5"
checkpoint = ModelCheckpoint(fpath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')


BS = 42
tb = TensorBoard(log_dir="tensorboard-logs/{}".format(time.time()))
callback_lst = [checkpoint, tb]
#steps_per_epoch is num of batches that make up 1 epoch, defaults to size of train set
model.fit(X,y,batch_size=BS, validation_split=.15, epochs=20, callbacks=callback_lst, verbose=1)


# Interesting things
# - estimate how much memory a tensor of shape (185677, 300), where each value is a float32 bit
# 
# -training a model of ~60mil uses ~10g of memory. As verified nvidia-smi 
# 
# 

# In[12]:


clear_session()


# In[ ]:





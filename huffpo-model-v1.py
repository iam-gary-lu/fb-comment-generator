#!/usr/bin/env python
# coding: utf-8

# re

# In[11]:


import json
import os
import time
import pandas as pd
from gensim.models import KeyedVectors 
from gensim.test.utils import get_tmpfile
from gensim.models import Word2Vec

import numpy as np
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


huffpo_wv = KeyedVectors.load("huffpov1.model",mmap='r')
with open("huffpo_train<20.json") as f:
    maxlen_data=json.load(f)


# In[36]:


train_batch = maxlen_data[:100]

t = Tokenizer()
t.fit_on_texts(train_batch)

batch_sz = 5
#vocab_sz = len(huffpo_wv.wv.vocab)
vocab_sz = len(t.word_index)+1
int_encoded = t.texts_to_sequences(train_batch)
sequences = list()
error_count = 0
no_embedding_error = 0

for comment in train_batch:
    
    try:        
        comment_vect = list(map(lambda x: huffpo_wv.wv.get_vector(x) ,comment ))        
        for i in range(1, len(comment_vect)):
            sequence = comment_vect[:i+1]
            sequences.append(sequence)
    except KeyError: #lazy workaround for now. Not sure why some words aren't showing up in dict 
        error_count +=1
        #print('got keyerror for', comment)
        pass
    
    
    
    
max_len = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, dtype='float32',maxlen=max_len, padding='pre')

embedding_matrix = np.zeros((vocab_sz,300))
for word, i in t.word_index.items():
    try:
        word_embedding = huffpo_wv.wv.get_vector(word)
        embedding_matrix[i] = word_embedding
    except KeyError:
        no_embedding_error +=1
        print('no embedding for: ',word)
        pass
            
X, y = sequences[:,:-1], sequences[:,-1]


# In[43]:


X[0]


# In[40]:


X[1].shape


# In[42]:


y[1]


# In[21]:


X, y = sequences[:,:-1], sequences[:,-1]


# In[22]:


t.word_index


# In[18]:


len(embedding_matrix[3])


# In[20]:


t.word_index.items()


# In[9]:


sequences


# In[7]:





# In[ ]:





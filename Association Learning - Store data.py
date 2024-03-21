#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

import warnings
warnings.filterwarnings('ignore')


# Reading the file

# In[2]:


store_data = []

with open('D:\data analysis\dataset for analysis\store_data.csv') as f:
    content = f.readlines()
    traxs = [x.strip() for x in content]
    for each_trxs in traxs:
        store_data.append(each_trxs.split(','))


# In[3]:


store_data[:5]


# Feature Encoding

# In[4]:


one_hot_encoding = TransactionEncoder()
one_hot_tranxs = one_hot_encoding.fit(store_data).transform(store_data)
one_hot_tranxs


# In[5]:


#converting this matrix into dataframe
one_hot_tranxs_df = pd.DataFrame(one_hot_tranxs, columns=one_hot_encoding.columns_)
one_hot_tranxs_df.head()


# In[11]:


#model training
freq_itemset = apriori(one_hot_tranxs_df, min_support= 0.02, use_colnames= True)
freq_itemset.sample(10)


# In[12]:


rules = association_rules(freq_itemset, 
                         metric= 'lift',
                         min_threshold= 1
                         )


# In[13]:


rules.head(10)


# In[14]:


rules.sort_values('confidence', ascending=False)[:10]






#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv('Seattle-Weather.csv')
df = df.drop('date',axis=1)
X = df.drop('weather',axis=1)
y = df['weather']


X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y, test_size =0.4, random_state =42)

from sklearn.svm import SVC
LinSVC = SVC(kernel = 'linear', random_state = 0)
LinSVC.fit(X_train, y_train)
LinSVC.predict(X_test)

pickle.dump((LinSVC), open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# # Reading data

# In[2]:


data = pd.read_csv('Bengaluru_House_Data.csv')


# In[3]:


data.head()


# In[4]:


data['area_type'].value_counts()


# In[5]:


print(data.items())


# In[6]:


data['availability'].value_counts()


# In[7]:


data.drop(['availability'], inplace=True, axis = 1)


# In[8]:


data.head()


# In[9]:


data.isna()


# In[10]:


data.dropna(inplace=True)


# In[11]:


data.head()


# In[12]:


data = data.reset_index()
data.head()


# In[13]:


data = data.drop('index', axis = 1)


# In[14]:


data.head()


# In[15]:


data.drop('society', inplace=True, axis = 1)


# In[16]:


data.head()


# In[17]:


data = data[data['price'] < 500]


# In[18]:


data[data['price'] > 400].count()


# In[19]:


data['area_type'].value_counts()


# In[20]:


locations = data['location'].unique()
locations


# In[21]:


sizes = data['size'].unique()
sizes


# # EDA

# In[22]:


plt.figure(figsize=(12, 5))
sns.histplot(data['size'], bins = 100)
plt.tight_layout()


# In[23]:


sns.kdeplot(data['price'])


# In[24]:


sns.boxplot(data)
plt.tight_layout()


# # Preprocessing

# In[25]:


from sklearn.preprocessing import LabelEncoder


# In[26]:


labelEncoder = LabelEncoder()


# In[27]:


def encodeArea(areaType):
    if areaType == 'Super built-up  Area':
        return 0
    if areaType == 'Built-up  Area':
        return 1
    if areaType == 'Plot  Area':
        return 2
    if areaType == 'Carpet  Area':
        return 3


# In[28]:


data['area_type'] = data['area_type'].apply(encodeArea)


# In[29]:


data.head()


# In[30]:


locations = data['location'].unique()
locs = {}
for i in range(len(locations)):
    locs[locations[i]] = i

# locs


# In[31]:


def encodeLocation(loc):
    return locs[loc]


# In[32]:


data['location'] = data['location'].apply(encodeLocation)


# In[33]:


data.head()


# In[34]:


sizes = data['size'].unique()
s = {}
for i in range(len(sizes)):
    s[sizes[i]] = i
    
s


# In[35]:


def encodeSize(size):
    return s[size]


# In[36]:


data['size'] = data['size'].apply(encodeSize)


# In[37]:


data.head()


# In[38]:


def cleanSqft(sqft):
    if(sqft.isdigit()):
        return int(sqft)
    else:
        return 0


# In[39]:


data['total_sqft'] = data['total_sqft'].apply(cleanSqft)
data.info()


# In[40]:


data.head()


# In[41]:


data = data[data['total_sqft'] != 0]
data.info()


# In[42]:


data.head()


# In[43]:


from sklearn.preprocessing import StandardScaler


# # Model

# In[44]:


# from sklearn.linear_model import LinearRegression
# model = LinearRegression()


# In[45]:


from sklearn.model_selection import train_test_split


# In[46]:


x = data.drop('price', axis = 1)
y = data['price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)
x_train


# In[47]:


# model.fit(x_train, y_train)


# # Making Predictions

# In[48]:


# pred = model.predict(x_test)


# In[49]:


from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[50]:


# print(mean_squared_error(y_test, pred))


# In[51]:


# print(mean_absolute_error(y_test, pred))


# In[52]:


# z = [[2, 3, 3, 2785, 5.0, 3]]
# p = model.predict(z)
# print(p)


# # Other Models

# ### 1. Neural Network

# In[53]:


# from keras.models import Sequential
# import tensorflow as tf
# from keras.layers import Dense


# # In[54]:


# model2 = Sequential([
#     Dense(1000, input_shape = (6,), activation='relu'),
#     Dense(500, activation='relu'),
#     Dense(250, activation='relu'),
#     Dense(125, activation='relu'),
#     Dense(50, activation='relu'),
#     Dense(25, activation='relu'),
#     Dense(1, activation='relu')
# ])


# # In[55]:


# model2.compile(optimizer='adam', loss=tf.keras.losses.MeanAbsoluteError())


# # In[56]:


# model2.fit(x_train, y_train, epochs=10)


# # In[57]:


# pred2 = model2.predict(x_test)


# # In[58]:


# print(mean_absolute_error(y_test, pred2))


# # In[59]:


# print(mean_squared_error(y_test, pred2))


# # In[60]:


# z = [[2, 3, 1, 2785, 5.0, 3]]
# p = model2.predict(z)
# print(p)


# ### 2. Decision Tree Regressor

# In[61]:


# from sklearn.tree import DecisionTreeRegressor


# # In[62]:


# model3 = DecisionTreeRegressor()


# # In[63]:


# model3.fit(x_train, y_train)


# # In[64]:


# pred3 = model3.predict(x_test)


# # In[65]:


# print(mean_absolute_error(y_test, pred3))
# print(mean_squared_error(y_test, pred3))


# # In[66]:


# z = [[2, 3, 1, 2785, 5.0, 3]]
# z2 = [[0, 3, 0, 1170, 2.0, 1.0]]
# p = model3.predict(z2)
# print(p)


# ### 4. LightGBM

# In[67]:


from lightgbm import LGBMRegressor


# In[68]:


model4 = LGBMRegressor()


# In[69]:


model4.fit(x_train, y_train)


# In[70]:


pred4 = model4.predict(x_test)
print(mean_absolute_error(y_test, pred4))
print(np.sqrt( mean_squared_error(y_test, pred4)))


# In[71]:


# z = [[2, 3, 1, 2785, 5.0, 3]]
# z2 = [[0, 3, 0, 1170, 2.0, 1.0]]
# p = model4.predict(z)
# print(p)


# # Saving the model

# In[72]:


# import joblib


# # In[73]:


# joblib.dump(model4, "model.pkl")


# In[ ]:





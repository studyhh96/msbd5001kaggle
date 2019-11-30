#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from scipy import io as spio
from sklearn.decomposition import pca
from sklearn.preprocessing import StandardScaler


# In[2]:


train_data=pd.read_csv("../train.csv",parse_dates=['purchase_date','release_date'])
test_data=pd.read_csv("../test.csv",parse_dates=['purchase_date','release_date'])


# In[3]:


train_data


# In[4]:


test_data


# In[5]:


# find the missing data
train_data.isnull().sum()


# In[6]:


test_data.isnull().sum()


# In[7]:


train_data.total_positive_reviews.dropna().describe()


# In[8]:


train_data.total_negative_reviews.dropna().describe()


# In[9]:


#because mean is larger than 75% quarter, then choose the median as the null value
train_data.total_positive_reviews[train_data.total_positive_reviews.isnull()] = np.median(train_data.total_positive_reviews.dropna())
train_data.total_negative_reviews[train_data.total_negative_reviews.isnull()] = np.median(train_data.total_negative_reviews.dropna())


# In[10]:


#using train data to fill the test data
test_data.total_positive_reviews[test_data.total_positive_reviews.isnull()] = np.median(train_data.total_positive_reviews.dropna())
test_data.total_negative_reviews[test_data.total_negative_reviews.isnull()] = np.median(train_data.total_negative_reviews.dropna())


# In[11]:


train_data['purchase_date'].value_counts()


# In[12]:


#train_data['purchase_date'][train_data['id']==76]="2017-06-28"


# In[13]:


#using the mode to fill the missing data
train_data['purchase_date']=train_data['purchase_date'].fillna("2019-06-27")


# In[14]:


train_data[train_data['id']==5]


# In[15]:


test_data['purchase_date'].value_counts()


# In[16]:


#using the mode to fill the missing data
test_data['purchase_date']=test_data['purchase_date'].fillna("2018-06-21")


# In[17]:


train_data=train_data.drop(columns=['id'])
test_data=test_data.drop(columns=['id'])


# In[18]:


train_data.isnull().sum()


# In[19]:


test_data.isnull().sum()


# In[20]:


# transfer the true or false data to 1 or 1
train_data['free'] = train_data['is_free'].apply(lambda x: 1 if x ==True else 0)


# In[21]:


test_data['free'] = test_data['is_free'].apply(lambda x: 1 if x ==True else 0)


# In[22]:


#dummy the text data
multi_columns = ['genres','categories','tags']

genres_dummy = train_data['genres'].str.get_dummies(",").add_prefix("genres_")
categories_dummy = train_data['categories'].str.get_dummies(",").add_prefix("categories_")
tags_dummy = train_data['tags'].str.get_dummies(",").add_prefix("tags_")

train_data = train_data.drop(columns=multi_columns)
train_data = pd.concat([train_data,genres_dummy,categories_dummy,tags_dummy], axis=1)

test_genres_dummy = test_data['genres'].str.get_dummies(",").add_prefix("genres_")
test_categories_dummy = test_data['categories'].str.get_dummies(",").add_prefix("categories_")
test_tags_dummy = test_data['tags'].str.get_dummies(",").add_prefix("tags_")

test_data = test_data.drop(columns=multi_columns)
test_data = pd.concat([test_data,test_genres_dummy,test_categories_dummy,test_tags_dummy], axis=1)

train_data,test_data=train_data.align(test_data,join='left',axis=1)


# In[23]:


test_data=test_data.fillna(0)


# In[24]:


y_train = train_data['playtime_forever']


# In[25]:


train_data = train_data.drop(columns=['playtime_forever','is_free'])
test_data = test_data.drop(columns=['playtime_forever','is_free'])


# In[26]:


#do pca to reduce the dimensions
train_for_pca = train_data.drop(columns=['price','purchase_date','release_date','total_positive_reviews','total_negative_reviews','free'])
test_for_pca = test_data.drop(columns=['price','purchase_date','release_date','total_positive_reviews','total_negative_reviews','free'])


# In[27]:


model = pca.PCA(n_components=0.6).fit(train_for_pca)   # 拟合数据，n_components定义要降的维度
train_pca = model.transform(train_for_pca)
test_pca = model.transform(test_for_pca)


# In[28]:


train_join = train_data.loc[:,['price','purchase_date','release_date','total_positive_reviews','total_negative_reviews','free']]
test_join = test_data.loc[:,['price','purchase_date','release_date','total_positive_reviews','total_negative_reviews','free']]


# In[29]:


train_pca = pd.DataFrame(train_pca)
train_data = pd.concat([train_join,train_pca],axis=1)


# In[30]:


test_pca = pd.DataFrame(test_pca)
test_data = pd.concat([test_join,test_pca],axis=1)


# In[31]:


date='2006-1-1'
date = pd.to_datetime(date,format = '%Y-%m-%d %H:%M:%S')


# In[32]:


train_data['purchase_date'] = pd.to_datetime(train_data['purchase_date'],format = '%Y-%m-%d %H:%M:%S')


# In[33]:


# to deal with the date data
train_data['purchase_date'] = train_data['purchase_date']-date
train_data['release_date'] = train_data['release_date']-date


# In[34]:


test_data['purchase_date'] = pd.to_datetime(test_data['purchase_date'],format = '%Y-%m-%d')
test_data['purchase_date'] = test_data['purchase_date']-date
test_data['release_date'] = test_data['release_date']-date


# In[35]:


train_data['purchase_date'] = train_data['purchase_date'].astype('timedelta64[D]').astype(int)
train_data['release_date'] = train_data['release_date'].astype('timedelta64[D]').astype(int)


# In[36]:


test_data['purchase_date'] = test_data['purchase_date'].astype('timedelta64[D]').astype(int)
test_data['release_date'] = test_data['release_date'].astype('timedelta64[D]').astype(int)


# In[37]:


train = (train_data['purchase_date'] - train_data.min()) / (train_data.max() - train_data.min())
#test = (test_data - train_data.min()) / (train_data.max() - train_data.min())


# In[38]:


from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split


# In[39]:


lasso = Lasso(alpha =0.5)
lasso.fit(train_data, y_train)


# In[40]:


lasso_pred = lasso.predict(train_data).flatten()
print('MSE:', mean_squared_error(y_train,lasso_pred))


# In[41]:


lasso_test = lasso.predict(test_data).flatten()


# In[42]:


lasso_test.min()


# In[46]:


lasso_test=np.where(lasso_test>0,lasso_test,0)
lasso_test1=pd.DataFrame(lasso_test)
lasso_test1.columns = ['playtime_forever']
lasso_test1['id'] = range(len(lasso_test1))
order=['id','playtime_forever']
lasso_test1=lasso_test1[order]
lasso_test1.to_csv("lasso-0.5.csv",index=True)


# In[47]:


lasso_test1.mean()


# In[48]:


lasso_test2=np.where(lasso_test>5,lasso_test*3,lasso_test/2)
lasso_test2


# In[49]:


lasso_test=pd.DataFrame(lasso_test)
lasso_test.columns = ['playtime_forever']
lasso_test['id'] = range(len(lasso_test))
order=['id','playtime_forever']
lasso_test=lasso_test[order]
lasso_test
lasso_test.to_csv("lasso.csv",index=True)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import cufflinks as cf
import sklearn
from sklearn import svm, preprocessing 
import seaborn as sns
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('pprint', '')

import plotly
import plotly.graph_objs as go

# Must enable in order to use plotly off-line 
plotly.offline.init_notebook_mode()


# In[2]:


data = pd.read_csv("talysrisk.csv", sep=",")
data.head()


# In[3]:


data.shape


# In[4]:


quali_to_quanti= {
"Assurances":1,
"Industries et Services":2,
"Leasing":3,
"Banques":4,
"Microfinances": 5
}
data["Sector_Name"] = data["Sector_Name"].map(quali_to_quanti)
d = data.drop(['Unnamed: 0', 'Project_ID','Project_Name'], axis=1)
d.head()


# In[5]:



# spliting training and testing data
from sklearn.model_selection import train_test_split

X = d.drop(['RiskLevel','Sector_Name'], axis = 1)
y = d.iloc[:,3]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=15)


# In[6]:


# data normalization with sklearn
from sklearn.preprocessing import MinMaxScaler

# fit scaler on training data
norm = MinMaxScaler().fit(X_train)

# transform training data
X_train_norm = norm.transform(X_train)

# transform testing dataabs
X_test_norm = norm.transform(X_test)


# In[7]:


# data standardization with  sklearn
from sklearn.preprocessing import StandardScaler

# copy of datasets
X_train_stand = X_train.copy()
X_test_stand = X_test.copy()
X_stand = X.copy()

# numerical features

# apply standardization on numerical features
    
# fit on training data column
scale = StandardScaler().fit(X_train_stand)
    
# transform the training data column
X_train_stand = scale.transform(X_train_stand)
    
# transform the testing data column
X_test_stand = scale.transform(X_test_stand)

# transform all dataset
X_final = scale.transform(X_stand)


# In[8]:



# training a KNN model
from sklearn.neighbors import KNeighborsRegressor
# measuring RMSE score
from sklearn.metrics import mean_squared_error

# knn 
knn = KNeighborsRegressor(n_neighbors=7)

rmse = []

# raw, normalized and standardized training and testing data
trainX = [X_train, X_train_norm, X_train_stand]
testX = [X_test, X_test_norm, X_test_stand]

# model fitting and measuring RMSE
for i in range(len(trainX)):
    
    # fit
    knn.fit(trainX[i],y_train)
    # predict
    pred = knn.predict(testX[i])
    # RMSE
    rmse.append(np.sqrt(mean_squared_error(y_test,pred)))

# visualizing the result
df_knn = pd.DataFrame({'RMSE':rmse},index=['Original','Normalized','Standardized'])
df_knn


# In[9]:


from sklearn.model_selection import cross_val_score
score = []
for k in range(1,20):   # running for different K values to know which yields the max accuracy. 
    clf = KNeighborsRegressor(n_neighbors = k,  weights = 'distance', p=1)
    clf.fit(X_train_stand, y_train)
    score.append(clf.score(X_test_stand, y_test)) 
    #cross_val = cross_val_score(clf,X_test_stand,y_test, cv=5).mean()
    #score.append(cross_val)


# In[10]:


trace0 = go.Scatter(
    y = score,
    x = np.arange(1,len(score)+1), 
    mode = 'lines+markers', 
    marker = dict(
        color = 'rgb(150, 10, 10)'
    )
)
layout = go.Layout(
    title = '', 
    xaxis = dict(
        title = 'K value', 
        tickmode = 'linear'
    ),
    yaxis = dict(
        title = 'Score',
#         range = [0, 10000]
    )
)
fig = go.Figure(data = [trace0], layout = layout)
iplot(fig, filename='basic-line')


# In[11]:


k_max = score.index(max(score))+1
print( "At K = {}, Max Accuracy = {}".format(k_max, max(score)*100))


# In[15]:


clf = KNeighborsRegressor(n_neighbors = k_max,  weights = 'distance', p=1)
clf.fit(X_train_stand, y_train)
print(clf.score(X_test_stand, y_test ))   
y_pred = clf.predict(X_test_stand)
y_pred_final = clf.predict(X_final)


# In[13]:


trace0 = go.Scatter(
    y = y_test,
    x = np.arange(200), 
    mode = 'lines', 
    name = 'Actual Risk Level',
    marker = dict(
    color = 'rgb(10, 150, 50)')
)

trace1 = go.Scatter(
    y = y_pred,
    x = np.arange(200), 
    mode = 'lines', 
    name = 'Predicted Risk Level',
    line = dict(
        color = 'rgb(110, 50, 140)',
        dash = 'dot'
    )
)


layout = go.Layout(
    xaxis = dict(title = 'Index'), 
    yaxis = dict(title = 'Normalized Risk Level')
)

figure = go.Figure(data = [trace0, trace1], layout = layout)
plotly.offline.iplot(figure)


# In[32]:


y_pred_final = y_pred_final.round(decimals=0)


# In[33]:


result = d 
result['RiskLevel_pred'] = y_pred_final


# In[34]:


result['Project_ID'] =  data['Project_ID']
result


# In[35]:


quanti_to_quali= {
1.0:"Low",
2.0:"Medium",
3.0:"High"
}
result['RiskLevel_pred'] = result['RiskLevel_pred'].map(quanti_to_quali)
result.head()


# In[ ]:





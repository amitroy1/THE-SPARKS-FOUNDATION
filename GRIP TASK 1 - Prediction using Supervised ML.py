#!/usr/bin/env python
# coding: utf-8

# In[181]:


# NAME: Amit Roy
# BATCH: JULY21
# GRIP TASK 1 - Prediction using Supervised ML
# Predict the percentage of an student based on the no. of study hours


# In[ ]:


# Importing the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# In[167]:


# Importing the Data 
data = pd.read_csv ('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')
data.head(10)


# In[168]:


# To find if any Null Value is Present 
data.isnull().sum()


# In[169]:


# To see summary statistics

data.describe().T


# In[170]:


# To find outliers 
outl=data.columns
for i in outl:
    sns.boxplot(y=df[i],color="g")
    plt.show()


# In[171]:


## Data visualization with Line Plot
plt.figure(figsize=(15,25))
data.plot(kind="line",color=["r","g"])
plt.title("Hours Vs Scores", size=15)
plt.xlabel("Hours Studied", size=15)
plt.ylabel("Marks Percentage",size=15)


# In[172]:


## Data visualization with Area Plot
xmax = max(data.Hours)
xmin = min(data.Hours)
plt.figure(figsize=(20,25))
data.plot(kind="area",xlim=(xmin,xmax),color=["r","g"])
plt.title("Hours Vs Scores", size=15)
plt.xlabel("Hours Studied", size=15)
plt.ylabel("Marks Percentage",size=15)


# In[173]:


## Data visualization with Scatterplot
sns.set_style("darkgrid")
plt.figure(figsize=(6,6))
y=data["Scores"]
x=data["Hours"]
plt.scatter(x, y,color="g")
plt.title("Hours Vs Scores", size=15)
plt.xlabel("Hours Studied", size=15,color="g")
plt.ylabel("Marks Percentage",size=15,color="g")


# In[ ]:


# From above Data Visualization there looks to be correlation between the "Marks Percentage" and " Hours studies"


# In[174]:


## Let plot a regression line to confirm the correlation between the ' Marks Percentage' and 'Hours Studies'
sns.set_style("darkgrid")
plt.figure(figsize=(6,6))
sns.regplot(y=data["Scores"],x=data["Hours"],color="g")
plt.title("Regression Plot b/w Hours & Scores", size=15)
plt.xlabel("Hours Studied", size=15,color="g")
plt.ylabel("Marks Percentage",size=15,color="g")
print(data.corr())


# In[ ]:


# It is confirmed that the variable are positively correlated


# In[175]:


#Creating Model
X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
regression = LinearRegression()
regression.fit(train_X, train_y)


# In[152]:


pred_y = regression.predict(val_X)
prediction = pd.DataFrame({'Hours': [i[0] for i in val_X], 'Predicted Marks': [k for k in pred_y]})
prediction


# In[176]:



compare_scores = pd.DataFrame({'Actual Marks': val_y, 'Predicted Marks': pred_y})
compare_scores


# In[177]:


plt.scatter(x=val_X, y=val_y, color='r')
plt.plot(val_X, pred_y, color='g')
plt.title('Actual vs Predicted', size=20)
plt.ylabel('Marks Percentage', size=15)
plt.xlabel('Hours Studied', size=15)
plt.show()


# In[178]:


# Calculating the accuracy of the model
print('Mean absolute error: ',mean_absolute_error(val_y,pred_y))


# In[179]:


hours = [7]
answer = regression.predict([hours])
print("Score = {}".format(round(answer[0],3)))


# In[180]:


# According to the regression model if a student studies for 7 hours a day he/she is likely to score 71.524 marks.


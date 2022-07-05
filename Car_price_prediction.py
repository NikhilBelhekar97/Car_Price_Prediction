#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[10]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics


# # Importing csv file

# In[11]:


df=pd.read_csv('E:/data analysis  projects/data_analysis_project/project2/car data.csv')


# In[12]:


df.head(5)


# In[13]:


#No of Rows and columns


# In[14]:


df.shape


# In[15]:


#information about missing values


# In[16]:


df.isnull().sum()


# In[17]:


#Checking types of fuel type, seller type and transmission type


# In[18]:


print(df.Fuel_Type.value_counts())
print(df.Seller_Type.value_counts())
print(df.Transmission.value_counts())


# In[19]:


#Encoding the categorical data for fuel_type, seller_type and transmission_type


# In[20]:


df.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)
df.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)
df.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)


# In[21]:


# first five rows after encoding


# In[22]:


df.head(5)


# In[23]:


#Splitting the data and Target


# In[26]:


X = df.drop(['Car_Name','Selling_Price'],axis=1)
Y = df['Selling_Price']


# In[27]:


print(X)
print(Y)


# In[28]:


#Splitting Training and Test data


# In[29]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state=2)


# In[30]:


#Model_training


# In[31]:


# loading the linear regression model
lin_reg_model = LinearRegression()


# In[32]:


lin_reg_model.fit(X_train,Y_train)


# In[33]:


#Model_Evaluation
# prediction on Training data
training_data_prediction = lin_reg_model.predict(X_train)


# In[34]:


# R squared Error
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R squared Error : ", error_score)


# In[35]:


#Visualize the actual prices and Predicted prices
plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()


# In[36]:


# prediction on Training data
test_data_prediction = lin_reg_model.predict(X_test)


# In[37]:


# R squared Error
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared Error : ", error_score)


# In[39]:


plt.scatter(Y_test, test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()


# In[40]:


#Using_Lasso_regression


# In[41]:


lass_reg_model = Lasso()
lass_reg_model.fit(X_train,Y_train)


# In[42]:


training_data_prediction = lass_reg_model.predict(X_train)


# In[43]:


# R squared Error
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R squared Error : ", error_score)


# In[44]:


plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()


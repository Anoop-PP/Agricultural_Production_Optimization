#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

#for interactive analysis
from ipywidgets import interact


# In[2]:


df=pd.read_csv('data_agr.csv')


# In[3]:


df.shape


# In[11]:


df.isnull().sum()


# In[12]:


df.head()


# In[3]:


df.label.value_counts()


# Average Climatic and soil requirements

# In[4]:



print(" Average Ratio of Nitrogen in the Soil : {0:.2f}".format(df['N'].mean()))
print(" Average Ratio of Phosphorous in the Soil : {0:.2f}".format(df['P'].mean()))
print(" Average Ratio of Potassium in the Soil : {0:.2f}".format(df['K'].mean()))
print(" Average Temperature in Celsius : {0:.2f}".format(df['temperature'].mean()))
print(" Average Relative Humidity in % : {0:.2f}".format(df['humidity'].mean()))
print(" Average PH Value of the Soil : {0:.2f}".format(df['ph'].mean()))
print(" Average Rainfall in mm : {0:.2f}".format(df['rainfall'].mean()))


# In[5]:


@interact
def summary(crops = list(df['label'].value_counts().index)):
    x = df[df['label'] == crops]
    print("...........................................")
    print("Statistics for Nitrogen")
    print("Minimum Nitrogen required:", x['N'].min())
    print("Average Nitrogen required:", x['N'].mean())
    print("Maximum Nitrogen required:", x['N'].max())
    print("...........................................")
    print("Statistics for Phosphorous")
    print("Minimum Phosphorous required:", x['P'].min())
    print("Average Phosphorous required:", x['P'].mean())
    print("Maximum Phosphorous required:", x['P'].max())
    print("...........................................")
    print("Statistics for Pottasium")
    print("Minimum Pottasium required:", x['K'].min())
    print("Average Pottasium required:", x['K'].mean())
    print("Maximum Pottasium required:", x['K'].max())
    print("...........................................")
    print("Statistics for Temperature")
    print("Minimum Temperature required: {0:.2f}".format(x['temperature'].min()))
    print("Average Temperature required: {0:.2f}".format(x['temperature'].mean()))
    print("Maximum Temperature required: {0:.2f}".format(x['temperature'].max()))
    print("...........................................")
    print("Statistics for Humidity")
    print("Minimum Humidity required: {0:.2f}".format(x['humidity'].min()))
    print("Average Humidity required: {0:.2f}".format(x['humidity'].mean()))
    print("Maximum Humidity required: {0:.2f}".format(x['humidity'].max()))
    print("...........................................")
    print("Statistics for PH")
    print("Minimum PH required: {0:.2f}".format(x['ph'].min()))
    print("Average PH required: {0:.2f}".format(x['ph'].mean()))
    print("Maximum PH required: {0:.2f}".format(x['ph'].max()))
    print("...........................................")
    print("Statistics for Rainfall")
    print("Minimum Rainfall required: {0:.2f}".format(x['rainfall'].min()))
    print("Average Rainfall required: {0:.2f}".format(x['rainfall'].mean()))
    print("Maximum Rainfall required: {0:.2f}".format(x['rainfall'].max()))


# In[6]:


#Comparing Average requirement and conditions for each crop

@interact
def compare(conditions = ['N', 'P', 'K', 'temperature', 'ph', 'humidity', 'rainfall']):
    print("Average Value for", conditions, "is {0:.2f}".format(df[conditions].mean()))
    print("...........................................")
    print("Rice : {0:.2f}".format(df[(df['label'] == 'rice')][conditions].mean()))
    print("Black grams : {0:.2f}".format(df[(df['label'] == 'blackgram')][conditions].mean()))
    print("Banana : {0:.2f}".format(df[(df['label'] == 'banana')][conditions].mean()))
    print("Jute : {0:.2f}".format(df[(df['label'] == 'jute')][conditions].mean()))
    print("Coconut : {0:.2f}".format(df[(df['label'] == 'coconut')][conditions].mean()))
    print("Apple : {0:.2f}".format(df[(df['label'] == 'apple')][conditions].mean()))
    print("Papaya : {0:.2f}".format(df[(df['label'] == 'papaya')][conditions].mean()))
    print("Muskmelon : {0:.2f}".format(df[(df['label'] == 'muskmelon')][conditions].mean()))
    print("Grapes : {0:.2f}".format(df[(df['label'] == 'grapes')][conditions].mean()))
    print("Watermelon : {0:.2f}".format(df[(df['label'] == 'watermelon')][conditions].mean()))
    print("Kidney Beans : {0:.2f}".format(df[(df['label'] == 'kidneybeans')][conditions].mean()))
    print("Mung Beans : {0:.2f}".format(df[(df['label'] == 'mungbean')][conditions].mean()))
    print("Oranges : {0:.2f}".format(df[(df['label'] == 'orange')][conditions].mean()))
    print("Chick Peas : {0:.2f}".format(df[(df['label'] == 'chickpea')][conditions].mean()))
    print("Lentils : {0:.2f}".format(df[(df['label'] == 'lentil')][conditions].mean()))
    print("Cotton : {0:.2f}".format(df[(df['label'] == 'cotton')][conditions].mean()))
    print("Maize : {0:.2f}".format(df[(df['label'] == 'maize')][conditions].mean()))
    print("Moth Beans : {0:.2f}".format(df[(df['label'] == 'mothbeans')][conditions].mean()))
    print("Pigeon Peas : {0:.2f}".format(df[(df['label'] == 'pigeonpeas')][conditions].mean()))
    print("Mango : {0:.2f}".format(df[(df['label'] == 'mango')][conditions].mean()))
    print("Pomegranate : {0:.2f}".format(df[(df['label'] == 'pomegranate')][conditions].mean()))
    print("Coffee : {0:.2f}".format(df[(df['label'] == 'coffee')][conditions].mean()))


# In[7]:


#Checking the below and above Average Conditions

@interact
def compare(conditions = ['N', 'P', 'K', 'temperature', 'ph', 'humidity', 'rainfall']):
    print("Crops that require greater than average", conditions, '\n')
    print(df[df[conditions] > df[conditions].mean()]['label'].unique())
    print("...........................................")
    print("Crops that require less than average", conditions, '\n')
    print(df[df[conditions] <= df[conditions].mean()]['label'].unique())


# In[8]:


#Checking distributiion for each crop

plt.subplot(3,4,1)
sns.histplot(df['N'], color="yellow")
plt.xlabel('Nitrogen', fontsize = 12)
plt.grid()

plt.subplot(3,4,2)
sns.histplot(df['P'], color="orange")
plt.xlabel('Phosphorous', fontsize = 12)
plt.grid()

plt.subplot(3,4,3)
sns.histplot(df['K'], color="darkblue")
plt.xlabel('Pottasium', fontsize = 12)
plt.grid()

plt.subplot(3,4,4)
sns.histplot(df['temperature'], color="black")
plt.xlabel('Temperature', fontsize = 12)
plt.grid()

plt.subplot(2,4,5)
sns.histplot(df['rainfall'], color="grey")
plt.xlabel('Rainfall', fontsize = 12)
plt.grid()

plt.subplot(2,4,6)
sns.histplot(df['humidity'], color="lightgreen")
plt.xlabel('Humidity', fontsize = 12)
plt.grid()

plt.subplot(2,4,7)
sns.histplot(df['ph'], color="darkgreen")
plt.xlabel('PH Level', fontsize = 12)
plt.grid()

plt.suptitle('Distribution for Agricultural Conditions', fontsize = 20)
plt.show()


# In[27]:


#Checking that crops those have unusual requirements

print("Some Interesting Patterns")
print("...........................................")
print("Crops that require very High Ratio of Nitrogen Content in Soil:", df[df['N'] > 120]['label'].unique())
print("Crops that require very High Ratio of Phosphorous Content in Soil:", df[df['P'] > 100]['label'].unique())
print("Crops that require very High Ratio of Potassium Content in Soil:", df[df['K'] > 200]['label'].unique())
print("Crops that require very High Rainfall:", df[df['rainfall'] > 200]['label'].unique())
print("Crops that require very Low Temperature:", df[df['temperature'] < 10]['label'].unique())
print("Crops that require very High Temperature:", df[df['temperature'] > 40]['label'].unique())
print("Crops that require very Low Humidity:", df[df['humidity'] < 20]['label'].unique())
print("Crops that require very Low pH:", df[df['ph'] < 4]['label'].unique())
print("Crops that require very High pH:", df[df['ph'] > 9]['label'].unique())


# In[9]:


#Checking which crop to be grown according to the season

print("Summer Crops")
print(df[(df['temperature'] > 30) & (df['humidity'] > 50)]['label'].unique())
print("...........................................")
print("Winter Crops")
print(df[(df['temperature'] < 20) & (df['humidity'] > 30)]['label'].unique())
print("...........................................")
print("Monsoon Crops")
print(df[(df['rainfall'] > 200) & (df['humidity'] > 30)]['label'].unique())


# In[10]:


from sklearn.cluster import KMeans

#removing the labels column
x = df.drop(['label'], axis=1)

#selecting all the values of data
x = x.values

#checking the shape
print(x.shape)


# In[11]:


#Determining the optimum number of clusters within the Dataset

plt.rcParams['figure.figsize'] = (10,4)

wcss = []
for i in range(1,11):
    km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 2000, n_init = 10, random_state = 0)
    km.fit(x)
    wcss.append(km.inertia_)
    
#Plotting the results

plt.plot(range(1,11), wcss)
plt.title('Elbow Method', fontsize = 20)
plt.xlabel('No of Clusters')
plt.ylabel('wcss')
plt.show


# In[12]:



km = KMeans(n_clusters = 4, init = 'k-means++',  max_iter = 2000, n_init = 10, random_state = 0)
y_means = km.fit_predict(x)

#Finding the results
a = df['label']
y_means = pd.DataFrame(y_means)
z = pd.concat([y_means, a], axis = 1)
z = z.rename(columns = {0: 'cluster'})

#Checking the clusters for each crop
print("Lets Check the results after applying K Means Clustering Analysis \n")
print("Crops in First Cluster:", z[z['cluster'] == 0]['label'].unique())
print("...........................................")
print("Crops in Second Cluster:", z[z['cluster'] == 1]['label'].unique())
print("...........................................")
print("Crops in Third Cluster:", z[z['cluster'] == 2]['label'].unique())
print("...........................................")
print("Crops in Fourth Cluster:", z[z['cluster'] == 3]['label'].unique())


# In[13]:


#Splitting the Dataset for predictive modelling

y = df['label']
x = df.drop(['label'], axis=1)

print("Shape of x:", x.shape)
print("Shape of y:", y.shape)


# In[14]:


#Creating training and testing sets for results validation
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

print("The Shape Of x train:", x_train.shape)
print("The Shape Of x test:", x_test.shape)
print("The Shape Of y train:", y_train.shape)
print("The Shape Of y test:", y_test.shape)


# In[15]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)


# In[25]:


#Evaluating the model performance
from sklearn.metrics import confusion_matrix

#Printing the Confusing Matrix
plt.rcParams['figure.figsize'] = (10,10)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot = True, cmap = 'magma')
plt.title('Confusion Matrix For Logistic Regression', fontsize = 15)
plt.show()


# In[18]:


#Defining the classification Report
from sklearn.metrics import classification_report

#Printing the Classification Report
cr = classification_report(y_test, y_pred)
print(cr)


# In[22]:


df.tail()


# In[24]:


prediction = model.predict((np.array([[107, 40, 32, 20, 80, 7, 150]])))
print("The Suggested Crop for given climatic condition is :",prediction)


# In[ ]:





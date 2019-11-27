# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 21:24:37 2019

@author: Gyanendra
"""

# =============================================================================
# Classifying Personal Income
# =============================================================================
# =============================================================================
# Required Packages
# =============================================================================
#To work  with dataframe 
import pandas as pd

#To perform numerical operation
import numpy as np

# To visulaise data
import seaborn as sns

#partition the data
from sklearn.model_selection import train_test_split

# Importing Libraries for logistic Regression
from sklearn.linear_model import LogisticRegression

#importing performance matrics accuracy_score & confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix


###################################################

# =============================================================================
# Importing Data
# =============================================================================

data_income= pd.read_csv('income.csv')

#creating a copy of the data
data = data_income.copy()

"""
Exploratory Data Analysis

1. Getting to know the data
2. Data preprocessing (Missing Values)
3. Cross tables and data visualization

"""

# =============================================================================
# Getting to know the data
# =============================================================================

#*** To check for the variables' data type
print(data.info())

#*** Check for the missing value
data.isnull()

print('Data columns with null values:\n',data.isnull().sum())
#**** No missing values !

#*** Summary of numerical variables
summary_num = data.describe()
print(summary_num)

#*** Summary of Categorical variables
summary_cate = data.describe(include="O")
print(summary_cate)

#*** frequency of each category
data['JobType'].value_counts()
data['occupation'].value_counts()

#Checking for Unique Classes
print(np.unique(data['JobType']))
print(np.unique(data['occupation']))
#*** There exists '?' instead of non

"""
Go back and read the data by including "na_values['?'] to "
"""
data= pd.read_csv('income.csv',na_values=['?'])

# =============================================================================
# Data pre-processing
# =============================================================================
data.isnull().sum()

missing = data[data.isnull().any(axis=1)]
# axis=1 => to consider at least one column value is missing

"""
Ponits to note:
    1. Missing values in JobType = 1809
    2. Missing values inn Occupation = 1816
    3. There are 1809 rows where two specific columns i.e. occupation and JobType
    have missing values
    4. (1816-1809) = 7 => You still have occupation unfilled for these 7 rows. Because,
    JobType is never worked.
    
"""

data2 = data.dropna(axis=0)

#Relationship between independent variables
correlation = data2.corr()

# =============================================================================
# Cross tables & Data Visulaisation 
# =============================================================================

#Extracting the columns names
data.columns

# =============================================================================
# Gender proportion table:
# =============================================================================
gender = pd.crosstab(index = data2["gender"],
                     columns = 'count',
                     normalize = True)
print(gender)

# =============================================================================
# Gender vs salary Status:
# =============================================================================
gender_salstat = pd.crosstab(index = data2["gender"],
                             columns = data2['SalStat'],
                             margins = True,
                             normalize = 'index')
print(gender_salstat)

# =============================================================================
# Frequency distribution of 'Salary status'
# =============================================================================
Salstat = sns.countplot(data2['SalStat'])

""" 75 % of People's salary status is  <=50,000
& 25% of people's salary status is >50,000
"""

############# Histogram of Age ##############
sns.distplot(data2['age'],bins=10, kde=False)
# People with age 20-45 age are high in frequency


################ Box Plot - Age vs salary status ##############

sns.boxplot('SalStat','age',data=data2)
data2.groupby('SalStat')['age'].median()

## people with 35-50 age are more likely to earn > 50000
## people with 25-35 age are more likely to earn <= 50000

#*** Capital Gain 
sns.distplot(data2['capitalgain'],bins=10, kde=False)

sns.distplot(data2['capitalloss'],bins=10, kde=False)

# =============================================================================
#  Logistic Model
# =============================================================================

#reindexing the salary status names to 0,1
data['SalStat']=data2['SalStat'].map({'less than or equal to 50,000':0,'greater than 50,000':1})
print(data2['SalStat'])

new_data=pd.get_dummies(data2,drop_first=True)

#Starting the column names
columns_list=list(new_data.columns)
#Starting the input names from data
features = list(set(columns_list)-set(['SalStat']))
print(features)

#Starting the output values in y
y=new_data['SalStat'].values
print(y)

#Separating the input names from data
features = list(set(columns_list)-set(['SalStat']))

#Starting the output value in y
y=new_data['SalStat'].values
print(y)


#Storing the  value from input features
x= new_data[features].values
print(x)

#splitting the data into train and test
train_x,test_x,train_y,test_y= train_test_split(x,y,test_size=0.3,random_state=0)

# Make on instance of the model
logistic = LogisticRegression()

#Fitting the values for x and y
logistic.fit(train_x,train_y)
logistic.coef_
logistic.intercept_

#Prediction from test data
prediction = logistic.predict(test_x)
print(prediction)

#Confusion matrix
confusion_matrix = confusion_matrix(test_y, prediction)
print(confusion_matrix)

#Calculating the accuracy
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)

#Printing the misclassified values from prediction

print('Misclassified samples: %d' % (test_y != prediction).sum())

# =============================================================================
# Logistic Regression - Removing Insignificant Variables
# =============================================================================

#Reindexing the salary status names to 0,1
data2['SalStat']=data2['SalStat'].map({'less than or equal to 50,000':0,'greater than 50,000':1})
print(data2['SalStat'])

cols = ['gender','nativecountry','race','JobType']
new_data = data2.drop(cols,axis=1)

new_data = pd.get_dummies(new_data,  drop_first=True)

 #Storing the column names
columns_list = list(new_data.columns)
print(columns_list)

#Seprating the input names from data
features = list(set(columns_list)-set(['SalStat']))

#Storing the output values in y
y=new_data['SalStat'].values
print(y)

#Storing the values from input features
x = new_data[features].values
print(x)

#Splitiing the data into train and test
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3,random_state=0)

#Make an instance of the Model
logistic = LogisticRegression()

#Fitting the values for x and y
logistic.fit(train_x,train_y)

#Prediction from test data
prediction = logistic.predict(test_x)

#Calculating the accuracy
accuracy_score = accuracy_score(test_y,prediction)
print(accuracy_score)

#Printing the misclassified values from prediction
print('Misclassified samples: %d' % (test_y != prediction).sum())

# =============================================================================
# KNN
# =============================================================================

#importing the library of KNN
from sklearn.neighbors import KNeighborsClassifier

#Import library 
#import matplotlib.pyplot  as plt

#Storing the K nearest neighbors classifier
KNN_classifier =  KNeighborsClassifier(n_neighbors = 5)

#Fitting the values for X and Y
KNN_classifier.fit(train_x,train_y)

#Predicting the test values with model
prediction = KNN_classifier.predict(test_x)

#Performane metric check
confusion_matrix = confusion_matrix(test_y,prediction)
print("\t","Predicted values")
print("Original values","\n",confusion_matrix)

#Calculation the accuracy
accuracy_score = accuracy_score(test_y,prediction)
print(accuracy_score)

print('Misclassified samples: %d' % (test_y != prediction).sum())

"""
Effect of k value on classifier
"""

Misclassified_sample = []

#Calculating error for K values between 1 and 20
for i in range(1,20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x, train_y)
    pred_i = knn.predict(test_x)
    Misclassified_sample.append((test_y != pred_i).sum())

print(Misclassified_sample)

# =============================================================================
# End of Script
# =============================================================================

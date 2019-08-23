# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math


#Collect the data

titanic_data = pd.read_csv('D:/Subhasish/Subs/AI/titanic.csv')

#Analyse the data

print('No of pasangers   ' +str(len(titanic_data.index)))

sns.countplot(x='survived', data=titanic_data)

sns.countplot(x='survived', hue='sex', data=titanic_data)

sns.countplot(x='survived', hue='pclass', data=titanic_data)

titanic_data['age'].plot.hist()

titanic_data['fare'].plot.hist(bins=20, figsize=(10,5))

titanic_data.info()

sns.countplot(x='sibsp', data=titanic_data)

sns.countplot(x='survived', hue='sibsp', data=titanic_data)


#Data wrangling or Cleaning

data = titanic_data.isnull()

titanic_data.isnull().sum()

sns.heatmap(titanic_data.isnull(),yticklabels=False, cmap='viridis')

sns.boxplot(x='pclass', y='age', data=titanic_data)

titanic_data.drop('cabin', axis=1, inplace=True)


titanic_data.isna().sum()
##titanic_data.dropna(inplace=True) ######not possible as all the columns has null value

sns.heatmap(titanic_data.isnull(),yticklabels=False, cbar=False)

sex = pd.get_dummies(titanic_data['sex'], drop_first=True)

embark = pd.get_dummies(titanic_data['embarked'], drop_first=True)

pcl = pd.get_dummies(titanic_data['pclass'],drop_first=True)

titanic_data = pd.concat([titanic_data,sex, embark, pcl], axis=1)

titanic_data.drop(['sex', 'embarked', 'pclass', 'name', 'ticket', 'home.dest', 'boat'], axis=1, inplace=True)
titanic_data.drop('body', axis=1, inplace=True)

titanic_data.isna().sum()

from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values = np.nan, strategy = 'mean')

X = titanic_data.drop('survived', axis=1)
y = titanic_data['survived']

X.isna().sum()

X.iloc[:,0:4] = imp.fit_transform(X.iloc[:,0:4])


####Train data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

prediction = logmodel.predict(X_test)

from sklearn.metrics import classification_report

y_test = y_test.fillna(0)
classification_report(y_test, prediction)

from sklearn.metrics import confusion_matrix








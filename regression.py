# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 17:51:28 2019

@author: User
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'D:\Subhasish\Subs\AI\Simplilearn\ML\Hands-on-Assignment\OSL Datasets\Datasets for hands-on assignments\Chapter 2 - Techniques of ML\Regression\LinearR\Price.csv'
housedata = pd.read_csv(path)

X =  housedata.iloc[:,:-1].values
y = housedata.iloc[:,1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression

lnrregAgent = LinearRegression()
lnrregAgent.fit(X_train, y_train)

lnrregAgent.predict(X_test)

lnrregAgent.score(X_train, y_train)
lnrregAgent.score(X_test, y_test)

plt.scatter(X_train, y_train, color='green')
plt.plot(X_train, lnrregAgent.predict(X_train), color='red')
plt.title('Compare Training data')
plt.xlabel('House Area')
plt.ylabel('Price')
plt.show()


plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, lnrregAgent.predict(X_test), color='red')
plt.title('Compare Test data')
plt.xlabel('House Area')
plt.ylabel('Price')
plt.show()






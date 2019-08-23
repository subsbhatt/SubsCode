# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 19:00:21 2019

@author: User
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'D:\Subhasish\Subs\AI\Simplilearn\ML\Hands-on-Assignment\OSL Datasets\Datasets for hands-on assignments\Chapter 2 - Techniques of ML\Regression\PolynomialR\Insurance.csv'
insuranceData = pd.read_csv(path)

X = insuranceData.iloc[:,:-1].values
y = insuranceData.iloc[:,1].values


#from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
lnrregressor = LinearRegression()
#lnrregressor.fit(X_Train, y_train)
lnrregressor.fit(X,y)

plt.scatter(X, y, color='green')
plt.plot(X, lnrregressor.predict(X), color='red')
plt.title('Linear Data')
plt.xlabel('Age')
plt.ylabel('Preminum')
plt.show()

from sklearn.preprocessing import PolynomialFeatures

pln_ftr = PolynomialFeatures(degree=3)
poly_matrx = pln_ftr.fit_transform(X)

lnrregressor1 = LinearRegression()
lnrregressor1.fit(poly_matrx, y)


plt.scatter(X, y, color='red')
plt.plot(X, lnrregressor1.predict(poly_matrx), color='blue')
plt.title('Polynomial Data')
plt.xlabel('Age')
plt.ylabel('Preminum')
plt.show()



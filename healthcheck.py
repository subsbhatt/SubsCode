# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

path = 'D:\Subhasish\Subs\AI\Simplilearn\ML\Hands-on-Assignment\OSL Datasets\Datasets for hands-on assignments\Chapter 3 - Data pre-processing\Data-Preprocessing\Health.csv'

healthdata = pd.read_csv(path)

healthdata.dtypes

X = healthdata.iloc[:,:-1].values

y = healthdata.iloc[:,3].values

#from sklearn.preprocessing import Imputer
#missingValImp = Imputer(missing_values='NaN', strategy='mean', axis=0)

from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values=np.nan,strategy='mean')
X[:,1:3] = imp.fit_transform(X[:,1:3])

from sklearn.preprocessing import LabelEncoder

X_lblEncd = LabelEncoder()
y_lblEncd = LabelEncoder()

X[:,0] = X_lblEncd.fit_transform(X[:,0])
y = y_lblEncd.fit_transform(y)


from sklearn.preprocessing import OneHotEncoder

ohc = OneHotEncoder(categorical_features=[0])

X = ohc.fit_transform(X).toarray()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

X_train = scale.fit_transform(X_train)
X_test = scale.fit_transform(X_test)




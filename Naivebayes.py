# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 08:32:13 2019

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups()
categories = data.target_names

train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)

#print(train.data[5])

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

model = make_pipeline(TfidfVectorizer(),MultinomialNB())

model.fit(train.data, train.target)

labels = model.predict(test.data)

from sklearn.metrics import confusion_matrix

mat = confusion_matrix(test.target, labels)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', 
            cbar=False, xticklabels=train.target_names, 
            yticklabels = train.target_names)

plt.xlabel('Ture Label')
plt.ylabel('Predicted Label')

def predict_category(s, train=train, model=model):
    pred = model.predict([s])
    print("here is the out", pred)
    return train.target_names[pred[0]]

predict_category('Sending Orange to out of Country')


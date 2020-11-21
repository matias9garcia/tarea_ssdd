# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 15:47:06 2020

tarea de SSDD
"""
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

data_kddtest = pd.read_csv("KDDTest+.txt", header=None)
data_kddtrain = pd.read_csv("KDDTrain+_20Percent.txt", header=None)

"Se transforman los atributos no númericos, utilizando labelEncoder"
X_train = data_kddtrain.drop([42], axis=1)
etiqueta_train = data_kddtrain[41] #Etiqueta

X_test = data_kddtest.drop([42], axis=1)
etiqueta_test = data_kddtest[41] #Etiqueta

label_train=LabelEncoder()
for c in  X_train.columns:
    if(X_train[c].dtype=='object'):
        X_train[c]=label_train.fit_transform(X_train[c])
    else:
        X_train[c]=X_train[c]

label_test=LabelEncoder()
for c in  X_test.columns:
    if(X_test[c].dtype=='object'):
        X_test[c]=label_test.fit_transform(X_test[c])
    else:
        X_test[c]=X_test[c]

etiqueta_test_bipolar = []
for etiqueta in etiqueta_test:
    if  etiqueta == "normal":
        y = 1
    else :
        y = -1
    etiqueta_test_bipolar.append(y)

etiqueta_train_bipolar = []
for etiqueta in etiqueta_train:
    if etiqueta == "normal":
        y = 1
    else:
        y = -1
    etiqueta_train_bipolar.append(y)


"Normalización de la data"
train_std = (X_train - X_train.min(axis=0)) / (X_train.max(axis=0) - X_train.min(axis=0))
train_normalize = train_std * (0.99 - 0.1) + 0.1

test_std = (X_test - X_test.min(axis=0)) / (X_test.max(axis=0) - X_test.min(axis=0))
test_normalize = test_std * (0.99 - 0.1) + 0.1


train_normalize.to_csv(r'train.txt', header=None, index=None, sep=' ', mode='a')
test_normalize.to_csv(r'test.txt', header=None, index=None, sep=' ', mode='a')

print(etiqueta_test_bipolar)
print(etiqueta_train_bipolar)

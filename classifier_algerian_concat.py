# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 10:00:42 2021

@author: Neuza
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas as pd

data_bejaia = pd.read_csv("bejaia_region.csv")
data_sidi = pd.read_csv("Sidi-Bel_region.csv")

dfs = [data_bejaia,data_sidi]

data_concat = pd.concat(dfs)

X = data_concat.iloc[:, 3:-1]
y = data_concat.iloc[:, -1]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 0)

from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier(n_estimators=128)
RFC.fit(X_train,y_train)
test_pred = RFC.predict(X_test)
train_pred = RFC.predict(X_train)

from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test, test_pred)
acc2 = accuracy_score(y_train,train_pred)

print("Acuracia no teste: {}".format(acc))
print("Acuracia no treino: {}".format(acc2))


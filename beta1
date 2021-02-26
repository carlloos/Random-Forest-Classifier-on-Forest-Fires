# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas as pd

data_bejaia = pd.read_csv("bejaia_region.csv")
data_sidi = pd.read_csv("Sidi-Bel_region.csv")

X_bejaia = data_bejaia.iloc[:,3:-1]
y_bejaia = data_bejaia.iloc[:, -1]

X_sidi = data_sidi.iloc[:,3:-1]
y_sidi = data_sidi.iloc[:, -1]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit_transform(X_sidi)
scaler.fit_transform(X_bejaia)

from sklearn.ensemble import RandomForestClassifier

DTC = RandomForestClassifier()


#Utilizando os dados de beijaia como treinamento para o nosso classificador

DTC.fit(X_bejaia,y_bejaia)

y_pred = DTC.predict(X_sidi)

from sklearn.metrics import accuracy_score

acc = accuracy_score(y_sidi, y_pred)

print(acc)




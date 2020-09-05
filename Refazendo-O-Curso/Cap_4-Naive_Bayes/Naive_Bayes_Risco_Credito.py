#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 15:30:14 2020

@author: joao
"""

import pandas as pd

# Carregar base
base = pd.read_csv('../risco_credito.csv')

# Separar em Previsores e Classe
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

# Transformar dados categóricos em numéricos
from sklearn.preprocessing import LabelEncoder
labelEncoder_previsores = LabelEncoder()
previsores[:, 0] = labelEncoder_previsores.fit_transform(previsores[:, 0])
previsores[:, 1] = labelEncoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 2] = labelEncoder_previsores.fit_transform(previsores[:, 2])
previsores[:, 3] = labelEncoder_previsores.fit_transform(previsores[:, 3])

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores, classe)

# Classificação
resultado = classificador.predict([[0,0,1,2], [3,0,0,0]])

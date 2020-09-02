#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 19:52:09 2020

@author: joao
"""

import pandas as pd

base = pd.read_csv('../census.csv')

# Separar a base em Previsores e Classe
previsores = base.iloc[:, 0:14].values
classe= base.iloc[:, 14].values

# Transformando variáveis categórias em numéricas (previsores)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelEncoder_previsores = LabelEncoder()
previsores[:, 1] = labelEncoder_previsores.fit_transform(previsores[: ,1])
previsores[:, 3] = labelEncoder_previsores.fit_transform(previsores[: ,3])
previsores[:, 5] = labelEncoder_previsores.fit_transform(previsores[: ,5])
previsores[:, 6] = labelEncoder_previsores.fit_transform(previsores[: ,6])
previsores[:, 7] = labelEncoder_previsores.fit_transform(previsores[: ,7])
previsores[:, 8] = labelEncoder_previsores.fit_transform(previsores[: ,8])
previsores[:, 9] = labelEncoder_previsores.fit_transform(previsores[: ,9])
previsores[:, 13] = labelEncoder_previsores.fit_transform(previsores[: ,13])

# Variáveis Dummy
from sklearn.compose import ColumnTransformer

one_hot_encoder = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [1,3,5,6,7,8,9,13])], remainder='passthrough'
)
previsores = one_hot_encoder.fit_transform(previsores).toarray()

# Transformando variáveis categórias em numéricas (classe)
labelEncoder_class = LabelEncoder()
classe = labelEncoder_class.fit_transform(previsores[: ,14])

# Escalonamento dos valores
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

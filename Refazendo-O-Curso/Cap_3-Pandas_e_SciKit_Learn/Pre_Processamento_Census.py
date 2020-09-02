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

# Transformando variáveis categórias em numéricas
from sklearn.preprocessing import LabelEncoder
labelEncoder_previsores = LabelEncoder()
previsores[:, 1] = labelEncoder_previsores.fit_transform(previsores[: ,1])
previsores[:, 3] = labelEncoder_previsores.fit_transform(previsores[: ,3])
previsores[:, 5] = labelEncoder_previsores.fit_transform(previsores[: ,5])
previsores[:, 6] = labelEncoder_previsores.fit_transform(previsores[: ,6])
previsores[:, 7] = labelEncoder_previsores.fit_transform(previsores[: ,7])
previsores[:, 8] = labelEncoder_previsores.fit_transform(previsores[: ,8])
previsores[:, 9] = labelEncoder_previsores.fit_transform(previsores[: ,9])

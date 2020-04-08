#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 14:30:46 2020

@author: joao
"""

import pandas as pd

#importar base de dados
base_de_dados = pd.read_csv('risco_credito_2.csv')

#separar a base entre previsores e classe
previsores = base_de_dados.iloc[:, 0:4].values
classe = base_de_dados.iloc[:, 4].values

#pre processamento -> dados para formato discreto
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
previsores[:, 0] = label_encoder.fit_transform(previsores[:, 0])
previsores[:, 1] = label_encoder.fit_transform(previsores[:, 1])
previsores[:, 2] = label_encoder.fit_transform(previsores[:, 2])
previsores[:, 3] = label_encoder.fit_transform(previsores[:, 3])

#Regressão Logística
from sklearn.linear_model import LogisticRegression
classificador = LogisticRegression()
classificador.fit(previsores, classe)
print(classificador.intercept_)
print(classificador.coef_)

#resultado da classificação
resultado = classificador.predict([ [0,0,1,2], [3,0,0,0] ])
resultado2 = classificador.predict_proba([[0,0,1,2], [3,0,0,0]])

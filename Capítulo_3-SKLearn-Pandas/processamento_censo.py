#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 12:10:08 2020

@author: joao
"""

import pandas as pd

#importar a base de dados
base = pd.read_csv('census.csv')

#separar a base em dados categóricos
previsores = base.iloc[:, 0:14].values
classe = base.iloc[:,14].values

from sklearn.preprocessing import LabelEncoder

#transformar as informações que estão em string para o formato discreto
labelEncoder_previsores = LabelEncoder()
labelEncoder_previsores.fit_transform(previsores[:, 1])

#alterar na base de previsores para os valores discretos
previsores[:, 1] = labelEncoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelEncoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelEncoder_previsores.fit_transform(previsores[:, 5])
previsores[:, 6] = labelEncoder_previsores.fit_transform(previsores[:, 6])
previsores[:, 7] = labelEncoder_previsores.fit_transform(previsores[:, 7])
previsores[:, 8] = labelEncoder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelEncoder_previsores.fit_transform(previsores[:, 9])
previsores[:, 13] = labelEncoder_previsores.fit_transform(previsores[:, 13])

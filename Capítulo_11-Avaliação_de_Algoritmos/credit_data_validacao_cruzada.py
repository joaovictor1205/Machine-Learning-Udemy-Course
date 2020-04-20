#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 15:22:37 2020

@author: joao
"""

import pandas as pd

#importar base de dados
base = pd.read_csv('original.csv')

#tirar a média das idades (que estejam corretas) para corrigir os valores negativos
media = base['age'][base.age > 0].mean()

#atribuir o valor da média para os registros que estão com a idade incorreta
base.loc[base['age'] < 0, 'age'] = media

#Separação do dataframe em Previsores e Classe
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer()
imputer = imputer.fit(previsores[:, 0:3])
previsores[:, 0:3] = imputer.transform(previsores[:, 0:3])

#escalonar o atributo Previsores para todos terem o mesmo 'peso'
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

#cross validation
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

classificador = GaussianNB()

resultado = cross_val_score(classificador, previsores, classe, cv = 10)
resultado.mean() #media
resultado.std() #desvio padrão

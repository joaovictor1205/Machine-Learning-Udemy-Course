#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 15:35:02 2020

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

#naive bayes
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()

import numpy as np
a = np.zeros(5)
previsores.shape
previsores.shape[0]
b = np.zeros(shape=(previsores.shape[0], 1))

#stratified k fold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix

kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 0)
resultados = []
matrizes = []

for indice_treinamento, indice_teste in kfold.split(previsores,
                                                    np.zeros(shape = (previsores.shape[0], 1))):
    #print('Indice treinamento: ', indice_treinamento,
    #      'Indice teste: ', indice_teste)
    classificador.fit(previsores[indice_treinamento], classe[indice_treinamento])
    previsoes = classificador.predict(previsores[indice_teste])
    precisao = accuracy_score(classe[indice_teste], previsoes)
    matrizes.append(confusion_matrix(classe[indice_teste], previsoes))
    resultados.append(precisao)

matriz_final = np.mean(matrizes, axis = 0)
resultados = np.asarray(resultados)
resultados.mean()

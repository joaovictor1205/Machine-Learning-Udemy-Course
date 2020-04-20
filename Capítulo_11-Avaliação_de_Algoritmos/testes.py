#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 16:17:14 2020

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

import numpy as np

#naive bayes
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()

#stratified k fold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

resultados_testes = []

for i in range(30):   
    
    kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = i)
    resultados = []
    
    for indice_treinamento, indice_teste in kfold.split(previsores,
                                                        np.zeros(shape = (previsores.shape[0], 1))):
    
        classificador.fit(previsores[indice_treinamento], classe[indice_treinamento])
        previsoes = classificador.predict(previsores[indice_teste])
        precisao = accuracy_score(classe[indice_teste], previsoes)
        resultados.append(precisao)
    resultados = np.asarray(resultados)
    media = resultados.mean()
    resultados_testes.append(media)
    
resultados_testes = np.asarray(resultados_testes)
for i in range(resultados_testes.size):
    print(str(resultados_testes[i]).replace('.', ','))

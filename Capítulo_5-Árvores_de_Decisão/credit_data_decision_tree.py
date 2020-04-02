#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 18:15:55 2020

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

#dividir a base entre treino e teste
from sklearn.model_selection import train_test_split
previsores_treino, previsores_teste, classe_treino, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)
 
#Decision Tree
from sklearn.tree import DecisionTreeClassifier
classificador = DecisionTreeClassifier(criterion='entropy', random_state=0)
classificador.fit(previsores_treino, classe_treino)

#resultado com os datasets de teste
resultado = classificador.predict(previsores_teste)

#analisar erros da classificação
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, resultado)
matriz = confusion_matrix(classe_teste, resultado)

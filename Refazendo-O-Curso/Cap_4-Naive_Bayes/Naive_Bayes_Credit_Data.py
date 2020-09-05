#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 15:58:38 2020

@author: joao
"""

import pandas as pd

base = pd.read_csv('../credit_data.csv')

# Correção de dados inconsistentes -> idade negativa
media_idades = base['age'][base.age > 0].mean()
base.loc[base.age < 0, 'age'] = media_idades

# Separação da base em Previsores e Classe
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

# Correção de dados inconsistentes -> não preenchidos
from sklearn.impute import SimpleImputer

imputer = SimpleImputer()
imputer = imputer.fit(previsores[:, 0:3])
previsores[:, 0:3] = imputer.transform(previsores[:, 0:3])

# Escalonamento dos atributos
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# Separação dos previsores em Treino e Teste
from sklearn.model_selection import train_test_split
previsores_treino, previsores_teste, classe_treino, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

# Naive Bayes -> classificador cria a tabela de probabilidades
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores_treino, classe_treino)

# Resultado das previsoes do classificador
classificacao=classificador.predict(previsores_teste)

# Taxa de acerto
from sklearn.metrics import accuracy_score, confusion_matrix
precisao = accuracy_score(classe_teste, classificacao)
matriz = confusion_matrix(classe_teste, classificacao)

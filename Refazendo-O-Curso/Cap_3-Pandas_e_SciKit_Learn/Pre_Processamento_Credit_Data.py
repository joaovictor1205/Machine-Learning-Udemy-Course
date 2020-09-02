# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd

data_frame = pd.read_csv('credit_data.csv')
data_frame.describe()

# Dados inconsistentes -> idade negativa
data_frame.loc[data_frame['age'] < 0]

# 1a solução -> Deletar registros com valores inconsistentes
# data_frame.drop(data_frame[data_frame.age < 0].index, inplace=True)

# 2a solução -> Preencher os valores com a média das idades
# sem os valores inconsistentes
data_frame.mean()
media_idades = data_frame['age'][data_frame.age > 0].mean()
data_frame.loc[data_frame.age < 0, 'age'] = media_idades


# Separação do Data Frame em Previsores e Classe

# Pegar todas as linhas das colunas de 1 a 3, sem a coluna 0, pois o ID não será utilizado
previsores = data_frame.iloc[:, 1:4].values

# Pegar todas as linhas da coluna 4
classe = data_frame.iloc[:, 4].values


# Dados inconsistentes -> valor não preenchido
data_frame.loc[pd.isnull(data_frame['age'])]

from sklearn.impute import SimpleImputer

imputer = SimpleImputer()
imputer = imputer.fit(previsores[:, 0:3])
previsores[:, 0:3] = imputer.transform(previsores[:, 0:3])

# Escalonamento dos atributos
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)


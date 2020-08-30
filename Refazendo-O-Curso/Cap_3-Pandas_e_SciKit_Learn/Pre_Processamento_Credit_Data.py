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

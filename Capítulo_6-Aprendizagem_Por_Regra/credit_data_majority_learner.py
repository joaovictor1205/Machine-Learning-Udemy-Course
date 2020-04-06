#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 17:15:40 2020

@author: joao
"""

import Orange

base_de_dados = Orange.data.Table('original.csv')
base_de_dados.domain

base_dividida = Orange.evaluation.testing.sample(base_de_dados, n=0.25)
base_treino = base_dividida[1]
base_teste = base_dividida[0]

len(base_treino)
len(base_teste)

classificador = Orange.classification.MajorityLearner()
resultado = Orange.evaluation.testing.TestOnTestData(base_treino, base_teste, [classificador])
print(Orange.evaluation.CA(resultado))

from collections import Counter
print(Counter(str(d.get_class()) for d in base_teste))

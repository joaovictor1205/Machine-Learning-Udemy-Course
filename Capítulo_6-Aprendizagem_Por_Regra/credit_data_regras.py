#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:53:53 2020

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

cn2_learner = Orange.classification.rules.CN2Learner()
classificador = cn2_learner(base_treino)

for regras in classificador.rule_list:
    print(regras)

resultado = Orange.evaluation.testing.TestOnTestData(base_treino, base_teste, [classificador])
print(Orange.evaluation.Precision(resultado))

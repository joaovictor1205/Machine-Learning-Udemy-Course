#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:07:40 2020

@author: joao
"""

import Orange

base_de_dados = Orange.data.Table('risco_credito.csv')
base_de_dados.domain

cn2_learner = Orange.classification.rules.CN2Learner()
classificador = cn2_learner(base_de_dados)

for regras in classificador.rule_list:
    print(regras)

resultado = classificador([['boa', 'alta', 'nenhuma', 'acima_35'], ['ruim', 'alta', 'adequada', '0_15']])

for i in resultado:
    print(base_de_dados.domain.class_var.values[i])

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 14:26:43 2020

@author: joao
"""

from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit
from pybrain.structure import FullConnection

#criação da rede neural
rede = FeedForwardNetwork()

#criação das camadas
camada_entrada = LinearLayer(2)
camada_oculta = SigmoidLayer(3)
camada_saida = SigmoidLayer(1)

bias_1 = BiasUnit()
bias_2 = BiasUnit()

#inicialização das camadas dentro da rede neural
rede.addModule(camada_entrada)
rede.addModule(camada_oculta)
rede.addModule(camada_saida)
rede.addModule(bias_1)
rede.addModule(bias_2)

#ligação entre as camadas
ligacao_entrada_oculta = FullConnection(camada_entrada, camada_oculta)
ligacao_oculta_saida = FullConnection(camada_oculta, camada_saida)
ligacao_bias_oculta = FullConnection(bias_1, camada_oculta)
ligacao_bias_saida = FullConnection(bias_2, camada_saida)

#criação da rede
rede.sortModules()

print(rede)
print(ligacao_entrada_oculta.params)
print(ligacao_oculta_saida)
print(ligacao_bias_oculta)
print(ligacao_bias_saida)

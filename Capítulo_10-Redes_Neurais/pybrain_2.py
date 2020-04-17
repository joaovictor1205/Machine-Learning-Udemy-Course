#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 14:45:43 2020

@author: joao
"""

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.structure.modules import SigmoidLayer

#inicialização da rede com 2 camadas de entrada, 3 camadas ocultas e 1 camada de saída
rede = buildNetwork(2, 3, 1)
print(rede['in'])
print(rede['hidden0'])
print(rede['out'])
print(rede['bias'])

#2 atributos previsores + 1 classe
base = SupervisedDataSet(2,1)

#XOR
base.addSample((0, 0), (0, ))
base.addSample((0, 1), (1, ))
base.addSample((1, 0), (1, ))
base.addSample((1, 1), (0, ))

print(base['input']) #previsores
print(base['target']) #classe

treinamento = BackpropTrainer(rede, 
                              dataset = base, 
                              learningrate=0.01, 
                              momentum=0.06)

for i in range(1,1000):
    erro = treinamento.train()
    if i % 1000 == 0:
        print("Erro: %s" % erro)

print(rede.activate([0,0]))
print(rede.activate([0,1]))
print(rede.activate([1,0]))
print(rede.activate([1,1]))

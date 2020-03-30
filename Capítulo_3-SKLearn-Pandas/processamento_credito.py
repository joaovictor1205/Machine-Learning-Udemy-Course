import pandas as pd

#importar base de dados
base = pd.read_csv('original.csv') 

#encontrar clientes com idade negativa
base.loc[base['age'] < 0 ]

#apagar coluna da idade
base.drop('age', 1, inplace=True)

#apagar os registros que tenham idade negativa
base.drop(base[base.age < 0].index, inplace=True)

#tirar a média das idades para corrigir os valores negativos
base['age'].mean()

#tirar a média das idades (que estejam corretas) para corrigir os valores negativos
media = base['age'][base.age > 0].mean()

#atribuir o valor da média para os registros que estão com a idade incorreta
base.loc[base['age'] < 0, 'age'] = media

valor_nulo = base.loc[pd.isnull(base['age'])]


#Separação do dataframe em Previsores e Classes
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

import numpy as np 
import pandas as pd 

train = pd.read_csv('train.csv')
test =  pd.read_csv('test.csv')

from sklearn.tree import DecisionTreeClassifier

modelo = DecisionTreeClassifier(random_state=0)

def transformar_sexo(valor):
    if valor == 'female':
        return 1
    else: 
        return 0

train['Sex_binario'] = train['Sex'].map(transformar_sexo)

variaveis = ['Sex_binario','Age']

x = train[variaveis]
y = train['Survived']

x = x.fillna(-1)
#print(x.head())
#print(y.head())

modelo.fit(x,y)

test['Sex_binario'] = test['Sex'].map(transformar_sexo)

x_previsao = test[variaveis]

x_previsao = x_previsao.fillna(-1)

print(x_previsao.head())

p = modelo.predict(x_previsao)

submission = pd.Series(p, index = test['PassengerId'], name='Survived')

submission.to_csv("Previsao_modelo.csv",header=True)
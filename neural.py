import numpy as np 
import pandas as pd 

train = pd.read_csv('data/train.csv')
test =  pd.read_csv('data/test.csv')

#from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold

from sklearn.neural_network import MLPClassifier
            
modelo = MLPClassifier(solver = 'lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=0)

def transformar_sexo(valor):
    if valor == 'female':
        return 1
    else: 
        return 0

train['Sex_binario'] = train['Sex'].map(transformar_sexo)

train['Embarked_S'] = (train['Embarked'] =='S').astype(int)
train['Embarked_C'] = (train['Embarked'] =='C').astype(int)
train['Embarked_Q'] = (train['Embarked'] =='Q').astype(int)
train['Cabine_Nula'] = train['Cabin'].isnull().astype(int)

train['Miss'] = train['Name'].str.contains("Miss").astype(int)
train['Mrs'] = train['Name'].str.contains("Mrs").astype(int)
train['Master'] = train['Name'].str.contains("Master").astype(int)
train['Col'] = train['Name'].str.contains("Col").astype(int)
train['Major'] = train['Name'].str.contains("Major").astype(int)
train['Mr'] = train['Name'].str.contains("Mr").astype(int)

variaveis = ['Sex_binario','Age','Pclass','SibSp','Parch','Fare','Embarked_S','Embarked_C','Embarked_Q',
             'Cabine_Nula','Miss','Mrs','Master','Col','Major','Mr']

x = train[variaveis]
y = train['Survived']

x = x.fillna(-1)
#print(x.head())
#print(y.head())

print(train.head())

x_treino, x_validacao, y_treino, y_validacao = train_test_split(x, y)

modelo.fit(x_treino,y_treino)
p = modelo.predict(x_validacao)

print(np.mean(y_validacao == p))


resultados = []

kf = RepeatedKFold(n_splits = 2, n_repeats=10, random_state=10)
    
for line_train, line_valid in kf.split(x):
    print("Treino:", line_train.shape[0])  
    print("Validacao:", line_valid.shape[0])
        
    x_treino, x_validacao = x.iloc[line_train], x.iloc[line_valid]
    y_treino, y_validacao = y.iloc[line_train], y.iloc[line_valid]
   
    modelo = MLPClassifier(solver = 'lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=0)

    modelo.fit(x_treino,y_treino)
    p = modelo.predict(x_validacao)
    acc = np.mean(y_validacao == p)
    resultados.append(acc)
    print("Acc:",acc)
    print()
        
print("Acuracia:",np.mean(resultados))


'''
test['Sex_binario'] = test['Sex'].map(transformar_sexo)

x_previsao = test[variaveis]

x_previsao = x_previsao.fillna(-1)

print(x_previsao.head())

p = modelo.predict(x_previsao)

submission = pd.Series(p, index = test['PassengerId'], name='Survived')

submission.to_csv("Previsao_modelo.csv",header=True)
'''

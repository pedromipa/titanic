import numpy as np 
import pandas as pd 

train = pd.read_csv('train.csv')
test =  pd.read_csv('test.csv')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
np.random.seed(0)
            
modelo = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)

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

x_treino, x_validacao, y_treino, y_validacao = train_test_split(x, y)

modelo.fit(x_treino,y_treino)
p = modelo.predict(x_validacao)

print(np.mean(y_validacao == p))


resultados = []
for rep in range(10):
    print("Rep:",rep)
    kf = KFold(2, shuffle=True, random_state=rep)
    
    for line_train, line_valid in kf.split(x):
        print("Treino:", line_train.shape[0])  
        print("Validacao:", line_valid.shape[0])
        
        x_treino, x_validacao = x.iloc[line_train], x.iloc[line_valid]
        y_treino, y_validacao = y.iloc[line_train], y.iloc[line_valid]
        
        modelo = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
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

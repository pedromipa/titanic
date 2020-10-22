import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


train = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

women = train.loc[train.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)

men = train.loc[train.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)


y = train["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train[features])
X_test = pd.get_dummies(test_data[features])


#corr= train.corr()
#corr.style.background_gradient(cmap='coolwarm')
#plt.matshow(train.corr())
# train.corr().style.format("{:.2}").background_gradient(cmap=plt.get_cmap('coolwarm'),axis=1)

correlation = train.corr()

correlation.to_csv('correlacao.csv', index=False)

'''
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
'''
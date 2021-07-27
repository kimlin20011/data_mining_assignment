import pandas as pd
from sklearn import preprocessing, linear_model
from sklearn.feature_selection import f_regression
import numpy as np

### read csv
file = "weather.csv"
df = pd.read_csv(file, encoding='Shift-JIS')
print(df)

def sigmoid(z):
      return 1.0 / (1 + np.exp(-z))


# 創造 dummy variables
label_encoder = preprocessing.LabelEncoder()
encoded_play = label_encoder.fit_transform(df["play"])
print(encoded_play)

# 建立 train_X
train_X = pd.DataFrame([df["temperature"],df["humidity"]]).T
# 建立模型
logistic_regr = linear_model.LogisticRegression()
logistic_regr.fit(train_X, encoded_play)

# 印出係數
print("Coefficient : ",logistic_regr.coef_)

# 印出截距
print("Interception : ",logistic_regr.intercept_ )

# 印出 p-value
print("p-value: ",f_regression(train_X, encoded_play))

# 計算準確率
play_predictions = logistic_regr.predict(train_X)
accuracy = logistic_regr.score(train_X, encoded_play)
print("Mean accuracy : ", accuracy)

print("Decision function :\n ", logistic_regr.decision_function(train_X))

print("Probability of play [no yes] :\n", np.round(logistic_regr.predict_proba(train_X),3))
print("Paramaters : \n", logistic_regr.get_params(deep=True))
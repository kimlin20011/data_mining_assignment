import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_csv('weather.csv')
# print(df.head(5))
#sns.countplot('temperature', hue='WINorLOSS', data=df)
# plt.show()


x = df.drop('outlook', axis=1)
y = df['outlook']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=1)
logmodel = LogisticRegression()
logmodel.fit(x_train, y_train)

predictions = logmodel.predict(x_test)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(accuracy_score(y_test, predictions))

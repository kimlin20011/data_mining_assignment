import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
# from sklearn.datasets import load_iris
# from sklearn import tree
# from sklearn.metrics import accuracy_score
# import numpy as np
le = preprocessing.LabelEncoder()
df = pd.read_csv('./uriage6.csv', header=None)
arr = df.values
print(type(arr))
# df.head()

arr = df.values
X = arr[:, :-1]
y = arr[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)
clf = MLPClassifier()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

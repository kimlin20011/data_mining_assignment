from __future__ import print_function
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

iris = pd.read_csv("iris_noise.csv", encoding="utf-8")
iris_X = iris.data
iris_y = iris.target

#print(iris_X[:2, :])
# print(iris_y)
X_train, X_test, y_train, y_test = train_test_split(
    iris_X, iris_y, test_size=0.3)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)  # 訓練
# print(knn.predict(X_test))
# print(y_test)
a = knn.score(X_test, y_test)
print(a)

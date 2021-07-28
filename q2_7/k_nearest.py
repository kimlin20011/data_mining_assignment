from __future__ import print_function
from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

iris = pd.read_csv("iris_noise.csv", encoding="utf-8")
# iris_X = iris.data
# iris_y = iris.target

#df = pd.read_csv("data.csv", encoding="utf-8")
# iris = load_iris()
features = list(iris.columns[:4])
# 創造 dummy variables （假資料）  將windy outlook轉換成 0，1
label_encoder = preprocessing.LabelEncoder()
encoded_class = label_encoder.fit_transform(iris["class"])
iris["class"] = encoded_class

iris_X = iris[features]
iris_y = iris["class"]

print(iris_X)
# print(iris_y)
X_train, X_test, y_train, y_test = train_test_split(
    iris_X, iris_y, test_size=0.3)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)  # 訓練
# print(knn.predict(X_test))
# print(y_test)
a = knn.score(X_test, y_test)
print(a)

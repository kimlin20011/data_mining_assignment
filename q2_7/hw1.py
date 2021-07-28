from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

iris = pd.read_csv("iris_noise.csv", encoding="utf-8")
features = list(iris.columns[:4])
# 創造 dummy variables （假資料）  將windy outlook轉換成 0，1
label_encoder = preprocessing.LabelEncoder()
encoded_class = label_encoder.fit_transform(iris["class"])
iris["class"] = encoded_class

iris_X = iris[features]
iris_y = iris["class"]
k_range = list(range(1, 100))
weight_options = ["uniform", "distance"]

param_grid = dict(n_neighbors=k_range, weights=weight_options)


knn = KNeighborsClassifier()

grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
grid.fit(iris_X, iris_y)

print("test_k_range: 1-100")
print("CV  = 10")
print("best accuracy:", grid.best_score_)
# print("best K\n:", grid.best_params_)
#best_params = grid.best_params_
print("best K:", grid.best_params_["n_neighbors"])
print(grid.best_estimator_)

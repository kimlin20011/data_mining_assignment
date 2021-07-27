#from sklearn.datasets import load_iris
import pandas as pd
from sklearn import preprocessing, tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("data.csv", encoding="utf-8")
# iris = load_iris()
features = list(df.columns[1:21])
# 創造 dummy variables （假資料）  將windy outlook轉換成 0，1
label_encoder = preprocessing.LabelEncoder()
encoded_play = label_encoder.fit_transform(df["class"])
df["class"] = encoded_play

data_X = df[features]
data_Y = df["class"]

# 切分訓練與測試資料
train_X, test_X, train_y, test_y = train_test_split(
    data_X, data_Y, test_size=0.2)

# 建立分類器
clf = tree.DecisionTreeClassifier()
iris_clf = clf.fit(train_X, train_y)

# 預測
test_y_predicted = iris_clf.predict(test_X)

# 績效
accuracy = metrics.accuracy_score(test_y, test_y_predicted)
print(accuracy)

plt.figure()
# draw graph
classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(data_X, data_Y)
tree.plot_tree(classifier)
plt.savefig("tree.jpeg")
plt.figure()

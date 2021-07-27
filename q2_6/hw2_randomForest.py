#from sklearn.datasets import load_iris
import pandas as pd
from sklearn import preprocessing, ensemble
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd

df = pd.read_csv("data.csv", encoding="utf-8")
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

# 建立 random forest 模型
forest = ensemble.RandomForestClassifier(n_estimators=100)
forest_fit = forest.fit(train_X, train_y)

# 預測
test_y_predicted = forest.predict(test_X)

# 績效
accuracy = metrics.accuracy_score(test_y, test_y_predicted)
print("Random Forest Classifier Accuracy", accuracy)

from sklearn import tree
import pandas as pd
import graphviz
df = pd.read_csv("data.csv", encoding="utf-8")
print(df)
features = list(df.columns[1:21])
X = df[features]
y = df["class"]


classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(X, y)
tree.plot_tree(classifier)

dot_data = tree.export_graphviz(classifier, out_file=None,
                                feature_names=features,
                                class_names=["不能打羽球", "可以打球"],
                                filled=True, rounded=True, leaves_parallel=True)
graph = graphviz.Source(dot_data)

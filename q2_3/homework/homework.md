添付したiris.csvを用いて、k-means法、k-means++法を用いてクラスタリングしなさい。
また、クロスバリデーションにより、各手法の精度を比較しなさい。締め切りは2週間後(7/12)です。

k-means++法は、以下のように初期クラスタ重心を選ぶとする。
1.初期クラスタ中心1つをランダムに1つ選択する。C1とする。
2.各インスタンスXiと最も近いクラスタ中心の距離をD(Xi)とし、確率D(Xi)/ΣD(Xk)に基づいてインスタンスからクラスタ中心Cnを選ぶ。
3.必要数のクラスタ中心が得られるまで、上記の作業を繰り返す。

データのクラス数は3であるので、クラスタ数はk=3とする。
データ数は150である。各クラスを5分割し、うち40個ずつを学習データとし、10個ずつをテストデータとする。
学習データとテストデータの分け方を5通り行い、クロスバリデーションにより精度の平均を求める。
精度は、テストデータが正しく分類された割合とする。

ツールやライブラリは何を使っても良い。使わなくても良い。
(使用したツール等が分かるようにしてください。）
----
Conduct clustering iris.csv attached this page by k-means and k-means++ methods.
Further, evaluate each method with the accuracy. Deadline is 7/12, 2 weeks later.
For initial centroid selection of k-means++ is following.
1. Choose 1 instance as the first centroid randomly.
2. Let D(Xi) as the distance from instance Xi to the nearest centroid. Then choose the new centroid based on the probability of D(Xi)/ΣD(Xk), where k=1...N, N is a number of instances.
3. Until enough number of centroids are obtained, repeat step 2.

The class of the data is 3, so number of clusters, k=3.
The number of instances is 150, divide into 5 groups for each class. Each group has 10 instances. Then, let 40 instances of each class as the training data, and rest of them are test data. Use cross-validation to evaluate accuracy, changing the combination of training data and test data for 5 patterns(changing test data for each 5 groups).
The accuracy is the ratio of number of instances correctly classified to all test instances.

Any tools and libraries are available. Of course, it is ok you don't use them.
(Please note the tools or libraries you used.)
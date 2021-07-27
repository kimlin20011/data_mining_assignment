from sklearn import datasets
from sklearn.cluster import KMeans
import numpy as np


def mode(a):
    counts = np.bincount(a)
    return np.argmax(counts)


def calc_acc(y_p, y):
    return sum(y_p == y)/y.shape[0]


if __name__ == '__main__':

    iris = datasets.load_iris()

    x = iris.get('data')
    y = iris.get('target')

    # Randomly divide training cluster and test cluster
    num = x.shape[0]  # data number
    ratio = 4/1  # ration= training data:test data
    num_test = int(num/(1+ratio))  # number of test data
    num_train = num - num_test  # number of training data
    index = np.arange(num)  # 產生樣本號 generate data index
    np.random.shuffle(index)  # 洗牌 shuffle
    x_test = x[index[:num_test], :]  # 取出洗牌后前 num_test 作为测试集
    y_test = y[index[:num_test]]
    x_train = x[index[num_test:], :]  # 剩余作为训练集 Remaining as training set
    y_train = y[index[num_test:]]

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(x_train)

    centers = kmeans.cluster_centers_
    for i in range(3):
        index = y_train == i
        p = kmeans.predict(x_train[index, :])
        # Find the actual category to be the category label pp corresponding to i // 求實際類別為 i 所對應的類別標號 pp
        pp = mode(p)
        kmeans.cluster_centers_[i] = centers[pp]  # 相應的調整類別標號，以正確預測

    y_test_pre = kmeans.predict(x_test)
    print("y_test_pre:")
    print(y_test_pre)
    print("y_test:")
    print(y_test)

    # calulate the accuracy
    acc = calc_acc(y_test_pre, y_test)
    print('the accuracy is', acc)  # Show prediction accuracy
